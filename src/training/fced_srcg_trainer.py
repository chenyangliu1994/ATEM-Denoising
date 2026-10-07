from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.denoiser import FCED, FCED_SRCG
from src.losses.tem_losses import relative_loss, hybrid_components, continuous_alpha_star
from src.data.io import load_supervised_80col


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_fced(model, loader, device, lambda_nrmse, eps_relative_scaled, eps_nrmse):
    model.eval()
    rel_sum = 0.0
    sq_err_sum = 0.0
    target_sq_sum = 0.0
    count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        err = pred - y
        denom = torch.where(
            torch.abs(y) < eps_relative_scaled,
            torch.full_like(y, eps_relative_scaled),
            torch.abs(y),
        )
        rel_sum += torch.sum(torch.abs(err) / denom).item()
        sq_err_sum += torch.sum(err ** 2).item()
        target_sq_sum += torch.sum(y ** 2).item()
        count += y.numel()

    rel = rel_sum / count
    nrmse = math.sqrt(sq_err_sum / max(target_sq_sum, eps_nrmse))
    return {
        "relative": float(rel),
        "nrmse": float(nrmse),
        "hybrid": float(rel + lambda_nrmse * nrmse),
        "snr": float(np.inf if sq_err_sum == 0 else 10.0 * math.log10(target_sq_sum / sq_err_sum)),
    }


@torch.no_grad()
def evaluate_srcg(model, loader, device, lambda_nrmse, eps_relative_scaled, eps_nrmse, eps_alpha):
    model.eval()
    model.backbone.eval()

    rel_sum = 0.0
    sq_err_sum = 0.0
    target_sq_sum = 0.0
    count = 0
    alpha_sum = 0.0
    alpha_count = 0
    alpha_min = np.inf
    alpha_max = -np.inf
    teacher_sum = 0.0
    teacher_abs_sum = 0.0
    teacher_sq_sum = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, z, alpha = model(x, return_aux=True)
        err = pred - y
        a_star = continuous_alpha_star(x, z, y, eps_alpha)

        denom = torch.where(
            torch.abs(y) < eps_relative_scaled,
            torch.full_like(y, eps_relative_scaled),
            torch.abs(y),
        )
        rel_sum += torch.sum(torch.abs(err) / denom).item()
        sq_err_sum += torch.sum(err ** 2).item()
        target_sq_sum += torch.sum(y ** 2).item()
        count += y.numel()

        alpha_sum += alpha.sum().item()
        alpha_count += alpha.numel()
        alpha_min = min(alpha_min, alpha.min().item())
        alpha_max = max(alpha_max, alpha.max().item())

        teacher_sum += a_star.sum().item()
        diff = alpha - a_star
        teacher_abs_sum += torch.sum(torch.abs(diff)).item()
        teacher_sq_sum += torch.sum(diff ** 2).item()

    rel = rel_sum / count
    nrmse = math.sqrt(sq_err_sum / max(target_sq_sum, eps_nrmse))
    return {
        "relative": float(rel),
        "nrmse": float(nrmse),
        "hybrid": float(rel + lambda_nrmse * nrmse),
        "snr": float(np.inf if sq_err_sum == 0 else 10.0 * math.log10(target_sq_sum / sq_err_sum)),
        "alpha_min": float(alpha_min),
        "alpha_mean": float(alpha_sum / alpha_count),
        "alpha_max": float(alpha_max),
        "continuous_teacher_mean": float(teacher_sum / alpha_count),
        "alpha_teacher_mae": float(teacher_abs_sum / alpha_count),
        "alpha_teacher_rmse": float(math.sqrt(teacher_sq_sum / alpha_count)),
    }


def train_fced_srcg(train_file, val_file, out_dir, scenario_id, group, cfg):
    """
    Paper-final FC-ED + SRCG protocol.

    Stage 1:
      H512 FC-ED, 220 epochs, Relative Error loss.
      Select checkpoint by validation Hybrid = Relative + 2*NRMSE.

    Stage 2:
      Restore and freeze best Stage-1 FC-ED.
      Train SRCG64 for 100 epochs.
      Objective = Hybrid + 0.05*SmoothL1(alpha, alpha_star).
      Select checkpoint by validation Hybrid.

    Independent test data are not referenced here.
    """
    seed = int(cfg["seed"])
    dcfg = cfg["denoiser"]

    input_dim = int(dcfg["input_dim"])
    hidden_dim = int(dcfg["hidden_dim"])
    srcg_hidden = int(dcfg["srcg_hidden_dim"])
    batch_size = int(dcfg["batch_size"])
    scale = float(dcfg["scale"])
    stage1_epochs = int(dcfg["stage1_epochs"])
    stage1_lr = float(dcfg["stage1_lr"])
    stage2_epochs = int(dcfg["stage2_epochs"])
    stage2_lr = float(dcfg["stage2_lr"])
    lambda_nrmse = float(dcfg["lambda_nrmse"])
    teacher_weight = float(dcfg["continuous_teacher_weight"])
    eps_rel = float(dcfg["eps_relative_scaled"])
    eps_nrmse = float(dcfg["eps_nrmse"])
    eps_alpha = float(dcfg["eps_alpha"])
    min_improvement = float(dcfg["min_improvement"])

    # Paper-final guard
    assert hidden_dim == 512
    assert srcg_hidden == 64
    assert stage1_epochs == 220
    assert stage2_epochs == 100

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_best = out_dir / "FCED_stage1_H512_E220_best.pth"
    stage2_best = out_dir / "FCED_H512_SRCG_continuous_best.pth"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_noisy, train_clean = load_supervised_80col(train_file)
    val_noisy, val_clean = load_supervised_80col(val_file)

    train_ds = TensorDataset(
        torch.from_numpy((train_noisy * scale).astype(np.float32)),
        torch.from_numpy((train_clean * scale).astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy((val_noisy * scale).astype(np.float32)),
        torch.from_numpy((val_clean * scale).astype(np.float32)),
    )

    g1 = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g1)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Stage 1
    backbone = FCED(input_dim, hidden_dim).to(device)
    opt1 = optim.Adam(backbone.parameters(), lr=stage1_lr)
    best_h = np.inf
    best_epoch = 0
    best_metrics = None
    rows1 = []

    for epoch in range(1, stage1_epochs + 1):
        backbone.train()
        train_sum = 0.0
        nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt1.zero_grad(set_to_none=True)
            pred = backbone(x)
            loss = relative_loss(pred, y, eps_rel)
            loss.backward()
            opt1.step()
            train_sum += loss.item()
            nb += 1

        val = evaluate_fced(backbone, val_loader, device, lambda_nrmse, eps_rel, eps_nrmse)
        rows1.append({"Epoch": epoch, "TrainRelative": train_sum/max(nb,1),
                      "ValRelative": val["relative"], "ValNRMSE": val["nrmse"],
                      "ValHybrid": val["hybrid"], "ValSNR_dB": val["snr"]})

        if val["hybrid"] < best_h - min_improvement:
            best_h = val["hybrid"]
            best_epoch = epoch
            best_metrics = dict(val)
            torch.save({
                "model_state_dict": backbone.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "scale": scale,
                "seed": seed,
                "stage1_epoch": epoch,
                "stage1_val_metrics": val,
                "training_data": str(train_file),
                "validation_data": str(val_file),
                "scenario": scenario_id,
                "group": group,
                "test_data_accessed": False,
            }, stage1_best)

    pd.DataFrame(rows1).to_csv(out_dir / "stage1_history.csv", index=False)

    # Stage 2
    ckpt1 = torch.load(stage1_best, map_location=device, weights_only=False)
    best_backbone = FCED(input_dim, hidden_dim).to(device)
    best_backbone.load_state_dict(ckpt1["model_state_dict"], strict=True)
    for p in best_backbone.parameters():
        p.requires_grad = False
    best_backbone.eval()

    model = FCED_SRCG(best_backbone, input_dim, srcg_hidden).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.srcg.parameters():
        p.requires_grad = True

    g2 = torch.Generator().manual_seed(seed)
    train_loader2 = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g2)
    opt2 = optim.Adam(model.srcg.parameters(), lr=stage2_lr)

    best_h2 = np.inf
    best_epoch2 = 0
    best_metrics2 = None
    rows2 = []

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        model.backbone.eval()
        total_sum = hybrid_sum = teacher_sum = 0.0
        nb = 0

        for x, y in train_loader2:
            x, y = x.to(device), y.to(device)
            opt2.zero_grad(set_to_none=True)
            pred, z, alpha = model(x, return_aux=True)
            hybrid, _, _ = hybrid_components(pred, y, lambda_nrmse, eps_rel, eps_nrmse)
            with torch.no_grad():
                a_star = continuous_alpha_star(x, z, y, eps_alpha)
            t_loss = F.smooth_l1_loss(alpha, a_star, reduction="mean")
            total = hybrid + teacher_weight * t_loss
            total.backward()
            opt2.step()

            total_sum += total.item()
            hybrid_sum += hybrid.item()
            teacher_sum += t_loss.item()
            nb += 1

        val = evaluate_srcg(
            model, val_loader, device, lambda_nrmse, eps_rel, eps_nrmse, eps_alpha
        )
        rows2.append({
            "Epoch": epoch,
            "TrainObjective": total_sum/max(nb,1),
            "TrainHybrid": hybrid_sum/max(nb,1),
            "TrainTeacher": teacher_sum/max(nb,1),
            "ValRelative": val["relative"],
            "ValNRMSE": val["nrmse"],
            "ValHybrid": val["hybrid"],
            "ValSNR_dB": val["snr"],
            "AlphaMin": val["alpha_min"],
            "AlphaMean": val["alpha_mean"],
            "AlphaMax": val["alpha_max"],
            "TeacherMean": val["continuous_teacher_mean"],
            "AlphaTeacherMAE": val["alpha_teacher_mae"],
            "AlphaTeacherRMSE": val["alpha_teacher_rmse"],
        })

        if val["hybrid"] < best_h2 - min_improvement:
            best_h2 = val["hybrid"]
            best_epoch2 = epoch
            best_metrics2 = dict(val)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "srcg_hidden": srcg_hidden,
                "scale": scale,
                "seed": seed,
                "stage1_model": str(stage1_best),
                "stage1_epoch": best_epoch,
                "stage2_epoch": epoch,
                "stage2_val_metrics": val,
                "continuous_teacher_weight": teacher_weight,
                "lambda_nrmse": lambda_nrmse,
                "training_data": str(train_file),
                "validation_data": str(val_file),
                "scenario": scenario_id,
                "group": group,
                "formula": "y_hat = x + alpha * (z - x)",
                "continuous_teacher_formula": "clip(((s-x)*(z-x))/((z-x)^2+eps),0,1)",
                "test_data_accessed": False,
            }, stage2_best)

    pd.DataFrame(rows2).to_csv(out_dir / "stage2_history.csv", index=False)

    metadata = {
        "scenario": scenario_id,
        "group": group,
        "seed": seed,
        "scale": scale,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "srcg_hidden": srcg_hidden,
        "batch_size": batch_size,
        "stage1_epochs": stage1_epochs,
        "stage1_lr": stage1_lr,
        "stage2_epochs": stage2_epochs,
        "stage2_lr": stage2_lr,
        "lambda_nrmse": lambda_nrmse,
        "continuous_teacher_weight": teacher_weight,
        "stage1_best_epoch": best_epoch,
        "stage1_best_validation": best_metrics,
        "stage2_best_epoch": best_epoch2,
        "stage2_best_validation": best_metrics2,
        "test_data_accessed": False,
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata
