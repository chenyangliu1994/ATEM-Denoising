from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.denoiser import ATEMDenoiser
from src.losses.tem_losses import RelativeLoss, HybridTEMLoss
from src.data.io import load_supervised_80col


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(path, scale):
    noisy, clean = load_supervised_80col(path, dtype=np.float32)
    X = torch.tensor(noisy * scale, dtype=torch.float32)
    y = torch.tensor(clean * scale, dtype=torch.float32)
    return TensorDataset(X, y)


@torch.no_grad()
def evaluate_loader(model, loader, device, lambda_nrmse, eps_relative_scaled, eps_nrmse):
    model.eval()
    rel_sum = 0.0
    sq_err_sum = 0.0
    target_sq_sum = 0.0
    count = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
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

    relative = rel_sum / count
    nrmse = np.sqrt(sq_err_sum / max(target_sq_sum, eps_nrmse))
    hybrid = relative + lambda_nrmse * nrmse
    snr = np.inf if sq_err_sum == 0 else 10.0 * np.log10(target_sq_sum / sq_err_sum)

    return {
        "relative": float(relative),
        "nrmse": float(nrmse),
        "hybrid": float(hybrid),
        "snr": float(snr),
    }


def save_checkpoint(path, model, metadata):
    payload = dict(metadata)
    payload["model_state_dict"] = model.state_dict()
    torch.save(payload, path)


def train_rh(train_file, val_file, out_dir, scenario_id, group, cfg):
    """
    Exact two-stage R-H protocol used by the controlled A/B experiments.

    Stage 1:
        BCG fixed at identity.
        Train encoder-decoder using Relative Loss.
        Select checkpoint by validation Hybrid.

    Stage 2:
        Restore best Stage-1 checkpoint.
        Freeze encoder-decoder.
        Train BCG using Relative + 2*NRMSE + gate regularization.
        Select checkpoint by validation Hybrid.
    """
    seed = int(cfg["seed"])
    dcfg = cfg["denoiser"]

    input_dim = int(dcfg["input_dim"])
    hidden_dim = int(dcfg["hidden_dim"])
    gate_hidden_dim = int(dcfg["gate_hidden_dim"])
    max_delta = float(dcfg["max_gate_delta"])
    batch_size = int(dcfg["batch_size"])
    scale = float(dcfg["scale"])

    stage1_epochs = int(dcfg["stage1_epochs"])
    stage1_lr = float(dcfg["stage1_lr"])
    stage2_epochs = int(dcfg["stage2_epochs"])
    stage2_lr = float(dcfg["stage2_lr"])

    lambda_nrmse = float(dcfg["lambda_nrmse"])
    gate_reg_weight = float(dcfg["gate_regularization"])
    grad_clip = float(dcfg["stage2_grad_clip"])
    eps_rel = float(dcfg["eps_relative_scaled"])
    eps_nrmse = float(dcfg["eps_nrmse"])
    min_improvement = float(dcfg["min_improvement"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_model_out = out_dir / "stage1_best.pth"
    final_model_out = out_dir / "final_RH_model.pth"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = make_dataset(train_file, scale)
    val_dataset = make_dataset(val_file, scale)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = ATEMDenoiser(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        gate_hidden_dim=gate_hidden_dim,
        max_delta=max_delta,
    ).to(device)

    # ---------------- Stage 1 ----------------
    for p in model.gate.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.decoder.parameters():
        p.requires_grad = True

    stage1_loss_fn = RelativeLoss(eps_relative_scaled=eps_rel)
    optimizer1 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=stage1_lr,
    )

    best1_hybrid = np.inf
    best1_epoch = 0
    best1_val = None
    history1 = []

    for epoch in range(1, stage1_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer1.zero_grad()
            pred = model(X)
            loss = stage1_loss_fn(pred, y)
            loss.backward()
            optimizer1.step()

            train_loss_sum += float(loss.item())
            train_batches += 1

        train_relative = train_loss_sum / train_batches
        val = evaluate_loader(
            model, val_loader, device,
            lambda_nrmse=lambda_nrmse,
            eps_relative_scaled=eps_rel,
            eps_nrmse=eps_nrmse,
        )

        history1.append({
            "epoch": epoch,
            "train_relative": train_relative,
            "val_relative": val["relative"],
            "val_nrmse": val["nrmse"],
            "val_hybrid": val["hybrid"],
            "val_snr_db": val["snr"],
        })

        print(
            f"Stage1 [{epoch:03d}/{stage1_epochs}] "
            f"TrainRel={train_relative:.6f} "
            f"ValRel={val['relative']:.6f} "
            f"ValNRMSE={val['nrmse']:.6f} "
            f"ValHybrid={val['hybrid']:.6f} "
            f"ValSNR={val['snr']:.3f} dB"
        )

        if val["hybrid"] < best1_hybrid - min_improvement:
            best1_hybrid = val["hybrid"]
            best1_epoch = epoch
            best1_val = dict(val)

            save_checkpoint(
                stage1_model_out,
                model,
                {
                    "scenario": scenario_id,
                    "group": group,
                    "stage": 1,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "gate_hidden_dim": gate_hidden_dim,
                    "max_delta": max_delta,
                    "scale": scale,
                    "seed": seed,
                    "lambda_nrmse": lambda_nrmse,
                    "stage1_epoch": epoch,
                    "stage1_val_relative": val["relative"],
                    "stage1_val_nrmse": val["nrmse"],
                    "stage1_val_hybrid": val["hybrid"],
                    "stage1_val_snr": val["snr"],
                    "stage1_selection": "validation_hybrid",
                    "validation_source": "explicit_val_real_only.dat",
                    "test_used_in_training": False,
                    "training_mode": "RH_two_stage_explicit_validation",
                },
            )

    pd.DataFrame(history1).to_csv(
        out_dir / "stage1_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ckpt1 = torch.load(stage1_model_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt1["model_state_dict"])

    # ---------------- Stage 2 ----------------
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.gate.parameters():
        p.requires_grad = True

    stage2_loss_fn = HybridTEMLoss(
        lambda_nrmse=lambda_nrmse,
        eps_relative_scaled=eps_rel,
        eps_nrmse=eps_nrmse,
    )
    optimizer2 = optim.Adam(model.gate.parameters(), lr=stage2_lr)

    initial_val = evaluate_loader(
        model, val_loader, device,
        lambda_nrmse=lambda_nrmse,
        eps_relative_scaled=eps_rel,
        eps_nrmse=eps_nrmse,
    )

    best2_hybrid = initial_val["hybrid"]
    best2_epoch = 0
    best2_val = dict(initial_val)
    history2 = []

    base_meta = {
        "scenario": scenario_id,
        "group": group,
        "stage": 2,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "gate_hidden_dim": gate_hidden_dim,
        "max_delta": max_delta,
        "scale": scale,
        "seed": seed,
        "lambda_nrmse": lambda_nrmse,
        "stage1_epoch": best1_epoch,
        "stage1_val_relative": best1_val["relative"],
        "stage1_val_nrmse": best1_val["nrmse"],
        "stage1_val_hybrid": best1_val["hybrid"],
        "stage1_val_snr": best1_val["snr"],
        "stage1_selection": "validation_hybrid",
        "stage2_selection": "validation_hybrid",
        "validation_source": "explicit_val_real_only.dat",
        "test_used_in_training": False,
        "training_mode": "RH_two_stage_explicit_validation",
    }

    save_checkpoint(
        final_model_out,
        model,
        {
            **base_meta,
            "stage2_epoch": 0,
            "stage2_val_relative": initial_val["relative"],
            "stage2_val_nrmse": initial_val["nrmse"],
            "stage2_val_hybrid": initial_val["hybrid"],
            "stage2_val_snr": initial_val["snr"],
        },
    )

    for epoch in range(1, stage2_epochs + 1):
        model.train()

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer2.zero_grad()

            pred, gates = model(X, return_gates=True)
            total, _, _ = stage2_loss_fn(pred, y)
            gate_reg = torch.mean((gates - 1.0) ** 2)
            loss = total + gate_reg_weight * gate_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.gate.parameters(), grad_clip)
            optimizer2.step()

        val = evaluate_loader(
            model, val_loader, device,
            lambda_nrmse=lambda_nrmse,
            eps_relative_scaled=eps_rel,
            eps_nrmse=eps_nrmse,
        )

        model.eval()
        with torch.no_grad():
            X0, _ = next(iter(val_loader))
            _, gates = model(X0.to(device), return_gates=True)
            gate_min = float(gates.min().item())
            gate_mean = float(gates.mean().item())
            gate_max = float(gates.max().item())

        history2.append({
            "epoch": epoch,
            "val_relative": val["relative"],
            "val_nrmse": val["nrmse"],
            "val_hybrid": val["hybrid"],
            "val_snr_db": val["snr"],
            "gate_min": gate_min,
            "gate_mean": gate_mean,
            "gate_max": gate_max,
        })

        print(
            f"Stage2 [{epoch:02d}/{stage2_epochs}] "
            f"ValHybrid={val['hybrid']:.6f} "
            f"ValRel={val['relative']:.6f} "
            f"ValNRMSE={val['nrmse']:.6f} "
            f"ValSNR={val['snr']:.3f} dB "
            f"Gate={gate_min:.4f}/{gate_mean:.4f}/{gate_max:.4f}"
        )

        if val["hybrid"] < best2_hybrid - min_improvement:
            best2_hybrid = val["hybrid"]
            best2_epoch = epoch
            best2_val = dict(val)
            save_checkpoint(
                final_model_out,
                model,
                {
                    **base_meta,
                    "stage2_epoch": epoch,
                    "stage2_val_relative": val["relative"],
                    "stage2_val_nrmse": val["nrmse"],
                    "stage2_val_hybrid": val["hybrid"],
                    "stage2_val_snr": val["snr"],
                },
            )

    pd.DataFrame(history2).to_csv(
        out_dir / "stage2_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([{
        "Scenario": scenario_id,
        "Group": group,
        "Seed": seed,
        "Training_Rows": len(train_dataset),
        "Validation_Rows": len(val_dataset),
        "Test_Used_In_Training": False,
        "Stage1_Best_Epoch": best1_epoch,
        "Stage1_Best_Val_Relative": best1_val["relative"],
        "Stage1_Best_Val_NRMSE": best1_val["nrmse"],
        "Stage1_Best_Val_Hybrid": best1_val["hybrid"],
        "Stage1_Best_Val_SNR_dB": best1_val["snr"],
        "Stage2_Best_Epoch": best2_epoch,
        "Stage2_Best_Val_Relative": best2_val["relative"],
        "Stage2_Best_Val_NRMSE": best2_val["nrmse"],
        "Stage2_Best_Val_Hybrid": best2_val["hybrid"],
        "Stage2_Best_Val_SNR_dB": best2_val["snr"],
    }])

    summary.to_csv(
        out_dir / "training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return final_model_out, summary.iloc[0].to_dict()
