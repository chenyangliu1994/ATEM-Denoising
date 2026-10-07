# -*- coding: utf-8 -*-
"""
18 - E3 WGAN ablation training: Full H512 FC-ED + SRCG WITHOUT WGAN augmentation.

This script TRAINs only. It never opens or references the independent test set.

Stage 1
-------
H512 FC-ED, 220 epochs, Relative Error loss.
Best checkpoint selected by validation Hybrid = Relative + 2 * NRMSE.

Stage 2
-------
Freeze the best Stage-1 FC-ED and train SRCG for 100 epochs.

    z = FCED(x)
    alpha = SRCG(x, z, z-x)
    y_hat = x + alpha * (z - x)

Training objective:
    Hybrid + 0.05 * SmoothL1(alpha, alpha_star)

where alpha_star is used only during supervised training:
    alpha_star = clip(((s-x)*(z-x))/((z-x)^2 + eps), 0, 1)

Expected files
--------------
E3_城市道路旁_16次叠加/train_B_real_resampled_sameN.dat
E3_城市道路旁_16次叠加/val_real_only.dat

Outputs
-------
E3_城市道路旁_16次叠加/PAPERFINAL_E3_NO_WGAN_SAMEN/
    FCED_stage1_H512_E220_best.pth
    FCED_H512_SRCG_continuous_best.pth
    stage1_history.csv
    stage2_history.csv
    training_efficiency.csv
    training_metadata.json
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. Paths -- no test path exists in this training script
# ============================================================
SCENARIO_DIR = Path(r"E3_城市道路旁_16次叠加")
TRAIN_DAT = SCENARIO_DIR / "train_B_real_resampled_sameN.dat"
VAL_DAT = SCENARIO_DIR / "val_real_only.dat"

OUT_DIR = SCENARIO_DIR / "PAPERFINAL_E3_NO_WGAN_SAMEN"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1_BEST = OUT_DIR / "FCED_stage1_H512_E220_best.pth"
STAGE2_BEST = OUT_DIR / "FCED_H512_SRCG_continuous_best.pth"
STAGE1_HISTORY = OUT_DIR / "stage1_history.csv"
STAGE2_HISTORY = OUT_DIR / "stage2_history.csv"
TRAINING_EFFICIENCY = OUT_DIR / "training_efficiency.csv"
TRAINING_METADATA = OUT_DIR / "training_metadata.json"


# ============================================================
# 1. Frozen formal protocol
# ============================================================
INPUT_DIM = 40
HIDDEN_DIM = 512
SRCG_HIDDEN = 64

BATCH_SIZE = 32
SCALE = 1e19
SEED = 2024

STAGE1_EPOCHS = 220
STAGE1_LR = 1e-3

STAGE2_EPOCHS = 100
STAGE2_LR = 1e-5

LAMBDA_NRMSE = 2.0
CONTINUOUS_TEACHER_WEIGHT = 0.05

EPS_REL = 1.0
EPS_NRMSE = 1e-12
EPS_ALPHA_TRAIN = 1e-12
MIN_IMPROVEMENT = 1e-7

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 2. Utilities
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_model_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return float(path.stat().st_size / (1024.0 ** 2))


# ============================================================
# 3. Data
# ============================================================
def load_project_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Formal E2 .dat format:
        whitespace-delimited, no header, N x 80
        columns 1-40  : noisy ATEM response
        columns 41-80 : theoretical clean response

    E2 uses held-out measured noise for validation/test contamination.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 80:
        raise ValueError(f"{path}: expected 80 columns, got {df.shape[1]}")

    arr = df.to_numpy(dtype=np.float32, copy=True)
    del df

    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: NaN/Inf detected")

    return arr[:, :40], arr[:, 40:80]


# ============================================================
# 4. Models
# ============================================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x


class FCED(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 512):
        super().__init__()
        self.encoder = NeuralNetwork(input_dim, hidden_dim, hidden_dim)
        self.decoder = NeuralNetwork(hidden_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SelectiveResidualCorrectionGate(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        # sigmoid(2.1972246) ~= 0.90; identical to the frozen formal SRCG protocol.
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.1972246)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-12)
        x_n = x / (rms + 1e-12)
        z_n = z / (rms + 1e-12)
        d_n = (z - x) / (rms + 1e-12)
        feat = torch.cat([x_n, z_n, d_n], dim=1)
        h = torch.relu(self.fc1(feat))
        return torch.sigmoid(self.fc2(h))


class FCED_SRCG(nn.Module):
    def __init__(self, backbone: FCED, input_dim: int = 40, srcg_hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.srcg = SelectiveResidualCorrectionGate(input_dim, srcg_hidden)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        z = self.backbone(x)
        alpha = self.srcg(x, z)
        pred = x + alpha * (z - x)
        if return_aux:
            return pred, z, alpha
        return pred


# ============================================================
# 5. Losses / validation
# ============================================================
def relative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = torch.where(
        torch.abs(target) < EPS_REL,
        torch.full_like(target, EPS_REL),
        torch.abs(target),
    )
    return torch.mean(torch.abs(pred - target) / denom)


def hybrid_components(pred: torch.Tensor, target: torch.Tensor):
    err = pred - target
    rel = relative_loss(pred, target)
    nrmse = torch.sqrt(
        torch.mean(err ** 2) / (torch.mean(target ** 2) + EPS_NRMSE)
    )
    return rel + LAMBDA_NRMSE * nrmse, rel, nrmse


@torch.no_grad()
def continuous_alpha_star_train(
    x: torch.Tensor, z: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    d = z - x
    alpha_star = ((target - x) * d) / (d * d + EPS_ALPHA_TRAIN)
    return torch.clamp(alpha_star, 0.0, 1.0)


@torch.no_grad()
def evaluate_fced(model: FCED, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    rel_sum = 0.0
    sq_err_sum = 0.0
    target_sq_sum = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        err = pred - y
        denom = torch.where(
            torch.abs(y) < EPS_REL,
            torch.full_like(y, EPS_REL),
            torch.abs(y),
        )
        rel_sum += torch.sum(torch.abs(err) / denom).item()
        sq_err_sum += torch.sum(err ** 2).item()
        target_sq_sum += torch.sum(y ** 2).item()
        count += y.numel()

    rel = rel_sum / count
    nrmse = math.sqrt(sq_err_sum / max(target_sq_sum, EPS_NRMSE))
    hybrid = rel + LAMBDA_NRMSE * nrmse
    snr = np.inf if sq_err_sum == 0 else 10.0 * math.log10(target_sq_sum / sq_err_sum)
    return {
        "relative": float(rel),
        "nrmse": float(nrmse),
        "hybrid": float(hybrid),
        "snr": float(snr),
    }


@torch.no_grad()
def evaluate_srcg(
    model: FCED_SRCG, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred, z, alpha = model(x, return_aux=True)
        err = pred - y
        alpha_star = continuous_alpha_star_train(x, z, y)

        denom = torch.where(
            torch.abs(y) < EPS_REL,
            torch.full_like(y, EPS_REL),
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

        teacher_sum += alpha_star.sum().item()
        diff = alpha - alpha_star
        teacher_abs_sum += torch.sum(torch.abs(diff)).item()
        teacher_sq_sum += torch.sum(diff ** 2).item()

    rel = rel_sum / count
    nrmse = math.sqrt(sq_err_sum / max(target_sq_sum, EPS_NRMSE))
    hybrid = rel + LAMBDA_NRMSE * nrmse
    snr = np.inf if sq_err_sum == 0 else 10.0 * math.log10(target_sq_sum / sq_err_sum)

    return {
        "relative": float(rel),
        "nrmse": float(nrmse),
        "hybrid": float(hybrid),
        "snr": float(snr),
        "alpha_min": float(alpha_min),
        "alpha_mean": float(alpha_sum / alpha_count),
        "alpha_max": float(alpha_max),
        "continuous_teacher_mean": float(teacher_sum / alpha_count),
        "alpha_teacher_mae": float(teacher_abs_sum / alpha_count),
        "alpha_teacher_rmse": float(math.sqrt(teacher_sq_sum / alpha_count)),
    }


# ============================================================
# 6. Checkpoints
# ============================================================
def save_stage1_checkpoint(
    path: Path,
    model: FCED,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "scale": SCALE,
            "seed": SEED,
            "stage1_epoch": int(epoch),
            "stage1_val_metrics": metrics,
            "training_data": str(TRAIN_DAT),
            "validation_data": str(VAL_DAT),
            "scenario": "E3_urban_roadside_16stack_measured_noise",
            "training_mode": "PAPERFINAL_E3_NO_WGAN_SAMEN_FCED_H512_stage1",
        },
        path,
    )


def save_stage2_checkpoint(
    path: Path,
    model: FCED_SRCG,
    epoch: int,
    metrics: Dict[str, float],
    stage1_epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "srcg_hidden": SRCG_HIDDEN,
            "scale": SCALE,
            "seed": SEED,
            "stage1_model": str(STAGE1_BEST),
            "stage1_epoch": int(stage1_epoch),
            "stage2_epoch": int(epoch),
            "stage2_val_metrics": metrics,
            "continuous_teacher_weight": CONTINUOUS_TEACHER_WEIGHT,
            "lambda_nrmse": LAMBDA_NRMSE,
            "training_data": str(TRAIN_DAT),
            "validation_data": str(VAL_DAT),
            "scenario": "E3_urban_roadside_16stack_measured_noise",
            "training_mode": "PAPERFINAL_E3_NO_WGAN_SAMEN_FCED_H512_SRCG",
            "formula": "y_hat = x + alpha * (z - x)",
            "continuous_teacher_formula":
                "clip(((s-x)*(z-x))/((z-x)^2+eps),0,1)",
        },
        path,
    )


# ============================================================
# 7. Main training
# ============================================================
def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("PAPERFINAL ABLATION TRAINING ONLY: E3 NO-WGAN SAME-N, H512 FC-ED + SRCG")
    print("=" * 88)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Scenario:", SCENARIO_DIR)
    print("Training file:", TRAIN_DAT)
    print("Validation file:", VAL_DAT)
    print("Output directory:", OUT_DIR)
    print("Independent test data accessed by this script: NO")
    print("Ablation variable: WGAN augmentation = OFF; measured-noise resampling = ON")
    print("Continuous teacher weight (kept as weak regularizer):", CONTINUOUS_TEACHER_WEIGHT)

    train_noisy, train_clean = load_project_dat(TRAIN_DAT)
    val_noisy, val_clean = load_project_dat(VAL_DAT)

    print("Optimization samples:", len(train_noisy))
    print("Validation samples:", len(val_noisy))

    train_dataset = TensorDataset(
        torch.from_numpy(train_noisy * SCALE),
        torch.from_numpy(train_clean * SCALE),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_noisy * SCALE),
        torch.from_numpy(val_clean * SCALE),
    )

    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ---------------- Stage 1 ----------------
    print("\n" + "=" * 88)
    print("STAGE 1: H512 FC-ED")
    print("Loss: Relative Error")
    print("Checkpoint: lowest validation Hybrid = Rel + 2*NRMSE")
    print("Epochs:", STAGE1_EPOCHS)
    print("=" * 88)

    backbone = FCED(INPUT_DIM, HIDDEN_DIM).to(device)
    print("FC-ED parameters:", count_parameters(backbone))
    optimizer1 = optim.Adam(backbone.parameters(), lr=STAGE1_LR)

    best_stage1_hybrid = np.inf
    best_stage1_epoch = 0
    best_stage1_metrics = None
    stage1_rows = []

    sync_cuda()
    stage1_t0 = time.perf_counter()

    for epoch in range(1, STAGE1_EPOCHS + 1):
        backbone.train()
        train_rel_sum = 0.0
        batches = 0
        epoch_t0 = time.perf_counter()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer1.zero_grad(set_to_none=True)
            pred = backbone(x)
            loss = relative_loss(pred, y)
            loss.backward()
            optimizer1.step()
            train_rel_sum += loss.item()
            batches += 1

        val = evaluate_fced(backbone, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_t0
        row = {
            "Epoch": epoch,
            "TrainRelative": train_rel_sum / max(batches, 1),
            "ValRelative": val["relative"],
            "ValNRMSE": val["nrmse"],
            "ValHybrid": val["hybrid"],
            "ValSNR_dB": val["snr"],
            "EpochSeconds": epoch_seconds,
        }
        stage1_rows.append(row)
        pd.DataFrame(stage1_rows).to_csv(STAGE1_HISTORY, index=False)

        print(
            f"Stage1 [{epoch:03d}/{STAGE1_EPOCHS}]  "
            f"TrainRel={row['TrainRelative']:.6f}  "
            f"ValRel={val['relative']:.6f}  "
            f"ValNRMSE={val['nrmse']:.6f}  "
            f"ValHybrid={val['hybrid']:.6f}  "
            f"ValSNR={val['snr']:.3f}dB  "
            f"Time={epoch_seconds:.1f}s"
        )

        if val["hybrid"] < best_stage1_hybrid - MIN_IMPROVEMENT:
            best_stage1_hybrid = val["hybrid"]
            best_stage1_epoch = epoch
            best_stage1_metrics = val.copy()
            save_stage1_checkpoint(
                STAGE1_BEST, backbone, epoch, val
            )
            print("  -> validation improved; Stage-1 best saved")

    sync_cuda()
    stage1_seconds = time.perf_counter() - stage1_t0

    print("\nStage 1 finished")
    print("Best epoch:", best_stage1_epoch)
    print("Best validation metrics:", best_stage1_metrics)
    print("Stage-1 wall time (s):", stage1_seconds)
    print("Saved:", STAGE1_BEST)

    # ---------------- Stage 2 ----------------
    stage1_ckpt = torch.load(STAGE1_BEST, map_location=device, weights_only=False)
    best_backbone = FCED(INPUT_DIM, HIDDEN_DIM).to(device)
    best_backbone.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
    for p in best_backbone.parameters():
        p.requires_grad = False
    best_backbone.eval()

    model = FCED_SRCG(best_backbone, INPUT_DIM, SRCG_HIDDEN).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.srcg.parameters():
        p.requires_grad = True

    print("\n" + "=" * 88)
    print("STAGE 2: SRCG + Continuous Correction-Degree Supervision")
    print("Backbone frozen: True")
    print("Loss = Hybrid + 0.05 * SmoothL1(alpha, alpha_star)")
    print("Checkpoint: lowest validation Hybrid")
    print("=" * 88)
    print("Frozen FC-ED parameters:", count_parameters(model.backbone))
    print("SRCG trainable parameters:", count_trainable_parameters(model.srcg))
    print("Total final parameters:", count_parameters(model))
    print("Frozen Stage-1 validation:", evaluate_fced(model.backbone, val_loader, device))

    stage2_generator = torch.Generator().manual_seed(SEED)
    train_loader_stage2 = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=stage2_generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    optimizer2 = optim.Adam(model.srcg.parameters(), lr=STAGE2_LR)
    best_stage2_hybrid = np.inf
    best_stage2_epoch = 0
    best_stage2_metrics = None
    stage2_rows = []

    sync_cuda()
    stage2_t0 = time.perf_counter()

    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        model.backbone.eval()
        train_obj_sum = 0.0
        train_hybrid_sum = 0.0
        train_teacher_sum = 0.0
        batches = 0
        epoch_t0 = time.perf_counter()

        for x, y in train_loader_stage2:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer2.zero_grad(set_to_none=True)

            pred, z, alpha = model(x, return_aux=True)
            hybrid, _, _ = hybrid_components(pred, y)
            with torch.no_grad():
                alpha_star = continuous_alpha_star_train(x, z, y)
            teacher_loss = F.smooth_l1_loss(alpha, alpha_star, reduction="mean")
            total = hybrid + CONTINUOUS_TEACHER_WEIGHT * teacher_loss
            total.backward()
            optimizer2.step()

            train_obj_sum += total.item()
            train_hybrid_sum += hybrid.item()
            train_teacher_sum += teacher_loss.item()
            batches += 1

        val = evaluate_srcg(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_t0
        row = {
            "Epoch": epoch,
            "TrainObjective": train_obj_sum / max(batches, 1),
            "TrainHybrid": train_hybrid_sum / max(batches, 1),
            "TrainTeacher": train_teacher_sum / max(batches, 1),
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
            "EpochSeconds": epoch_seconds,
        }
        stage2_rows.append(row)
        pd.DataFrame(stage2_rows).to_csv(STAGE2_HISTORY, index=False)

        print(
            f"SRCG [{epoch:02d}/{STAGE2_EPOCHS}]  "
            f"TrainObj={row['TrainObjective']:.6f}  "
            f"TrainHybrid={row['TrainHybrid']:.6f}  "
            f"TrainTeacher={row['TrainTeacher']:.6f}  "
            f"ValHybrid={val['hybrid']:.6f}  "
            f"ValRel={val['relative']:.6f}  "
            f"ValNRMSE={val['nrmse']:.6f}  "
            f"ValSNR={val['snr']:.3f}dB  "
            f"Alpha={val['alpha_min']:.4f}/{val['alpha_mean']:.4f}/{val['alpha_max']:.4f}  "
            f"TeacherMean={val['continuous_teacher_mean']:.4f}  "
            f"AlphaTeacherMAE={val['alpha_teacher_mae']:.4f}  "
            f"Time={epoch_seconds:.1f}s"
        )

        if val["hybrid"] < best_stage2_hybrid - MIN_IMPROVEMENT:
            best_stage2_hybrid = val["hybrid"]
            best_stage2_epoch = epoch
            best_stage2_metrics = val.copy()
            save_stage2_checkpoint(
                STAGE2_BEST, model, epoch, val, best_stage1_epoch
            )
            print("  -> validation improved; final SRCG best saved")

    sync_cuda()
    stage2_seconds = time.perf_counter() - stage2_t0

    print("\nStage 2 finished")
    print("Best epoch:", best_stage2_epoch)
    print("Best validation metrics:", best_stage2_metrics)
    print("Stage-2 wall time (s):", stage2_seconds)
    print("Saved:", STAGE2_BEST)

    pd.DataFrame([
        {
            "Stage": "Stage1_FCED_H512",
            "Seconds": stage1_seconds,
            "Hours": stage1_seconds / 3600.0,
            "BestEpoch": best_stage1_epoch,
        },
        {
            "Stage": "Stage2_SRCG",
            "Seconds": stage2_seconds,
            "Hours": stage2_seconds / 3600.0,
            "BestEpoch": best_stage2_epoch,
        },
    ]).to_csv(TRAINING_EFFICIENCY, index=False)

    metadata = {
        "experiment": "E3_no_WGAN_sameN_H512_FCED_SRCG_paperfinal_ablation_training",
        "date_finished_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": SEED,
        "scale": SCALE,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "srcg_hidden": SRCG_HIDDEN,
        "batch_size_train": BATCH_SIZE,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage1_lr": STAGE1_LR,
        "stage2_epochs": STAGE2_EPOCHS,
        "stage2_lr": STAGE2_LR,
        "lambda_nrmse": LAMBDA_NRMSE,
        "continuous_teacher_weight": CONTINUOUS_TEACHER_WEIGHT,
        "scenario": "E3_urban_roadside_16stack_measured_noise",
        "training_dat": str(TRAIN_DAT),
        "validation_dat": str(VAL_DAT),
        "n_optimization": int(len(train_noisy)),
        "n_validation": int(len(val_noisy)),
        "stage1_best_epoch": int(best_stage1_epoch),
        "stage1_best_validation": best_stage1_metrics,
        "stage2_best_epoch": int(best_stage2_epoch),
        "stage2_best_validation": best_stage2_metrics,
        "fced_parameters": count_parameters(model.backbone),
        "srcg_parameters": count_parameters(model.srcg),
        "total_parameters": count_parameters(model),
        "stage1_model_size_MB": safe_model_size_mb(STAGE1_BEST),
        "stage2_model_size_MB": safe_model_size_mb(STAGE2_BEST),
        "stage1_train_seconds": stage1_seconds,
        "stage2_train_seconds": stage2_seconds,
        "test_data_accessed": False,
        "important_note": (
            "This script never opens the independent test set. "
            "Run 19_predict_E3_PAPERFINAL_NO_WGAN_SAMEN_v2.py only after training is complete."
        ),
    }
    save_json(TRAINING_METADATA, metadata)

    print("\n" + "=" * 88)
    print("FORMAL TRAINING FINISHED -- NO TEST DATA WERE USED")
    print("=" * 88)
    print("Best Stage-1 epoch:", best_stage1_epoch)
    print("Best Stage-2 epoch:", best_stage2_epoch)
    print("Stage-1 time (h):", stage1_seconds / 3600.0)
    print("Stage-2 time (h):", stage2_seconds / 3600.0)
    print("\nSaved:")
    for p in [
        STAGE1_BEST,
        STAGE2_BEST,
        STAGE1_HISTORY,
        STAGE2_HISTORY,
        TRAINING_EFFICIENCY,
        TRAINING_METADATA,
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
