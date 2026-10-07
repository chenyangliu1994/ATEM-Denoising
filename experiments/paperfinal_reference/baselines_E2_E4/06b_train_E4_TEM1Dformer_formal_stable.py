# -*- coding: utf-8 -*-
"""
06 - Paper-guided TEM1Dformer reimplementation, formal training for E4 measured-noise-contaminated ATEM.

Important reproducibility note
------------------------------
The original TEM1Dformer paper (IEEE Sensors Journal, 2024) does not provide
official source code and does not fully specify every internal dimension.

This implementation follows the paper-supported structure:
  - 1-D input
  - two Conv1d shallow feature layers
  - one ViT/Transformer encoder block
  - Q/K/V formed by pointwise + depthwise 1-D convolutions
  - no BatchNorm
  - ten residual blocks, each with two Conv1d + ReLU operations
  - two Conv1d dimension-recovery layers
  - residual/noise learning
  - multitask loss: 0.8*MSE + 0.2*MAE
  - Adam, lr=1e-4, batch size=32
  - MultiStepLR milestones [40, 60, 80], gamma=0.1

The original paper reports 77,921 trainable parameters for TEM1Dformer.
Because several hidden dimensions are not explicitly specified, this
paper-guided implementation uses:
  feature channels = 32
  attention heads = 4
  FFN width = 64
which keeps the model close to the reported scale while preserving the
published architecture.

Fair-comparison protocol
------------------------
- Same E4 train_B_wgan_aug.dat / val_real_only.dat protocol as FC-ED / SRCG / TEMDnet.
- Validation is the fixed held-out val_real_only.dat set.
- Independent test_real_only.dat is NEVER read here.
- Checkpoint selected by minimum validation NRMSE.
- Train-only global scalar normalization for numerical conditioning.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset


# ============================================================
# 0. Paths
# ============================================================
SCENARIO_DIR = Path(r"E4_城市道路旁_1次叠加")
TRAIN_DAT = SCENARIO_DIR / "train_B_wgan_aug.dat"
VAL_DAT = SCENARIO_DIR / "val_real_only.dat"
OUT_DIR = SCENARIO_DIR / "FORMAL_E4_BASELINES"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = OUT_DIR / "TEM1Dformer_E4_FORMAL_stable_best.pth"
HISTORY_OUT = OUT_DIR / "TEM1Dformer_E4_stable_training_history.csv"
EFFICIENCY_OUT = OUT_DIR / "TEM1Dformer_E4_stable_training_efficiency.csv"
METADATA_OUT = OUT_DIR / "TEM1Dformer_E4_stable_training_metadata.json"


# ============================================================
# 1. Paper-guided protocol
# ============================================================
N_CH = 40

FEATURES = 32
N_HEADS = 4
FFN_DIM = 64
N_RESBLOCKS = 10
DROPOUT = 0.10
DROP_PATH = 0.00

BATCH = 32
EPOCHS = 220
LR0 = 1e-4
MILESTONES = [40, 60, 80]
LR_GAMMA = 0.1

LOSS_MSE_W = 0.8
LOSS_MAE_W = 0.2

SEED = 2024
EPS = 1e-12
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

PAPER_REPORTED_PARAMS = 77921

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================
# 2. Utilities / data
# ============================================================
def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_project_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print("Loading:", path)
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] < 80:
        raise ValueError(
            f"{path}: expected at least 80 data columns after index, got {df.shape[1]}"
        )

    arr = df.iloc[:, :80].to_numpy(dtype=np.float32, copy=True)
    del df

    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: NaN/Inf detected")

    return arr[:, :40], arr[:, 40:80]


def load_split(path: Path, n_total: int):
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "For a fair benchmark, reuse the exact split from the H512 formal run."
        )

    split = np.load(path)
    if "train_indices" not in split or "val_indices" not in split:
        raise KeyError(
            f"{path} must contain 'train_indices' and 'val_indices'."
        )

    train_idx = split["train_indices"].astype(np.int64)
    val_idx = split["val_indices"].astype(np.int64)

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Empty train or validation split.")
    if train_idx.min() < 0 or val_idx.min() < 0:
        raise ValueError("Negative indices detected in split.")
    if train_idx.max() >= n_total or val_idx.max() >= n_total:
        raise ValueError("Split indices exceed dataset size.")

    return train_idx, val_idx



def load_dat(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print("Loading:", path)
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 80:
        raise ValueError(f"{path}: expected N x 80, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: NaN/Inf detected")
    return arr[:, :40], arr[:, 40:80]


# ============================================================
# 3. TEM1Dformer paper-guided architecture
# ============================================================
class DropPath(nn.Module):
    """Stochastic depth. DROP_PATH=0 by default because the paper gives no rate."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep) * random_tensor


class ConvQKVProjection(nn.Module):
    """
    Paper Eq. (7)-(9):
      pointwise convolution -> depthwise convolution
    for Q, K and V.
    Input/output: [B, L, C]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pointwise = nn.Conv1d(
            channels, channels, kernel_size=1, bias=True
        )
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

    def forward(self, x):
        x = x.transpose(1, 2)      # [B, C, L]
        x = self.pointwise(x)
        x = self.depthwise(x)
        return x.transpose(1, 2)   # [B, L, C]


class TEM1DformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = ConvQKVProjection(dim)
        self.k_proj = ConvQKVProjection(dim)
        self.v_proj = ConvQKVProjection(dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        b, l, c = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).contiguous().reshape(b, l, c)
        z = self.out_proj(z)
        z = self.out_drop(z)
        return z


class TEM1DformerEncoderBlock(nn.Module):
    """
    One ViT encoder block, matching the paper's best ablation configuration:
    LayerNorm -> MHA -> residual
    LayerNorm -> MLP -> residual
    """

    def __init__(
        self,
        dim: int = FEATURES,
        num_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        drop_path: float = DROP_PATH,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TEM1DformerAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ResidualBlock1D(nn.Module):
    """
    Fig. 4 of the paper:
      Conv1d + ReLU -> Conv1d + ReLU -> local residual add
    No BatchNorm.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        z = self.relu1(self.conv1(x))
        z = self.relu2(self.conv2(z))
        return x + z


class TEM1Dformer(nn.Module):
    def __init__(
        self,
        features: int = FEATURES,
        num_heads: int = N_HEADS,
        ffn_dim: int = FFN_DIM,
        n_resblocks: int = N_RESBLOCKS,
        dropout: float = DROPOUT,
        drop_path: float = DROP_PATH,
    ):
        super().__init__()

        # Fig. 2: two Conv1d feature-extraction layers, both followed by ReLU.
        self.feature = nn.Sequential(
            nn.Conv1d(1, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(features, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
        )

        # Paper ablation: one ViT encoder layer performs best.
        self.transformer = TEM1DformerEncoderBlock(
            dim=features,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            drop_path=drop_path,
        )

        # Fig. 2 explicitly shows ResidualBlock1 ... ResidualBlock10.
        self.residual_learning = nn.Sequential(
            *[ResidualBlock1D(features) for _ in range(n_resblocks)]
        )

        # Fig. 2: Conv1d + ReLU -> Conv1d -> Flatten.
        self.recovery1 = nn.Conv1d(
            features, features, kernel_size=3, padding=1, bias=True
        )
        self.recovery_relu = nn.ReLU(inplace=False)
        self.recovery2 = nn.Conv1d(
            features, 1, kernel_size=3, padding=1, bias=True
        )

        self.apply(self._init_weights)

        # Stability fix for residual/noise learning.
        # The original TEM1Dformer paper does not specify initialization.
        # Start the denoiser as the identity mapping (noise_hat ~= 0) rather
        # than with a large random residual. This changes no architecture,
        # parameter count, loss, optimizer, scheduler, or data protocol.
        for block in self.residual_learning:
            nn.init.zeros_(block.conv2.weight)
            if block.conv2.bias is not None:
                nn.init.zeros_(block.conv2.bias)

        nn.init.zeros_(self.recovery2.weight)
        if self.recovery2.bias is not None:
            nn.init.zeros_(self.recovery2.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, noisy):
        """
        noisy: [B, 40], scaled domain

        The paper describes training by reducing the error between predicted
        noise and actual noise. We therefore use residual/noise learning:
            noise_hat = F(noisy)
            clean_hat = noisy - noise_hat
        """
        z = noisy.unsqueeze(1)          # [B, 1, 40]
        z = self.feature(z)             # [B, C, 40]

        z = z.transpose(1, 2)           # [B, 40, C]
        z = self.transformer(z)
        z = z.transpose(1, 2)           # [B, C, 40]

        z = self.residual_learning(z)

        z = self.recovery_relu(self.recovery1(z))
        noise_hat = self.recovery2(z).squeeze(1)  # [B, 40]
        clean_hat = noisy - noise_hat

        return clean_hat, noise_hat


# ============================================================
# 4. Validation
# ============================================================
@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    sqe = 0.0
    signal = 0.0
    abse = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred, _ = model(x)
        e = pred - y

        sqe += torch.sum(e * e).item()
        signal += torch.sum(y * y).item()
        abse += torch.sum(torch.abs(e)).item()
        count += y.numel()

    mse = sqe / count
    rmse = np.sqrt(mse)
    nrmse = np.sqrt(sqe / max(signal, EPS))
    snr = np.inf if sqe == 0 else 10.0 * np.log10(signal / sqe)
    mae = abse / count

    return {
        "mse_scaled": float(mse),
        "rmse_scaled": float(rmse),
        "nrmse": float(nrmse),
        "snr_db": float(snr),
        "mae_scaled": float(mae),
    }



@torch.inference_mode()
def evaluate_identity(loader, device):
    """Metrics for the trivial residual baseline noise_hat=0 => clean_hat=noisy."""
    sqe = 0.0
    signal = 0.0
    abse = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        e = x - y
        sqe += torch.sum(e * e).item()
        signal += torch.sum(y * y).item()
        abse += torch.sum(torch.abs(e)).item()
        count += y.numel()

    mse = sqe / count
    rmse = np.sqrt(mse)
    nrmse = np.sqrt(sqe / max(signal, EPS))
    snr = np.inf if sqe == 0 else 10.0 * np.log10(signal / sqe)
    mae = abse / count
    return {
        "mse_scaled": float(mse),
        "rmse_scaled": float(rmse),
        "nrmse": float(nrmse),
        "snr_db": float(snr),
        "mae_scaled": float(mae),
    }


# ============================================================
# 5. Main
# ============================================================
def main() -> None:
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("FORMAL TRAINING ONLY: TEM1Dformer paper-guided reimplementation, E4")
    print("=" * 92)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Scenario:", SCENARIO_DIR)
    print("Training file:", TRAIN_DAT)
    print("Validation file:", VAL_DAT)
    print("Output directory:", OUT_DIR)
    print("Independent test data accessed by this script: NO")
    print("Official TEM1Dformer code available: NO")
    print("Implementation type: paper-guided reimplementation")
    print("Architecture: Conv1d x2 -> 1 ViT encoder -> ResBlock x10 -> Conv1d x2")
    print("BatchNorm: NO")
    print("Residual/noise learning: YES")
    print("Loss: 0.8*MSE + 0.2*MAE on predicted noise")
    print("Optimizer: Adam, lr=1e-4")
    print("Scheduler: MultiStepLR milestones [40,60,80], gamma=0.1")
    print("Batch:", BATCH)
    print("Epochs:", EPOCHS)
    print("Checkpoint: minimum validation NRMSE")
    print("=" * 92)

    train_noisy, train_clean = load_dat(TRAIN_DAT)
    val_noisy, val_clean = load_dat(VAL_DAT)

    n_train = len(train_noisy)
    n_val = len(val_noisy)
    print("Optimization samples:", n_train)
    print("Validation samples:", n_val)

    # Train-only scalar normalization: validation/test never determine the scale.
    train_clean64 = train_clean.astype(np.float64)
    train_noise64 = (
        train_noisy.astype(np.float64)
        - train_clean.astype(np.float64)
    )
    clean_rms = float(np.sqrt(np.mean(train_clean64 ** 2)))
    noise_rms = float(np.sqrt(np.mean(train_noise64 ** 2)))
    del train_clean64, train_noise64

    if not np.isfinite(clean_rms) or clean_rms <= 0:
        raise ValueError(f"Invalid train clean RMS: {clean_rms}")

    scale = 1.0 / clean_rms

    print("Train clean RMS (physical):", clean_rms)
    print("Train noise RMS (physical):", noise_rms)
    print("Scale multiplier:", scale)
    print("Scaled train noise RMS:", noise_rms * scale)

    train_ds = TensorDataset(
        torch.from_numpy((train_noisy * scale).astype(np.float32)),
        torch.from_numpy((train_clean * scale).astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy((val_noisy * scale).astype(np.float32)),
        torch.from_numpy((val_clean * scale).astype(np.float32)),
    )
    del train_noisy, train_clean, val_noisy, val_clean

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    raw_val = evaluate_identity(val_loader, device)
    print("\n===== Validation sanity baseline: identity / no denoising =====")
    print("Raw Val NRMSE:", raw_val["nrmse"])
    print("Raw Val SNR_dB:", raw_val["snr_db"])
    print("Raw Val RMSE_scaled:", raw_val["rmse_scaled"])

    model = TEM1Dformer().to(device)
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("Trainable parameters (this reimplementation):", params)
    print("Paper-reported TEM1Dformer parameters:", PAPER_REPORTED_PARAMS)
    print("Parameter difference:", params - PAPER_REPORTED_PARAMS)
    print(
        "Parameter difference (%):",
        (params - PAPER_REPORTED_PARAMS) / PAPER_REPORTED_PARAMS * 100.0,
    )

    model.eval()
    with torch.inference_mode():
        dummy = torch.zeros(2, N_CH, device=device)
        pred_dummy, noise_dummy = model(dummy)
    if pred_dummy.shape != (2, N_CH) or noise_dummy.shape != (2, N_CH):
        raise RuntimeError(
            f"Model output shape error: pred={pred_dummy.shape}, noise={noise_dummy.shape}"
        )
    print("Shape check: PASSED")

    # With the stable residual initialization, the initial prediction must be
    # essentially identical to the noisy input.
    init_val = evaluate(model, val_loader, device)
    print("\n===== Initialization sanity check =====")
    print("Initial model Val NRMSE:", init_val["nrmse"])
    print("Identity Raw Val NRMSE:", raw_val["nrmse"])
    print("Initial minus Raw Val NRMSE:", init_val["nrmse"] - raw_val["nrmse"])

    optimizer = optim.Adam(model.parameters(), lr=LR0)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=LR_GAMMA
    )

    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    best_nrmse = np.inf
    best_epoch = 0
    best_val = None
    history = []

    total_t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        model.train()

        loss_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        ns = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            _, noise_hat = model(batch_x)
            target_noise = batch_x - batch_y

            lmse = mse_loss(noise_hat, target_noise)
            lmae = mae_loss(noise_hat, target_noise)
            loss = LOSS_MSE_W * lmse + LOSS_MAE_W * lmae

            loss.backward()
            optimizer.step()

            b = batch_x.shape[0]
            loss_sum += float(loss.item()) * b
            mse_sum += float(lmse.item()) * b
            mae_sum += float(lmae.item()) * b
            ns += b

        val = evaluate(model, val_loader, device)
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_t0

        train_loss = loss_sum / ns
        train_mse = mse_sum / ns
        train_mae = mae_sum / ns

        row = {
            "Epoch": epoch,
            "LR": lr_now,
            "Train_MultiTaskLoss": train_loss,
            "Train_Noise_MSE": train_mse,
            "Train_Noise_MAE": train_mae,
            "Val_MSE_scaled": val["mse_scaled"],
            "Val_RMSE_scaled": val["rmse_scaled"],
            "Val_NRMSE": val["nrmse"],
            "Val_SNR_dB": val["snr_db"],
            "Val_MAE_scaled": val["mae_scaled"],
            "EpochSeconds": epoch_seconds,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(HISTORY_OUT, index=False)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR={lr_now:.3e} "
            f"TrainLoss={train_loss:.6e} "
            f"TrainMSE={train_mse:.6e} "
            f"TrainMAE={train_mae:.6e} "
            f"ValNRMSE={val['nrmse']:.6f} "
            f"ValSNR={val['snr_db']:.3f}dB "
            f"Time={epoch_seconds:.2f}s"
        )

        if val["nrmse"] < best_nrmse:
            best_nrmse = val["nrmse"]
            best_epoch = epoch
            best_val = val.copy()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val,
                    "scale": scale,
                    "train_clean_rms": clean_rms,
                    "train_noise_rms": noise_rms,
                    "architecture": "TEM1Dformer_paper_guided_reimplementation",
                    "official_code_available": False,
                    "paper_reported_parameters": PAPER_REPORTED_PARAMS,
                    "features": FEATURES,
                    "num_heads": N_HEADS,
                    "ffn_dim": FFN_DIM,
                    "n_resblocks": N_RESBLOCKS,
                    "dropout": DROPOUT,
                    "drop_path": DROP_PATH,
                    "training_loss": "0.8*MSE_noise + 0.2*MAE_noise",
                    "checkpoint_metric": "validation_NRMSE",
        "initialization": "zero-init residual-block second convs and final residual head; identity-preserving",
        "raw_validation_baseline": raw_val,
                    "batch_size": BATCH,
                    "epochs": EPOCHS,
                    "lr0": LR0,
                    "milestones": MILESTONES,
                    "lr_gamma": LR_GAMMA,
                    "seed": SEED,
                    "formal_baseline": True,
                    "test_used_during_training": False,
                    "training_data": str(TRAIN_DAT),
                    "validation_data": str(VAL_DAT),
                    "n_train": int(n_train),
                    "n_validation": int(n_val),
                    "parameters": params,
                },
                MODEL_OUT,
            )
            print("  -> best formal TEM1Dformer checkpoint saved")

        scheduler.step()

    total_seconds = time.perf_counter() - total_t0

    pd.DataFrame(
        [{
            "Model": "TEM1Dformer_E4_paper_guided",
            "Parameters": params,
            "PaperReportedParameters": PAPER_REPORTED_PARAMS,
            "BestEpoch": best_epoch,
            "BestValNRMSE": None if best_val is None else best_val["nrmse"],
            "BestValSNR_dB": None if best_val is None else best_val["snr_db"],
            "TrainingSeconds": total_seconds,
            "TrainingHours": total_seconds / 3600.0,
        }]
    ).to_csv(EFFICIENCY_OUT, index=False)

    metadata = {
        "experiment": "E4_measured_noise_TEM1Dformer_paper_guided_training",
        "date_finished_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "official_code_available": False,
        "implementation_note": (
            "Paper-guided TEM1Dformer reimplementation with identity-preserving residual initialization. "
            "Original paper specifies one ViT encoder, Conv1d feature extraction, "
            "10 residual blocks, no BN, residual/noise learning, and 0.8*MSE+0.2*MAE, "
            "but does not fully specify all hidden dimensions."
        ),
        "training_data": str(TRAIN_DAT),
        "validation_data": str(VAL_DAT),
        "model_out": str(MODEL_OUT),
        "history_out": str(HISTORY_OUT),
        "n_train": int(n_train),
        "n_validation": int(n_val),
        "seed": SEED,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "lr0": LR0,
        "milestones": MILESTONES,
        "lr_gamma": LR_GAMMA,
        "loss_mse_weight": LOSS_MSE_W,
        "loss_mae_weight": LOSS_MAE_W,
        "features": FEATURES,
        "num_heads": N_HEADS,
        "ffn_dim": FFN_DIM,
        "n_resblocks": N_RESBLOCKS,
        "dropout": DROPOUT,
        "drop_path": DROP_PATH,
        "parameters": params,
        "paper_reported_parameters": PAPER_REPORTED_PARAMS,
        "scale": scale,
        "train_clean_rms": clean_rms,
        "train_noise_rms": noise_rms,
        "best_epoch": best_epoch,
        "best_validation": best_val,
        "training_seconds": total_seconds,
        "training_hours": total_seconds / 3600.0,
        "test_used_during_training": False,
        "checkpoint_metric": "validation_NRMSE",
        "initialization": "zero-init residual-block second convs and final residual head; identity-preserving",
        "raw_validation_baseline": raw_val,
    }
    save_json(METADATA_OUT, metadata)

    print("\n" + "=" * 92)
    print("FORMAL TEM1Dformer E2 TRAINING FINISHED")
    print("=" * 92)
    print("Best epoch:", best_epoch)
    print("Best validation:", best_val)
    print("Training time (s):", total_seconds)
    print("Training time (h):", total_seconds / 3600.0)
    print("Saved model:", MODEL_OUT)
    print("Saved history:", HISTORY_OUT)
    print("Saved metadata:", METADATA_OUT)


if __name__ == "__main__":
    main()
