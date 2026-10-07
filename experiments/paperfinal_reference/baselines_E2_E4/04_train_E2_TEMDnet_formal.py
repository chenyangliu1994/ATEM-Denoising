# -*- coding: utf-8 -*-
"""
FORMAL external baseline: TEMDnet on E2

Purpose:
  Produce ONE defensible TEMDnet result for comparison with our method.

Frozen protocol:
  - Same E4 training / validation split as our method
  - No additional augmentation inside this script
  - Architecture follows released TEMDnet sig_noise_prior.py (ResBlock-v2 x3)
  - Signal-to-image follows paper / released transformation:
        40 -> repeat final channel to 49 -> 7x7 -> reverse odd rows
  - Residual/noise learning:
        noise_hat = F(x)
        clean_hat = x - noise_hat
  - TEMDnet training loss: MSE(noise_hat, x-clean)
  - Checkpoint selection: validation NRMSE (equivalent ordering to val RMSE/MSE)
  - Final test is NOT read here
  - Standard BatchNorm train/eval semantics are used for formal inference
  - Train-only global scalar normalization is used only for numerical conditioning

This is the formal baseline run. Do not use the earlier debugging checkpoints.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

SCENARIO_DIR = Path(r"E2_屏蔽室_1次叠加")
TRAIN_DAT = SCENARIO_DIR / "train_B_wgan_aug.dat"
VAL_DAT = SCENARIO_DIR / "val_real_only.dat"
MODEL_OUT = SCENARIO_DIR / "TEMDnet_E2_FORMAL_best.pth"

N_CH = 40
IMG = 7
PAD = 9

BATCH = 128
EPOCHS = 220
LR0 = 1e-3
LR_DECAY = 0.98
INIT_STD = 0.05
SEED = 2024
NUM_V2 = 3

# Evaluation formulas are used only for validation reporting / selection.
EPS = 1e-12

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Faster cuDNN path; unlike the prior debug script this does not force
# deterministic dilated-convolution kernels.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dat(path):
    if not path.exists():
        raise FileNotFoundError(path)
    a = np.loadtxt(path, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 80:
        raise ValueError(f"{path}: expected N x 80, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: NaN/Inf detected")
    return a[:, :40], a[:, 40:]


def snake(z):
    rows = []
    for r in range(IMG):
        row = z[:, :, r, :]
        if r % 2 == 1:
            row = torch.flip(row, dims=[-1])
        rows.append(row)
    return torch.stack(rows, dim=2)


def seq_to_img(x):
    x49 = torch.cat([x, x[:, -1:].expand(-1, PAD)], dim=1)
    return snake(x49.reshape(-1, 1, IMG, IMG))


def img_to_seq(z):
    return snake(z).reshape(z.shape[0], -1)[:, :N_CH]


def make_bn(c):
    # Standard BN semantics for the cross-framework reimplementation.
    # eps and momentum correspond approximately to TF1 defaults.
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class CBA(nn.Module):
    def __init__(self, ci, co, k=3, d=1, use_bn=True, act=True):
        super().__init__()
        p = d * (k // 2)
        self.conv = nn.Conv2d(
            ci, co, kernel_size=k,
            padding=p, dilation=d, bias=True
        )
        self.bn = make_bn(co) if use_bn else None
        self.act = nn.ReLU() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResV1(nn.Module):
    def __init__(self, ci, cm, co):
        super().__init__()
        self.shortcut = CBA(ci, co, 1, 1, True, False)
        self.c1 = CBA(ci, cm, 1, 1, True, True)
        self.c2 = CBA(cm, cm, 3, 1, True, True)
        self.c3 = CBA(cm, co, 1, 1, True, False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(
            self.shortcut(x) + self.c3(self.c2(self.c1(x)))
        )


class ResV2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = CBA(c, c, 3, 1, True, True)
        self.c2 = CBA(c, c, 3, 1, True, False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.c2(self.c1(x)))


class TEMDnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = CBA(1, 32, 3, 1, False, True)
        self.d2 = CBA(32, 64, 3, 2, False, True)
        self.up = ResV1(64, 64, 128)
        self.mid = nn.Sequential(*[ResV2(128) for _ in range(NUM_V2)])
        self.down = ResV1(128, 128, 64)
        self.d3 = CBA(64, 32, 3, 2, False, True)
        self.d4 = CBA(32, 1, 3, 1, False, False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(
                m.weight, mean=0.0, std=INIT_STD,
                a=-2 * INIT_STD, b=2 * INIT_STD
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, noisy):
        z = seq_to_img(noisy)
        z = self.d1(z)
        z = self.d2(z)
        z = self.up(z)
        z = self.mid(z)
        z = self.down(z)
        z = self.d3(z)
        noise_hat = img_to_seq(self.d4(z))
        clean_hat = noisy - noise_hat
        return clean_hat, noise_hat


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


if __name__ == "__main__":
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FORMAL TEMDnet external baseline - E2")
    print("=" * 80)
    print("Device:", device)
    print("Architecture: released sig_noise_prior.py, ResBlock-v2 x3")
    print("Signal-to-image: 40 -> 49 -> 7x7 snake")
    print("Loss: TEMDnet residual MSE")
    print("Checkpoint: minimum validation NRMSE")
    print("Formal inference: standard model.eval() BatchNorm")
    print("Test data used during training: False")
    print("Extra augmentation inside script: False")
    print("=" * 80)

    tx_raw, ty_raw = load_dat(TRAIN_DAT)
    vx_raw, vy_raw = load_dat(VAL_DAT)

    clean_rms = float(np.sqrt(np.mean(ty_raw.astype(np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean((tx_raw.astype(np.float64) -
                                       ty_raw.astype(np.float64)) ** 2)))
    scale = 1.0 / clean_rms

    print("Train samples:", len(tx_raw))
    print("Val samples:", len(vx_raw))
    print("Train clean RMS (physical):", clean_rms)
    print("Train noise RMS (physical):", noise_rms)
    print("Scale multiplier:", scale)
    print("Scaled noise RMS:", noise_rms * scale)

    tx = torch.from_numpy((tx_raw * scale).astype(np.float32))
    ty = torch.from_numpy((ty_raw * scale).astype(np.float32))
    vx = torch.from_numpy((vx_raw * scale).astype(np.float32))
    vy = torch.from_numpy((vy_raw * scale).astype(np.float32))

    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(tx, ty),
        batch_size=BATCH,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(vx, vy),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = TEMDnet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", params)

    opt = optim.Adam(model.parameters(), lr=LR0)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)
    mse = nn.MSELoss()

    best_nrmse = np.inf
    best_epoch = 0
    best_val = None

    t_all = time.perf_counter()

    for ep in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        loss_sum = 0.0
        ns = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            _, noise_hat = model(x)
            target_noise = x - y
            loss = mse(noise_hat, target_noise)

            loss.backward()
            opt.step()

            b = x.shape[0]
            loss_sum += loss.item() * b
            ns += b

        val = evaluate(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"Epoch [{ep:03d}/{EPOCHS}] "
            f"LR={lr_now:.3e} "
            f"TrainResidualMSE={loss_sum/ns:.6e} "
            f"ValNRMSE={val['nrmse']:.6f} "
            f"ValSNR={val['snr_db']:.3f}dB "
            f"Time={time.perf_counter()-t0:.2f}s"
        )

        if val["nrmse"] < best_nrmse:
            best_nrmse = val["nrmse"]
            best_epoch = ep
            best_val = val.copy()

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_metrics": val,
                "scale": scale,
                "train_clean_rms": clean_rms,
                "train_noise_rms": noise_rms,
                "architecture": "TEMDnet_released_sig_noise_prior_3xResV2",
                "signal_to_image": "40_to_49_repeat_last_7x7_snake",
                "training_loss": "residual_noise_MSE",
                "checkpoint_metric": "validation_NRMSE",
                "batch_size": BATCH,
                "epochs": EPOCHS,
                "lr0": LR0,
                "lr_decay": LR_DECAY,
                "seed": SEED,
                "formal_baseline": True,
                "standard_bn_eval": True,
                "test_used_during_training": False,
                "extra_augmentation_inside_script": False,
                "source_repo": "https://github.com/tonyckc/TEMDnet_demo",
            }, MODEL_OUT)

            print("  -> best formal TEMDnet checkpoint saved")

        scheduler.step()

    print("\n" + "=" * 80)
    print("FORMAL TEMDnet E2 TRAINING FINISHED")
    print("=" * 80)
    print("Best epoch:", best_epoch)
    print("Best validation:", best_val)
    print("Total time (s):", time.perf_counter() - t_all)
    print("Saved:", MODEL_OUT)
