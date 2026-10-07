# -*- coding: utf-8 -*-
"""
FORMAL E4 TEMDnet test.

Same final metrics as our FC-ED+SRCG experiments:
  RMSE, MAE, global SNR, MRE

Story-support diagnostics:
  - per-sample RMSE before/after denoising
  - harmful modification rate:
        fraction of test samples for which TEMDnet RMSE > raw RMSE
  - median / mean per-sample RMSE change

This script performs NO training and NO test-time BN adaptation.
It uses standard model.eval().
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCENARIO_DIR = Path(r"E4_城市道路旁_1次叠加")
TEST_DAT = SCENARIO_DIR / "test_real_only.dat"
MODEL_PATH = SCENARIO_DIR / "TEMDnet_E4_FORMAL_best.pth"

PRED_OUT = SCENARIO_DIR / "TEMDnet_E4_FORMAL_predictions.csv"
CHANNEL_OUT = SCENARIO_DIR / "TEMDnet_E4_FORMAL_channel_metrics.csv"
SAMPLE_OUT = SCENARIO_DIR / "TEMDnet_E4_FORMAL_sample_metrics.csv"

N_CH = 40
IMG = 7
PAD = 9
NUM_V2 = 3
BATCH = 256
EPS = np.finfo(np.float64).eps

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_dat(path):
    a = np.loadtxt(path, dtype=np.float64)
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
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.01)


class CBA(nn.Module):
    def __init__(self, ci, co, k=3, d=1, use_bn=True, act=True):
        super().__init__()
        p = d * (k // 2)
        self.conv = nn.Conv2d(ci, co, k, padding=p, dilation=d, bias=True)
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

    def forward(self, noisy):
        z = seq_to_img(noisy)
        z = self.d1(z)
        z = self.d2(z)
        z = self.up(z)
        z = self.mid(z)
        z = self.down(z)
        z = self.d3(z)
        noise_hat = img_to_seq(self.d4(z))
        return noisy - noise_hat, noise_hat


def metrics(pred, truth):
    e = pred - truth
    rmse = np.sqrt(np.mean(e ** 2))
    mae = np.mean(np.abs(e))
    signal = np.sum(truth ** 2)
    noise = np.sum(e ** 2)
    snr = np.inf if noise == 0 else 10.0 * np.log10(signal / noise)
    mre = np.mean(np.abs(e) / (np.abs(truth) + EPS))
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "SNR": float(snr),
        "MRE": float(mre),
    }


def show(name, m):
    print(f"\n===== {name} =====")
    print("RMSE:", m["RMSE"])
    print("MAE:", m["MAE"])
    print("SNR:", m["SNR"])
    print("Mean_Relative_Error:", m["MRE"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("FORMAL TEMDnet E4 independent test")
    print("=" * 80)
    print("Device:", device)
    print("Inference mode: standard model.eval()")
    print("Test-time BN adaptation: False")

    noisy, clean = load_dat(TEST_DAT)

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    scale = float(ckpt["scale"])

    model = TEMDnet()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print("Loaded:", MODEL_PATH)
    print("Best epoch:", ckpt.get("epoch"))
    print("Validation:", ckpt.get("val_metrics"))
    print("Test samples:", len(noisy))

    outs = []
    t0 = time.perf_counter()

    with torch.inference_mode():
        for i in range(0, len(noisy), BATCH):
            j = min(i + BATCH, len(noisy))
            x = torch.from_numpy(
                (noisy[i:j] * scale).astype(np.float32)
            ).to(device)
            pred_scaled, _ = model(x)
            outs.append(
                pred_scaled.cpu().numpy().astype(np.float64) / scale
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    pred = np.vstack(outs)

    raw_m = metrics(noisy, clean)
    temd_m = metrics(pred, clean)

    show("Raw noisy input", raw_m)
    show("TEMDnet", temd_m)

    print("\n===== Global gain =====")
    print("SNR gain vs raw:", temd_m["SNR"] - raw_m["SNR"], "dB")
    print(
        "RMSE reduction vs raw (%):",
        (raw_m["RMSE"] - temd_m["RMSE"]) / raw_m["RMSE"] * 100.0
    )

    raw_sample_rmse = np.sqrt(np.mean((noisy - clean) ** 2, axis=1))
    temd_sample_rmse = np.sqrt(np.mean((pred - clean) ** 2, axis=1))
    delta = raw_sample_rmse - temd_sample_rmse

    harmful = temd_sample_rmse > raw_sample_rmse
    harmful_rate = float(np.mean(harmful))
    improved_rate = float(np.mean(temd_sample_rmse < raw_sample_rmse))
    unchanged_rate = 1.0 - harmful_rate - improved_rate

    print("\n===== Story-support diagnostic =====")
    print(
        "Harmful modification rate "
        "(TEMDnet sample RMSE > raw sample RMSE):",
        harmful_rate
    )
    print("Improved-sample rate:", improved_rate)
    print("Unchanged/tied rate:", unchanged_rate)
    print(
        "Mean sample RMSE change (raw - TEMDnet):",
        float(np.mean(delta))
    )
    print(
        "Median sample RMSE change (raw - TEMDnet):",
        float(np.median(delta))
    )

    print("\n===== Efficiency =====")
    print("Inference total (s):", dt)
    print("Inference per sample (ms):", dt / len(noisy) * 1000.0)

    sample_df = pd.DataFrame({
        "sample": np.arange(1, len(noisy) + 1),
        "raw_rmse": raw_sample_rmse,
        "TEMDnet_rmse": temd_sample_rmse,
        "raw_minus_TEMDnet_rmse": delta,
        "harmful_modification": harmful.astype(int),
    })
    sample_df.to_csv(SAMPLE_OUT, index=False)

    rows = []
    for c in range(40):
        r = metrics(noisy[:, c:c+1], clean[:, c:c+1])
        p = metrics(pred[:, c:c+1], clean[:, c:c+1])
        rows.append({
            "Channel": c + 1,
            "Raw_RMSE": r["RMSE"],
            "TEMDnet_RMSE": p["RMSE"],
            "Raw_SNR_dB": r["SNR"],
            "TEMDnet_SNR_dB": p["SNR"],
            "Raw_MRE": r["MRE"],
            "TEMDnet_MRE": p["MRE"],
        })
    pd.DataFrame(rows).to_csv(CHANNEL_OUT, index=False)

    pred_df = pd.DataFrame(np.hstack([noisy, pred, clean]))
    pred_df.columns = (
        [f"Noisy_{i+1}" for i in range(40)]
        + [f"TEMDnet_{i+1}" for i in range(40)]
        + [f"Clean_{i+1}" for i in range(40)]
    )
    pred_df.to_csv(PRED_OUT, index=False)

    print("\nSaved:")
    print(PRED_OUT)
    print(CHANNEL_OUT)
    print(SAMPLE_OUT)
    print("\nFORMAL TEMDnet E4 TEST FINISHED")
