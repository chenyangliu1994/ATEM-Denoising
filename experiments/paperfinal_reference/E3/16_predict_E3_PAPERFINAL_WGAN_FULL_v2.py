# -*- coding: utf-8 -*-
"""
02 - Paper-final E3 WGAN-full independent prediction/evaluation for measured-noise-contaminated ATEM responses.

Run this ONLY after 15_train_E3_PAPERFINAL_WGAN_FULL.py has finished.
This script does no training and no test-time adaptation.

It loads the validation-selected Stage-1 and Stage-2 checkpoints, then evaluates
one independent test set and saves all data needed for the manuscript figures:

- Raw / H512 FC-ED / H512+SRCG overall metrics
- sample-level harmful modification
- 40-channel metrics
- early/middle/late metrics
- alpha statistics
- sample-channel useful/harmful correction selectivity
- representative sample indices
- full arrays: noisy, clean, FC-ED, SRCG, alpha, alpha_star
- figure-ready 120-column WGAN/FCED prediction files
- per-sample RMSE table and formal 40-channel time axis
- inference time / parameter counts / model sizes

Expected files
--------------
E3_城市道路旁_16次叠加/test_real_only.dat
E3_城市道路旁_16次叠加/PAPERFINAL_E3_WGAN_FULL/FCED_stage1_H512_E220_best.pth
E3_城市道路旁_16次叠加/PAPERFINAL_E3_WGAN_FULL/FCED_H512_SRCG_continuous_best.pth
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. Paths
# ============================================================
SCENARIO_DIR = Path(r"E3_城市道路旁_16次叠加")
TEST_DAT = SCENARIO_DIR / "test_real_only.dat"
OUT_DIR = SCENARIO_DIR / "PAPERFINAL_E3_WGAN_FULL"

STAGE1_BEST = OUT_DIR / "FCED_stage1_H512_E220_best.pth"
STAGE2_BEST = OUT_DIR / "FCED_H512_SRCG_continuous_best.pth"
TRAINING_METADATA = OUT_DIR / "training_metadata.json"

OVERALL_METRICS = OUT_DIR / "overall_metrics.csv"
CHANNEL_METRICS = OUT_DIR / "channel_metrics.csv"
REGION_METRICS = OUT_DIR / "region_metrics.csv"
SAMPLE_METRICS = OUT_DIR / "sample_metrics.csv"
SAMPLE_HARMFUL_SUMMARY = OUT_DIR / "sample_harmful_summary.csv"
GAIN_SUMMARY = OUT_DIR / "gain_summary.csv"
ALPHA_CHANNEL_STATS = OUT_DIR / "alpha_channel_stats.csv"
ALPHA_REGION_STATS = OUT_DIR / "alpha_region_stats.csv"
SELECTIVITY_SUMMARY = OUT_DIR / "correction_selectivity_summary.csv"
SELECTIVITY_CHANNEL = OUT_DIR / "correction_selectivity_by_channel.csv"
REPRESENTATIVE_SAMPLES = OUT_DIR / "representative_samples.csv"
EFFICIENCY_OUT = OUT_DIR / "efficiency.csv"
PREDICTION_METADATA = OUT_DIR / "prediction_metadata.json"
TEST_ARRAYS_OUT = OUT_DIR / "test_arrays_full.npz"
FIG_PRED_WGAN = OUT_DIR / "test_predictions_WGAN.dat"
FIG_PRED_FCED = OUT_DIR / "test_predictions_FCED.dat"
FIG_SAMPLE_RMSE = OUT_DIR / "figure_sample_rmse.csv"
TIME_CHANNELS = OUT_DIR / "time_channels.csv"


# ============================================================
# 1. Protocol
# ============================================================
INPUT_DIM = 40
HIDDEN_DIM = 512
SRCG_HIDDEN = 64
TEST_BATCH_SIZE = 256
DEFAULT_SCALE = 1e19

EPS_FLOAT = np.finfo(np.float64).eps
EPS_ALPHA_EVAL = np.finfo(np.float64).tiny

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

REGIONS = {
    "Early_1_2": (0, 2),
    "Middle_3_30": (2, 30),
    "Late_31_40": (30, 40),
}


# ============================================================
# 2. Utilities / data
# ============================================================
def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def safe_model_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return float(path.stat().st_size / (1024.0 ** 2))


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_project_dat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
# 3. Models -- exactly match training
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
# 4. Metrics / diagnostics
# ============================================================
def calc_metrics_np(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    err = pred.astype(np.float64) - truth.astype(np.float64)
    truth64 = truth.astype(np.float64)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    signal = float(np.sum(truth64 ** 2))
    noise = float(np.sum(err ** 2))
    snr = float(np.inf if noise == 0 else 10.0 * np.log10(signal / noise))
    nrmse = float(np.sqrt(noise / max(signal, EPS_FLOAT)))
    mre = float(np.mean(np.abs(err) / (np.abs(truth64) + EPS_FLOAT)))
    return {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "SNR_dB": snr, "MRE": mre}


def continuous_alpha_star_np(
    noisy: np.ndarray, candidate: np.ndarray, clean: np.ndarray
) -> np.ndarray:
    noisy64 = noisy.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    clean64 = clean.astype(np.float64)
    d = candidate64 - noisy64
    alpha_star = ((clean64 - noisy64) * d) / (d * d + EPS_ALPHA_EVAL)
    return np.clip(alpha_star, 0.0, 1.0)


def channel_metrics_df(noisy, fced, srcg, clean) -> pd.DataFrame:
    rows = []
    for c in range(INPUT_DIM):
        row = {"Channel": c + 1}
        for prefix, pred in [
            ("Raw", noisy),
            ("FCED_H512", fced),
            ("SRCG", srcg),
        ]:
            m = calc_metrics_np(pred[:, c:c+1], clean[:, c:c+1])
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def region_metrics_df(noisy, fced, srcg, clean) -> pd.DataFrame:
    rows = []
    for region_name, (a, b) in REGIONS.items():
        for model_name, pred in [
            ("Raw", noisy),
            ("FCED_H512", fced),
            ("FCED_H512_SRCG", srcg),
        ]:
            rows.append({
                "Region": region_name,
                "StartChannel": a + 1,
                "EndChannel": b,
                "Model": model_name,
                **calc_metrics_np(pred[:, a:b], clean[:, a:b]),
            })
    return pd.DataFrame(rows)


def alpha_channel_stats_df(alpha, alpha_star) -> pd.DataFrame:
    rows = []
    for c in range(INPUT_DIM):
        a = alpha[:, c].astype(np.float64)
        t = alpha_star[:, c].astype(np.float64)
        rows.append({
            "Channel": c + 1,
            "AlphaMean": float(np.mean(a)),
            "AlphaStd": float(np.std(a)),
            "AlphaMedian": float(np.median(a)),
            "AlphaP10": float(np.quantile(a, 0.10)),
            "AlphaP25": float(np.quantile(a, 0.25)),
            "AlphaP75": float(np.quantile(a, 0.75)),
            "AlphaP90": float(np.quantile(a, 0.90)),
            "AlphaStarMean": float(np.mean(t)),
            "AlphaStarMedian": float(np.median(t)),
            "AlphaTeacherMAE": float(np.mean(np.abs(a - t))),
        })
    return pd.DataFrame(rows)


def alpha_region_stats_df(alpha) -> pd.DataFrame:
    rows = []
    for region_name, (a, b) in REGIONS.items():
        values = alpha[:, a:b].astype(np.float64).reshape(-1)
        rows.append({
            "Region": region_name,
            "StartChannel": a + 1,
            "EndChannel": b,
            "AlphaMean": float(np.mean(values)),
            "AlphaStd": float(np.std(values)),
            "AlphaMedian": float(np.median(values)),
            "AlphaP10": float(np.quantile(values, 0.10)),
            "AlphaP90": float(np.quantile(values, 0.90)),
        })
    return pd.DataFrame(rows)


def correction_selectivity(noisy, fced, srcg, clean, alpha, alpha_star):
    raw_abs = np.abs(noisy.astype(np.float64) - clean.astype(np.float64))
    fced_abs = np.abs(fced.astype(np.float64) - clean.astype(np.float64))
    srcg_abs = np.abs(srcg.astype(np.float64) - clean.astype(np.float64))
    alpha64 = alpha.astype(np.float64)
    alpha_star64 = alpha_star.astype(np.float64)

    useful = fced_abs < raw_abs
    harmful = fced_abs > raw_abs
    tied = ~(useful | harmful)

    def masked_mean(values, mask):
        return float(np.mean(values[mask])) if np.sum(mask) else float("nan")

    n_total = useful.size
    n_useful = int(np.sum(useful))
    n_harmful = int(np.sum(harmful))
    n_tied = int(np.sum(tied))

    harmful_suppressed = harmful & (srcg_abs < fced_abs)
    harmful_rescued_to_raw = harmful & (srcg_abs <= raw_abs)
    useful_retained_vs_raw = useful & (srcg_abs < raw_abs)
    useful_not_worse_than_fced = useful & (srcg_abs <= fced_abs)
    srcg_better_than_fced = srcg_abs < fced_abs

    summary = pd.DataFrame([{
        "TotalSampleChannelCorrections": n_total,
        "UsefulCount": n_useful,
        "UsefulRate": n_useful / n_total,
        "HarmfulCount": n_harmful,
        "HarmfulRate": n_harmful / n_total,
        "TieCount": n_tied,
        "TieRate": n_tied / n_total,
        "AlphaMean_All": float(np.mean(alpha64)),
        "AlphaMean_Useful": masked_mean(alpha64, useful),
        "AlphaMean_Harmful": masked_mean(alpha64, harmful),
        "AlphaMedian_Useful": float(np.median(alpha64[useful])) if n_useful else float("nan"),
        "AlphaMedian_Harmful": float(np.median(alpha64[harmful])) if n_harmful else float("nan"),
        "AlphaStarMean_Useful": masked_mean(alpha_star64, useful),
        "AlphaStarMean_Harmful": masked_mean(alpha_star64, harmful),
        "AlphaUsefulMinusHarmful": masked_mean(alpha64, useful) - masked_mean(alpha64, harmful),
        "HarmfulSuppressionRate": float(np.sum(harmful_suppressed) / n_harmful) if n_harmful else float("nan"),
        "HarmfulRescuedToRawRate": float(np.sum(harmful_rescued_to_raw) / n_harmful) if n_harmful else float("nan"),
        "UsefulRetentionVsRawRate": float(np.sum(useful_retained_vs_raw) / n_useful) if n_useful else float("nan"),
        "UsefulNotWorseThanFCEDRate": float(np.sum(useful_not_worse_than_fced) / n_useful) if n_useful else float("nan"),
        "SRCGBetterThanFCED_PointRate": float(np.mean(srcg_better_than_fced)),
    }])

    rows = []
    for c in range(INPUT_DIM):
        u = useful[:, c]
        h = harmful[:, c]
        t = tied[:, c]
        n = len(u)
        nu = int(np.sum(u))
        nh = int(np.sum(h))
        hs = h & (srcg_abs[:, c] < fced_abs[:, c])
        hr = h & (srcg_abs[:, c] <= raw_abs[:, c])
        ur = u & (srcg_abs[:, c] < raw_abs[:, c])
        rows.append({
            "Channel": c + 1,
            "N": n,
            "UsefulCount": nu,
            "UsefulRate": nu / n,
            "HarmfulCount": nh,
            "HarmfulRate": nh / n,
            "TieCount": int(np.sum(t)),
            "TieRate": float(np.mean(t)),
            "AlphaMean_All": float(np.mean(alpha64[:, c])),
            "AlphaMean_Useful": float(np.mean(alpha64[u, c])) if nu else float("nan"),
            "AlphaMean_Harmful": float(np.mean(alpha64[h, c])) if nh else float("nan"),
            "AlphaStarMean_Useful": float(np.mean(alpha_star64[u, c])) if nu else float("nan"),
            "AlphaStarMean_Harmful": float(np.mean(alpha_star64[h, c])) if nh else float("nan"),
            "HarmfulSuppressionRate": float(np.sum(hs) / nh) if nh else float("nan"),
            "HarmfulRescuedToRawRate": float(np.sum(hr) / nh) if nh else float("nan"),
            "UsefulRetentionVsRawRate": float(np.sum(ur) / nu) if nu else float("nan"),
        })
    return summary, pd.DataFrame(rows)


def representative_samples_df(raw_rmse, fced_rmse, srcg_rmse) -> pd.DataFrame:
    gain = fced_rmse - srcg_rmse

    def closest_to(value: float) -> int:
        return int(np.argmin(np.abs(gain - value)))

    q25, q50, q75 = np.quantile(gain, [0.25, 0.50, 0.75])
    candidates = [
        ("gain_q25", closest_to(q25)),
        ("gain_median", closest_to(q50)),
        ("gain_q75", closest_to(q75)),
        ("largest_SRCG_gain", int(np.argmax(gain))),
        ("worst_SRCG_gain", int(np.argmin(gain))),
    ]

    rescued = np.where((fced_rmse > raw_rmse) & (srcg_rmse <= raw_rmse))[0]
    if len(rescued) > 0:
        ridx = rescued[int(np.argmax(gain[rescued]))]
        candidates.append(("harmful_FCED_rescued_by_SRCG", int(ridx)))

    rows = []
    for label, idx in candidates:
        rows.append({
            "Purpose": label,
            "SampleIndex_1based": idx + 1,
            "Raw_RMSE": float(raw_rmse[idx]),
            "FCED_RMSE": float(fced_rmse[idx]),
            "SRCG_RMSE": float(srcg_rmse[idx]),
            "FCED_minus_SRCG_RMSE": float(gain[idx]),
            "FCED_Harmful_vs_Raw": int(fced_rmse[idx] > raw_rmse[idx]),
            "SRCG_Harmful_vs_Raw": int(srcg_rmse[idx] > raw_rmse[idx]),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def benchmark_forward(
    model: nn.Module,
    sample_x: np.ndarray,
    device: torch.device,
    scale: float,
    batch_size: int,
    warmup: int = 20,
    repeats: int = 100,
) -> Dict[str, float]:
    model.eval()
    b = min(batch_size, len(sample_x))
    x = torch.from_numpy((sample_x[:b] * scale).astype(np.float32)).to(device)

    for _ in range(warmup):
        _ = model(x)
    sync_cuda()

    times = []
    for _ in range(repeats):
        sync_cuda()
        t0 = time.perf_counter()
        _ = model(x)
        sync_cuda()
        times.append(time.perf_counter() - t0)

    times = np.asarray(times, dtype=np.float64)
    mean_s = float(np.mean(times))
    std_s = float(np.std(times))
    return {
        "BatchSize": b,
        "Warmup": warmup,
        "Repeats": repeats,
        "MeanBatch_ms": mean_s * 1000.0,
        "StdBatch_ms": std_s * 1000.0,
        "MeanPerSample_ms": mean_s / b * 1000.0,
        "Throughput_samples_per_s": b / mean_s,
    }


# ============================================================
# 5. Main prediction/evaluation
# ============================================================
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("FORMAL INDEPENDENT PREDICTION: E3 measured-noise-contaminated ATEM")
    print("=" * 88)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Scenario:", SCENARIO_DIR)
    print("Test file:", TEST_DAT)
    print("Stage-1 model:", STAGE1_BEST)
    print("Stage-2 model:", STAGE2_BEST)
    print("Training performed by this script: NO")
    print("Test-time adaptation: NO")

    for p in [TEST_DAT, STAGE1_BEST, STAGE2_BEST]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load checkpoints and their actual saved dimensions/scale.
    stage1_ckpt = torch.load(STAGE1_BEST, map_location=device, weights_only=False)
    final_ckpt = torch.load(STAGE2_BEST, map_location=device, weights_only=False)

    input_dim = int(final_ckpt.get("input_dim", INPUT_DIM))
    hidden_dim = int(final_ckpt.get("hidden_dim", HIDDEN_DIM))
    srcg_hidden = int(final_ckpt.get("srcg_hidden", SRCG_HIDDEN))
    scale = float(final_ckpt.get("scale", DEFAULT_SCALE))

    if input_dim != INPUT_DIM:
        raise ValueError(f"This evaluator expects 40 channels, checkpoint has {input_dim}.")

    # Pure Stage-1 baseline from its own validation-selected checkpoint.
    fced_model = FCED(input_dim, hidden_dim).to(device)
    fced_model.load_state_dict(stage1_ckpt["model_state_dict"], strict=True)
    fced_model.eval()

    # Final model from Stage-2 checkpoint.
    final_backbone = FCED(input_dim, hidden_dim)
    final_model = FCED_SRCG(final_backbone, input_dim, srcg_hidden).to(device)
    final_model.load_state_dict(final_ckpt["model_state_dict"], strict=True)
    final_model.eval()

    print("Stage-1 best epoch:", stage1_ckpt.get("stage1_epoch"))
    print("Stage-1 validation:", stage1_ckpt.get("stage1_val_metrics"))
    print("Stage-2 best epoch:", final_ckpt.get("stage2_epoch"))
    print("Stage-2 validation:", final_ckpt.get("stage2_val_metrics"))
    print("Scale:", scale)
    print("FC-ED parameters:", count_parameters(fced_model))
    print("SRCG parameters:", count_parameters(final_model.srcg))
    print("Total parameters:", count_parameters(final_model))

    test_noisy, test_clean = load_project_dat(TEST_DAT)
    noisy64 = test_noisy.astype(np.float64)
    clean64 = test_clean.astype(np.float64)

    test_dataset = TensorDataset(torch.from_numpy(test_noisy * scale))
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # One pass through final_model gives both its frozen backbone candidate and SRCG.
    # The candidate is verified against the separately loaded Stage-1 checkpoint below.
    preds = []
    stage1_from_final = []
    alphas = []

    sync_cuda()
    test_t0 = time.perf_counter()
    with torch.inference_mode():
        for (x,) in test_loader:
            x = x.to(device, non_blocking=True)
            pred, z, alpha = final_model(x, return_aux=True)
            preds.append(pred.cpu().numpy().astype(np.float64) / scale)
            stage1_from_final.append(z.cpu().numpy().astype(np.float64) / scale)
            alphas.append(alpha.cpu().numpy().astype(np.float64))
    sync_cuda()
    test_seconds = time.perf_counter() - test_t0

    srcg_pred = np.vstack(preds)
    fced_pred = np.vstack(stage1_from_final)
    alpha = np.vstack(alphas)

    # Verify frozen Stage-2 backbone equals the official Stage-1 checkpoint.
    verify_preds = []
    with torch.inference_mode():
        for (x,) in test_loader:
            x = x.to(device, non_blocking=True)
            verify_preds.append(fced_model(x).cpu().numpy().astype(np.float64) / scale)
    fced_verify = np.vstack(verify_preds)
    backbone_max_abs_diff = float(np.max(np.abs(fced_verify - fced_pred)))
    print("Frozen-backbone verification max abs diff:", backbone_max_abs_diff)
    if not np.allclose(fced_verify, fced_pred, rtol=1e-6, atol=1e-20):
        raise RuntimeError(
            "Stage-2 frozen backbone does not match Stage-1 checkpoint. "
            "Stop and inspect checkpoints before reporting results."
        )

    alpha_star = continuous_alpha_star_np(noisy64, fced_pred, clean64)

    # ---------------- Overall ----------------
    overall_rows = []
    for model_name, pred in [
        ("Raw", noisy64),
        ("FCED_H512", fced_pred),
        ("FCED_H512_SRCG", srcg_pred),
    ]:
        m = calc_metrics_np(pred, clean64)
        overall_rows.append({"Model": model_name, **m})
        print(f"\n===== {model_name} =====")
        for k, v in m.items():
            print(f"{k}: {v}")
    pd.DataFrame(overall_rows).to_csv(OVERALL_METRICS, index=False)

    # ---------------- Sample-level harmful ----------------
    raw_sample_rmse = np.sqrt(np.mean((noisy64 - clean64) ** 2, axis=1))
    fced_sample_rmse = np.sqrt(np.mean((fced_pred - clean64) ** 2, axis=1))
    srcg_sample_rmse = np.sqrt(np.mean((srcg_pred - clean64) ** 2, axis=1))

    pd.DataFrame({
        "SampleIndex_1based": np.arange(1, len(noisy64) + 1),
        "Raw_RMSE": raw_sample_rmse,
        "FCED_H512_RMSE": fced_sample_rmse,
        "SRCG_RMSE": srcg_sample_rmse,
        "Raw_minus_FCED_RMSE": raw_sample_rmse - fced_sample_rmse,
        "Raw_minus_SRCG_RMSE": raw_sample_rmse - srcg_sample_rmse,
        "FCED_minus_SRCG_RMSE": fced_sample_rmse - srcg_sample_rmse,
        "FCED_Harmful_vs_Raw": (fced_sample_rmse > raw_sample_rmse).astype(int),
        "SRCG_Harmful_vs_Raw": (srcg_sample_rmse > raw_sample_rmse).astype(int),
        "FCED_Improved_vs_Raw": (fced_sample_rmse < raw_sample_rmse).astype(int),
        "SRCG_Improved_vs_Raw": (srcg_sample_rmse < raw_sample_rmse).astype(int),
    }).to_csv(SAMPLE_METRICS, index=False)

    sample_harmful_summary = {
        "FCED_sample_harmful_rate": float(np.mean(fced_sample_rmse > raw_sample_rmse)),
        "SRCG_sample_harmful_rate": float(np.mean(srcg_sample_rmse > raw_sample_rmse)),
        "FCED_sample_improved_rate": float(np.mean(fced_sample_rmse < raw_sample_rmse)),
        "SRCG_sample_improved_rate": float(np.mean(srcg_sample_rmse < raw_sample_rmse)),
        "SRCG_better_than_FCED_sample_rate": float(np.mean(srcg_sample_rmse < fced_sample_rmse)),
        "Mean_FCED_minus_SRCG_sample_RMSE": float(np.mean(fced_sample_rmse - srcg_sample_rmse)),
        "Median_FCED_minus_SRCG_sample_RMSE": float(np.median(fced_sample_rmse - srcg_sample_rmse)),
    }
    pd.DataFrame([sample_harmful_summary]).to_csv(SAMPLE_HARMFUL_SUMMARY, index=False)
    print("\n===== Sample-level harmful modification =====")
    for k, v in sample_harmful_summary.items():
        print(f"{k}: {v}")

    # ---------------- Gain summary ----------------
    raw_m = calc_metrics_np(noisy64, clean64)
    fced_m = calc_metrics_np(fced_pred, clean64)
    srcg_m = calc_metrics_np(srcg_pred, clean64)
    gain_summary = {
        "FCED_RMSE_reduction_vs_Raw_pct": (raw_m["RMSE"] - fced_m["RMSE"]) / raw_m["RMSE"] * 100.0,
        "SRCG_RMSE_reduction_vs_Raw_pct": (raw_m["RMSE"] - srcg_m["RMSE"]) / raw_m["RMSE"] * 100.0,
        "SRCG_RMSE_reduction_vs_FCED_pct": (fced_m["RMSE"] - srcg_m["RMSE"]) / fced_m["RMSE"] * 100.0,
        "FCED_SNR_gain_vs_Raw_dB": fced_m["SNR_dB"] - raw_m["SNR_dB"],
        "SRCG_SNR_gain_vs_Raw_dB": srcg_m["SNR_dB"] - raw_m["SNR_dB"],
        "SRCG_SNR_gain_vs_FCED_dB": srcg_m["SNR_dB"] - fced_m["SNR_dB"],
        "SRCG_MRE_change_vs_FCED_pct": (srcg_m["MRE"] - fced_m["MRE"]) / fced_m["MRE"] * 100.0,
    }
    pd.DataFrame([gain_summary]).to_csv(GAIN_SUMMARY, index=False)
    print("\n===== Gain summary =====")
    for k, v in gain_summary.items():
        print(f"{k}: {v}")

    # ---------------- Channel / region / alpha / selectivity ----------------
    channel_metrics_df(noisy64, fced_pred, srcg_pred, clean64).to_csv(CHANNEL_METRICS, index=False)
    region_metrics_df(noisy64, fced_pred, srcg_pred, clean64).to_csv(REGION_METRICS, index=False)
    alpha_channel_stats_df(alpha, alpha_star).to_csv(ALPHA_CHANNEL_STATS, index=False)
    alpha_region_stats_df(alpha).to_csv(ALPHA_REGION_STATS, index=False)

    selectivity_summary_df, selectivity_channel_df = correction_selectivity(
        noisy64, fced_pred, srcg_pred, clean64, alpha, alpha_star
    )
    selectivity_summary_df.to_csv(SELECTIVITY_SUMMARY, index=False)
    selectivity_channel_df.to_csv(SELECTIVITY_CHANNEL, index=False)
    print("\n===== Sample-channel correction selectivity =====")
    print(selectivity_summary_df.to_string(index=False))

    representative_samples_df(
        raw_sample_rmse, fced_sample_rmse, srcg_sample_rmse
    ).to_csv(REPRESENTATIVE_SAMPLES, index=False)

    # Full arrays for all later figures -- no need to rerun inference.
    np.savez_compressed(
        TEST_ARRAYS_OUT,
        noisy=noisy64.astype(np.float32),
        clean=clean64.astype(np.float32),
        fced=fced_pred.astype(np.float32),
        srcg=srcg_pred.astype(np.float32),
        alpha=alpha.astype(np.float32),
        alpha_star=alpha_star.astype(np.float32),
    )

    # Figure-ready exports. These make the final plotting scripts independent
    # from model inference and compatible with the earlier 120-column A/B style:
    # noisy40 | prediction40 | clean40.
    np.savetxt(
        FIG_PRED_WGAN,
        np.hstack([noisy64, srcg_pred, clean64]),
        fmt="%.18e",
    )
    np.savetxt(
        FIG_PRED_FCED,
        np.hstack([noisy64, fced_pred, clean64]),
        fmt="%.18e",
    )

    pd.DataFrame({
        "Sample": np.arange(1, len(noisy64) + 1),
        "Raw_RMSE": raw_sample_rmse,
        "FCED_RMSE": fced_sample_rmse,
        "SRCG_RMSE": srcg_sample_rmse,
        "AlphaMean": np.mean(alpha, axis=1),
    }).to_csv(FIG_SAMPLE_RMSE, index=False)

    # Formal 40-channel time axis used throughout the manuscript.
    time_s = np.logspace(-5, -2, INPUT_DIM)
    pd.DataFrame({
        "Channel": np.arange(1, INPUT_DIM + 1),
        "Time_s": time_s,
    }).to_csv(TIME_CHANNELS, index=False)

    # ---------------- Efficiency ----------------
    efficiency_rows = []
    for name, mdl in [
        ("FCED_H512", fced_model),
        ("FCED_H512_SRCG", final_model),
    ]:
        for b in [1, 256]:
            efficiency_rows.append({
                "Model": name,
                "Parameters": count_parameters(mdl),
                **benchmark_forward(mdl, noisy64, device, scale, b),
            })

    efficiency_rows.append({
        "Model": "FinalIndependentTestPass",
        "Parameters": np.nan,
        "BatchSize": TEST_BATCH_SIZE,
        "Warmup": np.nan,
        "Repeats": 1,
        "MeanBatch_ms": np.nan,
        "StdBatch_ms": np.nan,
        "MeanPerSample_ms": test_seconds / len(noisy64) * 1000.0,
        "Throughput_samples_per_s": len(noisy64) / test_seconds,
        "TotalTestSeconds": test_seconds,
    })

    # If available, copy training wall times into the same table for convenience.
    training_meta = None
    if TRAINING_METADATA.exists():
        with open(TRAINING_METADATA, "r", encoding="utf-8") as f:
            training_meta = json.load(f)
        efficiency_rows.append({
            "Model": "TrainingWallTime_Stage1",
            "Parameters": np.nan,
            "TrainingSeconds": training_meta.get("stage1_train_seconds"),
        })
        efficiency_rows.append({
            "Model": "TrainingWallTime_Stage2",
            "Parameters": np.nan,
            "TrainingSeconds": training_meta.get("stage2_train_seconds"),
        })

    pd.DataFrame(efficiency_rows).to_csv(EFFICIENCY_OUT, index=False)

    # ---------------- Metadata ----------------
    metadata = {
        "experiment": "E3_measured_noise_H512_FCED_SRCG_paperfinal_WGAN_independent_prediction",
        "date_finished_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scenario": "E3_urban_roadside_16stack_measured_noise",
        "test_dat": str(TEST_DAT),
        "n_test": int(len(noisy64)),
        "test_batch_size": TEST_BATCH_SIZE,
        "scale": scale,
        "stage1_checkpoint": str(STAGE1_BEST),
        "stage2_checkpoint": str(STAGE2_BEST),
        "stage1_best_epoch": stage1_ckpt.get("stage1_epoch"),
        "stage1_validation": stage1_ckpt.get("stage1_val_metrics"),
        "stage2_best_epoch": final_ckpt.get("stage2_epoch"),
        "stage2_validation": final_ckpt.get("stage2_val_metrics"),
        "fced_parameters": count_parameters(fced_model),
        "srcg_parameters": count_parameters(final_model.srcg),
        "total_parameters": count_parameters(final_model),
        "stage1_model_size_MB": safe_model_size_mb(STAGE1_BEST),
        "stage2_model_size_MB": safe_model_size_mb(STAGE2_BEST),
        "final_test_seconds": test_seconds,
        "backbone_verification_max_abs_diff": backbone_max_abs_diff,
        "sample_level_harmful_summary": sample_harmful_summary,
        "selectivity_summary": selectivity_summary_df.iloc[0].to_dict(),
        "training_performed": False,
        "test_time_adaptation": False,
        "important_note": (
            "Clean targets and alpha_star are used only for research evaluation. "
            "Model inference itself requires noisy x only."
        ),
    }
    save_json(PREDICTION_METADATA, metadata)

    print("\n" + "=" * 88)
    print("FORMAL INDEPENDENT PREDICTION FINISHED")
    print("=" * 88)
    print("Test samples:", len(noisy64))
    print("Final test pass time (s):", test_seconds)
    print("\nSaved outputs:")
    for p in [
        OVERALL_METRICS,
        CHANNEL_METRICS,
        REGION_METRICS,
        SAMPLE_METRICS,
        SAMPLE_HARMFUL_SUMMARY,
        GAIN_SUMMARY,
        ALPHA_CHANNEL_STATS,
        ALPHA_REGION_STATS,
        SELECTIVITY_SUMMARY,
        SELECTIVITY_CHANNEL,
        REPRESENTATIVE_SAMPLES,
        EFFICIENCY_OUT,
        TEST_ARRAYS_OUT,
        FIG_PRED_WGAN,
        FIG_PRED_FCED,
        FIG_SAMPLE_RMSE,
        TIME_CHANNELS,
        PREDICTION_METADATA,
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
