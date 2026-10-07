# -*- coding: utf-8 -*-
"""
07 - Independent prediction/evaluation for the paper-guided TEM1Dformer
reimplementation on E4 measured-noise-contaminated ATEM.

Run ONLY after 06b_train_E4_TEM1Dformer_formal_stable.py has finished.
This script performs:
  - NO training
  - NO test-time adaptation
  - NO parameter selection on the independent test set

Outputs are aligned with Wavelet / TEMDnet / H512 FC-ED / H512+SRCG
so the methods can be plotted together later.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 0. Paths
# ============================================================
SCENARIO_DIR = Path(r"E4_城市道路旁_1次叠加")
TEST_DAT = SCENARIO_DIR / "test_real_only.dat"
OUT_DIR = SCENARIO_DIR / "FORMAL_E4_BASELINES"

MODEL_PATH = OUT_DIR / "TEM1Dformer_E4_FORMAL_stable_best.pth"
TRAINING_METADATA = OUT_DIR / "TEM1Dformer_E4_stable_training_metadata.json"

OVERALL_OUT = OUT_DIR / "TEM1Dformer_E4_stable_overall_metrics.csv"
GAIN_OUT = OUT_DIR / "TEM1Dformer_E4_stable_gain_summary.csv"
SAMPLE_OUT = OUT_DIR / "TEM1Dformer_E4_stable_sample_metrics.csv"
CHANNEL_OUT = OUT_DIR / "TEM1Dformer_E4_stable_channel_metrics.csv"
REGION_OUT = OUT_DIR / "TEM1Dformer_E4_stable_region_metrics.csv"
PREDICTIONS_OUT = OUT_DIR / "TEM1Dformer_E4_stable_predictions.csv"
ARRAYS_OUT = OUT_DIR / "TEM1Dformer_E4_stable_test_arrays.npz"
EFFICIENCY_OUT = OUT_DIR / "TEM1Dformer_E4_stable_prediction_efficiency.csv"
METADATA_OUT = OUT_DIR / "TEM1Dformer_E4_stable_prediction_metadata.json"


# ============================================================
# 1. Fixed architecture -- must match training
# ============================================================
N_CH = 40

FEATURES = 32
N_HEADS = 4
FFN_DIM = 64
N_RESBLOCKS = 10
DROPOUT = 0.10
DROP_PATH = 0.00

TEST_BATCH = 256
EPS_FLOAT = np.finfo(np.float64).eps
PAPER_REPORTED_PARAMS = 77921

REGIONS = {
    "Early_1_2": (0, 2),
    "Middle_3_30": (2, 30),
    "Late_31_40": (30, 40),
}

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ============================================================
# 2. Utilities / data
# ============================================================
def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_model_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return float(path.stat().st_size / (1024.0 ** 2))


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
# 3. Model -- exact match to training script
# ============================================================
class DropPath(nn.Module):
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
        x = x.transpose(1, 2)
        x = self.pointwise(x)
        x = self.depthwise(x)
        return x.transpose(1, 2)


class TEM1DformerAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
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
            dim=dim, num_heads=num_heads, dropout=dropout
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

        self.feature = nn.Sequential(
            nn.Conv1d(1, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv1d(features, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
        )

        self.transformer = TEM1DformerEncoderBlock(
            dim=features,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.residual_learning = nn.Sequential(
            *[ResidualBlock1D(features) for _ in range(n_resblocks)]
        )

        self.recovery1 = nn.Conv1d(
            features, features, kernel_size=3, padding=1, bias=True
        )
        self.recovery_relu = nn.ReLU(inplace=False)
        self.recovery2 = nn.Conv1d(
            features, 1, kernel_size=3, padding=1, bias=True
        )

    def forward(self, noisy):
        z = noisy.unsqueeze(1)
        z = self.feature(z)

        z = z.transpose(1, 2)
        z = self.transformer(z)
        z = z.transpose(1, 2)

        z = self.residual_learning(z)
        z = self.recovery_relu(self.recovery1(z))

        noise_hat = self.recovery2(z).squeeze(1)
        clean_hat = noisy - noise_hat
        return clean_hat, noise_hat


# ============================================================
# 4. Metrics
# ============================================================
def calc_metrics_np(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    pred64 = pred.astype(np.float64)
    truth64 = truth.astype(np.float64)
    err = pred64 - truth64

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    signal = float(np.sum(truth64 ** 2))
    noise = float(np.sum(err ** 2))

    nrmse = float(np.sqrt(noise / max(signal, EPS_FLOAT)))
    snr = float(np.inf if noise == 0 else 10.0 * np.log10(signal / noise))
    mre = float(
        np.mean(np.abs(err) / (np.abs(truth64) + EPS_FLOAT))
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE": nrmse,
        "SNR_dB": snr,
        "MRE": mre,
    }


def channel_metrics_df(noisy, pred, clean) -> pd.DataFrame:
    rows = []
    for c in range(N_CH):
        raw_m = calc_metrics_np(
            noisy[:, c:c+1], clean[:, c:c+1]
        )
        tem_m = calc_metrics_np(
            pred[:, c:c+1], clean[:, c:c+1]
        )

        row = {"Channel": c + 1}
        for k, v in raw_m.items():
            row[f"Raw_{k}"] = v
        for k, v in tem_m.items():
            row[f"TEM1Dformer_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def region_metrics_df(noisy, pred, clean) -> pd.DataFrame:
    rows = []

    for region_name, (a, b) in REGIONS.items():
        for model_name, arr in [
            ("Raw", noisy),
            ("TEM1Dformer", pred),
        ]:
            rows.append(
                {
                    "Region": region_name,
                    "StartChannel": a + 1,
                    "EndChannel": b,
                    "Model": model_name,
                    **calc_metrics_np(
                        arr[:, a:b],
                        clean[:, a:b],
                    ),
                }
            )

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
    x = torch.from_numpy(
        (sample_x[:b] * scale).astype(np.float32)
    ).to(device)

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
# 5. Main
# ============================================================
def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 92)
    print("FORMAL INDEPENDENT PREDICTION: TEM1Dformer reimplementation, d=10")
    print("=" * 92)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Test file:", TEST_DAT)
    print("Model:", MODEL_PATH)
    print("Training performed by this script: NO")
    print("Test-time adaptation: NO")
    print("Official TEM1Dformer code available: NO")
    print("Implementation type: paper-guided reimplementation")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run 06_train_d10_TEM1Dformer_formal.py first."
        )

    ckpt = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    scale = float(ckpt["scale"])

    # Rebuild using the configuration stored in the checkpoint.
    model = TEM1Dformer(
        features=int(ckpt.get("features", FEATURES)),
        num_heads=int(ckpt.get("num_heads", N_HEADS)),
        ffn_dim=int(ckpt.get("ffn_dim", FFN_DIM)),
        n_resblocks=int(ckpt.get("n_resblocks", N_RESBLOCKS)),
        dropout=float(ckpt.get("dropout", DROPOUT)),
        drop_path=float(ckpt.get("drop_path", DROP_PATH)),
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()

    params = int(sum(p.numel() for p in model.parameters()))

    print("Best epoch:", ckpt.get("epoch"))
    print("Validation:", ckpt.get("val_metrics"))
    print("Scale:", scale)
    print("TEM1Dformer parameters (reimplementation):", params)
    print("Paper-reported parameters:", PAPER_REPORTED_PARAMS)

    noisy, clean = load_dat(TEST_DAT)
    noisy64 = noisy.astype(np.float64)
    clean64 = clean.astype(np.float64)

    preds = []
    test_t0 = time.perf_counter()

    with torch.inference_mode():
        for i in range(0, len(noisy), TEST_BATCH):
            j = min(i + TEST_BATCH, len(noisy))

            x = torch.from_numpy(
                (noisy[i:j] * scale).astype(np.float32)
            ).to(device)

            pred_scaled, _ = model(x)

            preds.append(
                pred_scaled.cpu().numpy().astype(np.float64)
                / scale
            )

    sync_cuda()
    test_seconds = time.perf_counter() - test_t0
    pred = np.vstack(preds)

    raw_m = calc_metrics_np(noisy64, clean64)
    tem_m = calc_metrics_np(pred, clean64)

    print("\n===== Raw =====")
    for k, v in raw_m.items():
        print(f"{k}: {v}")

    print("\n===== TEM1Dformer =====")
    for k, v in tem_m.items():
        print(f"{k}: {v}")

    pd.DataFrame(
        [
            {"Model": "Raw", **raw_m},
            {"Model": "TEM1Dformer", **tem_m},
        ]
    ).to_csv(
        OVERALL_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Sample-level harmful modification
    # --------------------------------------------------------
    raw_sample_rmse = np.sqrt(
        np.mean((noisy64 - clean64) ** 2, axis=1)
    )
    tem_sample_rmse = np.sqrt(
        np.mean((pred - clean64) ** 2, axis=1)
    )

    delta = raw_sample_rmse - tem_sample_rmse

    harmful = tem_sample_rmse > raw_sample_rmse
    improved = tem_sample_rmse < raw_sample_rmse
    tied = ~(harmful | improved)

    sample_df = pd.DataFrame(
        {
            "SampleIndex_1based": np.arange(
                1, len(noisy64) + 1
            ),
            "Raw_RMSE": raw_sample_rmse,
            "TEM1Dformer_RMSE": tem_sample_rmse,
            "Raw_minus_TEM1Dformer_RMSE": delta,
            "TEM1Dformer_Harmful_vs_Raw": harmful.astype(int),
            "TEM1Dformer_Improved_vs_Raw": improved.astype(int),
        }
    )
    sample_df.to_csv(SAMPLE_OUT, index=False)

    harmful_summary = {
        "TEM1Dformer_sample_harmful_rate":
            float(np.mean(harmful)),
        "TEM1Dformer_sample_improved_rate":
            float(np.mean(improved)),
        "TEM1Dformer_sample_tied_rate":
            float(np.mean(tied)),
        "Mean_raw_minus_TEM1Dformer_sample_RMSE":
            float(np.mean(delta)),
        "Median_raw_minus_TEM1Dformer_sample_RMSE":
            float(np.median(delta)),
    }

    print("\n===== Sample-level harmful modification =====")
    for k, v in harmful_summary.items():
        print(f"{k}: {v}")

    # --------------------------------------------------------
    # Gain summary
    # --------------------------------------------------------
    gain_summary = {
        "TEM1Dformer_RMSE_reduction_vs_Raw_pct":
            (raw_m["RMSE"] - tem_m["RMSE"])
            / raw_m["RMSE"] * 100.0,
        "TEM1Dformer_SNR_gain_vs_Raw_dB":
            tem_m["SNR_dB"] - raw_m["SNR_dB"],
        "TEM1Dformer_MRE_change_vs_Raw_pct":
            (tem_m["MRE"] - raw_m["MRE"])
            / raw_m["MRE"] * 100.0,
    }

    pd.DataFrame(
        [gain_summary]
    ).to_csv(
        GAIN_OUT,
        index=False,
    )

    print("\n===== Gain summary =====")
    for k, v in gain_summary.items():
        print(f"{k}: {v}")

    channel_metrics_df(
        noisy64,
        pred,
        clean64,
    ).to_csv(
        CHANNEL_OUT,
        index=False,
    )

    region_metrics_df(
        noisy64,
        pred,
        clean64,
    ).to_csv(
        REGION_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Plotting-friendly outputs
    # --------------------------------------------------------
    pred_df = pd.DataFrame(
        np.hstack([noisy64, pred, clean64])
    )

    pred_df.columns = (
        [f"Noisy_{i+1}" for i in range(N_CH)]
        + [f"TEM1Dformer_{i+1}" for i in range(N_CH)]
        + [f"Clean_{i+1}" for i in range(N_CH)]
    )

    pred_df.to_csv(
        PREDICTIONS_OUT,
        index=False,
    )

    np.savez_compressed(
        ARRAYS_OUT,
        noisy=noisy64.astype(np.float32),
        clean=clean64.astype(np.float32),
        tem1dformer=pred.astype(np.float32),
    )

    # --------------------------------------------------------
    # Efficiency
    # --------------------------------------------------------
    efficiency_rows = []

    for b in [1, 256]:
        efficiency_rows.append(
            {
                "Model": "TEM1Dformer",
                "Parameters": params,
                **benchmark_forward(
                    model,
                    noisy64,
                    device,
                    scale,
                    b,
                ),
            }
        )

    efficiency_rows.append(
        {
            "Model": "FinalIndependentTestPass",
            "Parameters": params,
            "BatchSize": TEST_BATCH,
            "Warmup": np.nan,
            "Repeats": 1,
            "MeanBatch_ms": np.nan,
            "StdBatch_ms": np.nan,
            "MeanPerSample_ms":
                test_seconds / len(noisy64) * 1000.0,
            "Throughput_samples_per_s":
                len(noisy64) / test_seconds,
            "TotalTestSeconds": test_seconds,
        }
    )

    if TRAINING_METADATA.exists():
        with open(
            TRAINING_METADATA,
            "r",
            encoding="utf-8",
        ) as f:
            train_meta = json.load(f)

        efficiency_rows.append(
            {
                "Model": "TrainingWallTime_TEM1Dformer",
                "Parameters": params,
                "TrainingSeconds":
                    train_meta.get("training_seconds"),
                "TrainingHours":
                    train_meta.get("training_hours"),
            }
        )

    pd.DataFrame(
        efficiency_rows
    ).to_csv(
        EFFICIENCY_OUT,
        index=False,
    )

    metadata = {
        "experiment":
            "E2_measured_noise_TEM1Dformer_paper_guided_independent_prediction",
        "date_finished_local":
            time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "gpu":
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
        "official_code_available": False,
        "implementation_type":
            "paper-guided reimplementation",
        "test_csv": str(TEST_DAT),
        "n_test": int(len(noisy64)),
        "test_batch_size": TEST_BATCH,
        "model_path": str(MODEL_PATH),
        "model_size_MB": safe_model_size_mb(MODEL_PATH),
        "parameters": params,
        "paper_reported_parameters":
            PAPER_REPORTED_PARAMS,
        "best_epoch": ckpt.get("epoch"),
        "validation": ckpt.get("val_metrics"),
        "scale": scale,
        "final_test_seconds": test_seconds,
        "training_performed": False,
        "test_time_adaptation": False,
        "sample_level_harmful_summary":
            harmful_summary,
        "gain_summary": gain_summary,
    }

    save_json(
        METADATA_OUT,
        metadata,
    )

    print("\n" + "=" * 92)
    print("FORMAL TEM1Dformer E2 INDEPENDENT PREDICTION FINISHED")
    print("=" * 92)
    print("Test samples:", len(noisy64))
    print("Final test pass time (s):", test_seconds)

    print("\nSaved outputs:")
    for p in [
        OVERALL_OUT,
        GAIN_OUT,
        SAMPLE_OUT,
        CHANNEL_OUT,
        REGION_OUT,
        PREDICTIONS_OUT,
        ARRAYS_OUT,
        EFFICIENCY_OUT,
        METADATA_OUT,
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()
