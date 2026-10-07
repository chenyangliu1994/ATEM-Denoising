# -*- coding: utf-8 -*-
"""
FORMAL traditional baseline: stationary-wavelet soft-threshold denoising
for E2 measured-noise-contaminated ATEM.

Purpose
-------
Provide a simple, reproducible traditional denoising baseline for the same
E2 independent test_real_only.dat used by TEMDnet / H512 FC-ED / H512+SRCG.

Method
------
- Stationary Wavelet Transform (SWT)
- Symlet-6 (sym6)
- 3 decomposition levels
- Soft thresholding
- Noise sigma estimated by MAD from the finest-scale detail coefficients
- Universal threshold: lambda = sigma * sqrt(2*log(N))
- No training, no clean target used by the denoiser itself
- Clean target is used only for research evaluation

Why level=3?
------------
The ATEM response here has only 40 channels. 40 is divisible by 2^3 but not
2^4, so SWT level 3 is the highest decomposition level that can be used
directly without artificially extending the sequence.

Dependency
----------
pip install PyWavelets
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "PyWavelets is required. Install it in this environment with:\n"
        "pip install PyWavelets"
    ) from exc


# ============================================================
# Paths
# ============================================================
SCENARIO_DIR = Path(r"E2_屏蔽室_1次叠加")
TEST_DAT = SCENARIO_DIR / "test_real_only.dat"
OUT_DIR = SCENARIO_DIR / "FORMAL_E2_BASELINES"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_OUT = OUT_DIR / "Wavelet_E2_overall_metrics.csv"
GAIN_OUT = OUT_DIR / "Wavelet_E2_gain_summary.csv"
SAMPLE_OUT = OUT_DIR / "Wavelet_E2_sample_metrics.csv"
CHANNEL_OUT = OUT_DIR / "Wavelet_E2_channel_metrics.csv"
REGION_OUT = OUT_DIR / "Wavelet_E2_region_metrics.csv"
PRED_OUT = OUT_DIR / "Wavelet_E2_predictions.csv"
NPZ_OUT = OUT_DIR / "Wavelet_E2_test_arrays.npz"
EFF_OUT = OUT_DIR / "Wavelet_E2_prediction_efficiency.csv"
META_OUT = OUT_DIR / "Wavelet_E2_prediction_metadata.json"


# ============================================================
# Fixed traditional-wavelet settings
# ============================================================
N_CH = 40
WAVELET = "sym6"
LEVEL = 3
THRESHOLD_MODE = "soft"
THRESHOLD_SCALE = 1.0

EPS = np.finfo(np.float64).eps


# ============================================================
# Data
# ============================================================
def load_test_dat(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 80:
        raise ValueError(f"{path}: expected N x 80, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: NaN/Inf detected")

    return arr[:, :40], arr[:, 40:80]


# ============================================================
# Wavelet denoiser
# ============================================================
def wavelet_denoise_one(x):
    """
    SWT-Sym6 + soft universal threshold.

    Sigma is estimated from the finest-scale detail coefficients using MAD:
        sigma = median(|d1 - median(d1)|) / 0.67448975
    then
        threshold = sigma * sqrt(2*log(N))
    """
    x = np.asarray(x, dtype=np.float64)

    coeffs = pywt.swt(
        x,
        wavelet=WAVELET,
        level=LEVEL,
        start_level=0,
        axis=-1,
        trim_approx=False,
        norm=False,
    )

    # PyWavelets SWT returns [(cA_n,cD_n), ..., (cA_1,cD_1)].
    finest_detail = np.asarray(coeffs[-1][1], dtype=np.float64)

    med = np.median(finest_detail)
    mad = np.median(np.abs(finest_detail - med))
    sigma = mad / 0.6744897501960817

    threshold = (
        THRESHOLD_SCALE
        * sigma
        * np.sqrt(2.0 * np.log(len(x)))
    )

    denoised_coeffs = []
    for cA, cD in coeffs:
        cD_thr = pywt.threshold(
            cD,
            value=threshold,
            mode=THRESHOLD_MODE,
        )
        denoised_coeffs.append((cA, cD_thr))

    y = pywt.iswt(
        denoised_coeffs,
        wavelet=WAVELET,
        axis=-1,
        norm=False,
    )

    return np.asarray(y, dtype=np.float64), float(threshold), float(sigma)


def wavelet_denoise_batch(noisy):
    pred = np.empty_like(noisy, dtype=np.float64)
    thresholds = np.empty(len(noisy), dtype=np.float64)
    sigmas = np.empty(len(noisy), dtype=np.float64)

    for i in range(len(noisy)):
        pred[i], thresholds[i], sigmas[i] = wavelet_denoise_one(noisy[i])

    return pred, thresholds, sigmas


# ============================================================
# Metrics
# ============================================================
def metrics(pred, truth):
    e = pred - truth

    rmse = np.sqrt(np.mean(e ** 2))
    mae = np.mean(np.abs(e))

    signal = np.sum(truth ** 2)
    noise = np.sum(e ** 2)

    nrmse = np.sqrt(
        noise / max(signal, EPS)
    )

    snr = (
        np.inf
        if noise == 0
        else 10.0 * np.log10(signal / noise)
    )

    mre = np.mean(
        np.abs(e) / (np.abs(truth) + EPS)
    )

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "NRMSE": float(nrmse),
        "SNR_dB": float(snr),
        "MRE": float(mre),
    }


def print_metrics(name, m):
    print(f"\n===== {name} =====")
    for k, v in m.items():
        print(f"{k}: {v}")


def region_metrics(noisy, pred, clean):
    # Keep the same coarse regions used in the other formal scripts.
    regions = [
        ("Early_1_2", 0, 2),
        ("Middle_3_30", 2, 30),
        ("Late_31_40", 30, 40),
        ("All_1_40", 0, 40),
    ]

    rows = []
    for name, a, b in regions:
        raw_m = metrics(noisy[:, a:b], clean[:, a:b])
        wav_m = metrics(pred[:, a:b], clean[:, a:b])

        rows.append({
            "Region": name,
            "StartChannel": a + 1,
            "EndChannel": b,
            "Raw_RMSE": raw_m["RMSE"],
            "Wavelet_RMSE": wav_m["RMSE"],
            "Raw_NRMSE": raw_m["NRMSE"],
            "Wavelet_NRMSE": wav_m["NRMSE"],
            "Raw_SNR_dB": raw_m["SNR_dB"],
            "Wavelet_SNR_dB": wav_m["SNR_dB"],
            "Raw_MRE": raw_m["MRE"],
            "Wavelet_MRE": wav_m["MRE"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 88)
    print("FORMAL TRADITIONAL BASELINE: SWT-Sym6 soft-threshold, E2 measured-noise-contaminated ATEM")
    print("=" * 88)
    print("Test file:", TEST_DAT)
    print("Training performed by this script: NO")
    print("Wavelet:", WAVELET)
    print("Transform: SWT")
    print("Level:", LEVEL)
    print("Threshold mode:", THRESHOLD_MODE)
    print("Threshold estimator: MAD + universal threshold")
    print("Threshold scale:", THRESHOLD_SCALE)

    noisy, clean = load_test_dat(TEST_DAT)
    print("Test samples:", len(noisy))

    # Warm-up
    _ = wavelet_denoise_one(noisy[0])

    t0 = time.perf_counter()
    pred, thresholds, sigmas = wavelet_denoise_batch(noisy)
    dt = time.perf_counter() - t0

    raw_m = metrics(noisy, clean)
    wav_m = metrics(pred, clean)

    print_metrics("Raw", raw_m)
    print_metrics("Wavelet", wav_m)

    rmse_reduction = (
        (raw_m["RMSE"] - wav_m["RMSE"])
        / raw_m["RMSE"] * 100.0
    )
    snr_gain = wav_m["SNR_dB"] - raw_m["SNR_dB"]
    mre_change = (
        (wav_m["MRE"] - raw_m["MRE"])
        / raw_m["MRE"] * 100.0
    )

    print("\n===== Gain summary =====")
    print("Wavelet_RMSE_reduction_vs_Raw_pct:", rmse_reduction)
    print("Wavelet_SNR_gain_vs_Raw_dB:", snr_gain)
    print("Wavelet_MRE_change_vs_Raw_pct:", mre_change)

    # --------------------------------------------------------
    # Sample-level harmful modification
    # --------------------------------------------------------
    raw_sample_rmse = np.sqrt(
        np.mean((noisy - clean) ** 2, axis=1)
    )
    wav_sample_rmse = np.sqrt(
        np.mean((pred - clean) ** 2, axis=1)
    )

    harmful = wav_sample_rmse > raw_sample_rmse
    improved = wav_sample_rmse < raw_sample_rmse
    tied = ~(harmful | improved)

    delta = raw_sample_rmse - wav_sample_rmse

    print("\n===== Sample-level harmful modification =====")
    print("Wavelet_sample_harmful_rate:", float(np.mean(harmful)))
    print("Wavelet_sample_improved_rate:", float(np.mean(improved)))
    print("Wavelet_sample_tied_rate:", float(np.mean(tied)))
    print(
        "Mean_raw_minus_Wavelet_sample_RMSE:",
        float(np.mean(delta)),
    )
    print(
        "Median_raw_minus_Wavelet_sample_RMSE:",
        float(np.median(delta)),
    )

    # --------------------------------------------------------
    # Save overall metrics
    # --------------------------------------------------------
    pd.DataFrame([
        {"Method": "Raw", **raw_m},
        {"Method": "Wavelet", **wav_m},
    ]).to_csv(OVERALL_OUT, index=False)

    pd.DataFrame([{
        "Wavelet_RMSE_reduction_vs_Raw_pct": rmse_reduction,
        "Wavelet_SNR_gain_vs_Raw_dB": snr_gain,
        "Wavelet_MRE_change_vs_Raw_pct": mre_change,
    }]).to_csv(GAIN_OUT, index=False)

    # --------------------------------------------------------
    # Sample metrics
    # --------------------------------------------------------
    pd.DataFrame({
        "Sample": np.arange(1, len(noisy) + 1),
        "Raw_RMSE": raw_sample_rmse,
        "Wavelet_RMSE": wav_sample_rmse,
        "Raw_minus_Wavelet_RMSE": delta,
        "Wavelet_harmful_modification": harmful.astype(int),
        "Wavelet_improved": improved.astype(int),
        "Wavelet_threshold": thresholds,
        "Wavelet_sigma_MAD": sigmas,
    }).to_csv(SAMPLE_OUT, index=False)

    # --------------------------------------------------------
    # Channel metrics
    # --------------------------------------------------------
    channel_rows = []

    for c in range(N_CH):
        raw_c = metrics(
            noisy[:, c:c+1],
            clean[:, c:c+1],
        )
        wav_c = metrics(
            pred[:, c:c+1],
            clean[:, c:c+1],
        )

        channel_rows.append({
            "Channel": c + 1,
            "Raw_RMSE": raw_c["RMSE"],
            "Wavelet_RMSE": wav_c["RMSE"],
            "Raw_NRMSE": raw_c["NRMSE"],
            "Wavelet_NRMSE": wav_c["NRMSE"],
            "Raw_SNR_dB": raw_c["SNR_dB"],
            "Wavelet_SNR_dB": wav_c["SNR_dB"],
            "Raw_MRE": raw_c["MRE"],
            "Wavelet_MRE": wav_c["MRE"],
            "Wavelet_RMSE_reduction_vs_Raw_pct":
                (raw_c["RMSE"] - wav_c["RMSE"])
                / raw_c["RMSE"] * 100.0,
        })

    pd.DataFrame(channel_rows).to_csv(
        CHANNEL_OUT,
        index=False,
    )

    region_metrics(
        noisy,
        pred,
        clean,
    ).to_csv(
        REGION_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Full predictions for later plotting
    # --------------------------------------------------------
    pred_df = pd.DataFrame(
        np.hstack([noisy, pred, clean])
    )

    pred_df.columns = (
        [f"Noisy_{i+1}" for i in range(N_CH)]
        + [f"Wavelet_{i+1}" for i in range(N_CH)]
        + [f"Clean_{i+1}" for i in range(N_CH)]
    )

    pred_df.to_csv(PRED_OUT, index=False)

    np.savez_compressed(
        NPZ_OUT,
        noisy=noisy,
        wavelet=pred,
        clean=clean,
        thresholds=thresholds,
        sigma_mad=sigmas,
    )

    # --------------------------------------------------------
    # Efficiency and metadata
    # --------------------------------------------------------
    pd.DataFrame([{
        "Method": "Wavelet_SWT_Sym6_L3_soft",
        "TestSamples": len(noisy),
        "TotalInference_s": dt,
        "PerSample_ms": dt / len(noisy) * 1000.0,
        "TrainableParameters": 0,
    }]).to_csv(EFF_OUT, index=False)

    metadata = {
        "method": "stationary_wavelet_soft_threshold",
        "wavelet": WAVELET,
        "level": LEVEL,
        "threshold_mode": THRESHOLD_MODE,
        "threshold_scale": THRESHOLD_SCALE,
        "sigma_estimator": "MAD of finest SWT detail",
        "threshold_formula": "sigma*sqrt(2*ln(N))",
        "test_file": str(TEST_DAT),
        "test_samples": int(len(noisy)),
        "training_required": False,
        "clean_used_by_denoiser": False,
        "clean_used_for_evaluation_only": True,
        "output_directory": str(OUT_DIR),
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n===== Efficiency =====")
    print("Total inference time (s):", dt)
    print("Per sample (ms):", dt / len(noisy) * 1000.0)

    print("\nSaved outputs:")
    for p in [
        OVERALL_OUT,
        GAIN_OUT,
        SAMPLE_OUT,
        CHANNEL_OUT,
        REGION_OUT,
        PRED_OUT,
        NPZ_OUT,
        EFF_OUT,
        META_OUT,
    ]:
        print(" -", p)

    print("\n" + "=" * 88)
    print("FORMAL WAVELET BASELINE FINISHED")
    print("=" * 88)
