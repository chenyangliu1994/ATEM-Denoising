# -*- coding: utf-8 -*-
"""
17 - Prepare E2 no-WGAN same-N ablation training set.

Purpose
-------
Build a control dataset with the SAME number of training samples and the SAME
clean-response rows as train_B_wgan_aug.dat, while replacing the WGAN-generated
noise half with measured-training-noise patterns resampled with replacement.

Frozen formal dataset organization
----------------------------------
train_B_wgan_aug.dat contains two equal halves:
    first half  : measured-training-noise synthesized samples
    second half : WGAN-generated-noise synthesized samples

For each row, the formal synthesis rule can be written as:
    x = s + eta * s_22
where eta = (x-s)/s_22 is the normalized injected-noise pattern and s_22 is
clean channel 22 (Python index 21).

Construction
------------
1) Keep the first half unchanged.
2) Extract eta from the first-half measured-noise samples.
3) Resample those eta rows with replacement (seed=2024).
4) Apply the resampled measured-noise eta to the SECOND-half clean responses.

Therefore:
- total N is unchanged;
- every clean response stays in exactly the same row/order as the Full WGAN set;
- only the source of the second-half noise patterns changes:
      WGAN-generated -> resampled measured training noise.
- validation/test files are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SCENARIO_DIR = Path(r"E1_屏蔽室_16次叠加")
SOURCE_DAT = SCENARIO_DIR / "train_B_wgan_aug.dat"
OUT_DAT = SCENARIO_DIR / "train_B_real_resampled_sameN.dat"
META_JSON = SCENARIO_DIR / "train_B_real_resampled_sameN_metadata.json"
INDICES_NPY = SCENARIO_DIR / "train_B_real_resampled_sameN_indices.npy"

SEED = 2024
N_CH = 40
S22_INDEX = 21  # channel 22, zero-based index
EXPECTED_TOTAL_N = 22672
EXPECTED_HALF_N = 11336


def load_dat(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 80:
        raise ValueError(f"{path}: expected N x 80, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: NaN/Inf detected")
    return a


def main() -> None:
    print("=" * 88)
    print("PREPARE ABLATION DATA: E1 NO-WGAN, SAME-N MEASURED-NOISE RESAMPLING")
    print("=" * 88)
    print("Source:", SOURCE_DAT)
    print("Output:", OUT_DAT)
    print("Seed:", SEED)
    print("Validation/test accessed: NO")

    a = load_dat(SOURCE_DAT)
    n = len(a)
    if n != EXPECTED_TOTAL_N:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_N} rows in formal E2 train_B_wgan_aug.dat, got {n}. "
            "Stop: do not guess the real/WGAN split."
        )
    if n % 2 != 0:
        raise ValueError(f"Expected even row count, got {n}")

    half = n // 2
    if half != EXPECTED_HALF_N:
        raise ValueError(f"Expected half size {EXPECTED_HALF_N}, got {half}")

    noisy = a[:, :N_CH]
    clean = a[:, N_CH:]

    noisy_real = noisy[:half]
    clean_real = clean[:half]
    clean_second = clean[half:]

    s22_real = clean_real[:, S22_INDEX]
    min_abs_s22 = float(np.min(np.abs(s22_real)))
    if min_abs_s22 < 1e-30:
        raise ValueError(
            f"Clean channel-22 contains values too close to zero (min abs={min_abs_s22:.3e}); "
            "cannot safely recover normalized measured-noise patterns."
        )

    # eta is exactly the normalized injected-noise pattern under the formal synthesis rule.
    eta_real = (noisy_real - clean_real) / s22_real[:, None]
    if not np.isfinite(eta_real).all():
        raise ValueError("Recovered measured-noise eta contains NaN/Inf")

    rng = np.random.default_rng(SEED)
    resampled_idx = rng.integers(0, half, size=half, endpoint=False)
    eta_resampled = eta_real[resampled_idx]

    # Preserve the Full-set second-half clean responses row-for-row.
    s22_second = clean_second[:, S22_INDEX]
    noisy_second_no_wgan = clean_second + eta_resampled * s22_second[:, None]

    out_noisy = np.vstack([noisy_real, noisy_second_no_wgan])
    out_clean = clean.copy()
    out = np.hstack([out_noisy, out_clean])

    if out.shape != a.shape:
        raise RuntimeError(f"Output shape mismatch: {out.shape} vs source {a.shape}")
    if not np.isfinite(out).all():
        raise RuntimeError("Output contains NaN/Inf")

    # Strong controls for the ablation design.
    clean_max_abs_diff = float(np.max(np.abs(out_clean - clean)))
    first_half_noisy_max_abs_diff = float(np.max(np.abs(out_noisy[:half] - noisy[:half])))
    unique_resampled_source_rows = int(np.unique(resampled_idx).size)

    np.savetxt(OUT_DAT, out, fmt="%.18e")
    np.save(INDICES_NPY, resampled_idx)

    metadata = {
        "experiment": "E2_no_WGAN_sameN_measured_noise_resampling_dataset",
        "source_dat": str(SOURCE_DAT),
        "output_dat": str(OUT_DAT),
        "seed": SEED,
        "total_samples": int(n),
        "first_half_measured_samples_kept": int(half),
        "second_half_clean_rows_reused": int(half),
        "second_half_noise_source": "resampled normalized measured-training-noise patterns",
        "clean_channel_for_scaling": 22,
        "formula": "x = s + eta * s_22; eta=(x-s)/s_22",
        "clean_max_abs_diff_vs_full_source": clean_max_abs_diff,
        "first_half_noisy_max_abs_diff_vs_full_source": first_half_noisy_max_abs_diff,
        "unique_measured_eta_rows_drawn": unique_resampled_source_rows,
        "validation_accessed": False,
        "test_accessed": False,
    }
    META_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== Construction check =====")
    print("Source shape:", a.shape)
    print("Output shape:", out.shape)
    print("Measured half:", half)
    print("Replacement half:", half)
    print("Clean max abs diff vs Full source:", clean_max_abs_diff)
    print("First-half noisy max abs diff vs Full source:", first_half_noisy_max_abs_diff)
    print("Unique measured eta rows drawn:", unique_resampled_source_rows, "/", half)
    print("Saved:", OUT_DAT)
    print("Saved:", META_JSON)
    print("Saved:", INDICES_NPY)
    print("=" * 88)
    print("DATA PREPARATION FINISHED -- NO VALIDATION/TEST DATA WERE USED")
    print("=" * 88)


if __name__ == "__main__":
    main()
