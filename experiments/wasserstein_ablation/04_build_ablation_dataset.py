from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import (
    load_common_config,
    load_scenario_config,
    load_measured_noise_80col,
    load_clean_40col,
    save_dat,
)
from src.data.synthesis import synthesize_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    common = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)
    seed = int(common["seed"])

    clean = load_clean_40col(ROOT / common["clean_response_file"])

    split_dir = ROOT / "outputs" / "01_split_real_noise" / scfg["slug"]
    _, train_real = load_measured_noise_80col(split_dir / "train_real_noise.dat")
    _, val_real = load_measured_noise_80col(split_dir / "val_real_noise.dat")
    _, test_real = load_measured_noise_80col(split_dir / "test_real_noise.dat")

    gen_file = ROOT / "outputs" / "02_wasserstein_generation" / scfg["slug"] / "generated_noise.dat"
    _, generated = load_measured_noise_80col(gen_file)

    N = len(train_real)
    if len(generated) != N:
        raise ValueError("Generated-noise count must equal measured training-noise count.")

    # A: N original measured + N bootstrap-resampled measured.
    rng = np.random.default_rng(seed)
    resample_idx = rng.integers(0, N, size=N)
    noise_A = np.vstack([train_real, train_real[resample_idx]])

    # B: the same N original measured + N generated.
    noise_B = np.vstack([train_real, generated])

    train_total = 2 * N
    n_val, n_test = len(val_real), len(test_real)
    required_clean = train_total + n_val + n_test

    if len(clean) < required_clean:
        raise ValueError(
            f"Clean-response file needs at least {required_clean} rows; got {len(clean)}."
        )

    # Exact contiguous clean-response allocation used in the controlled experiments.
    clean_train = clean[:train_total]
    clean_val = clean[train_total:train_total + n_val]
    clean_test = clean[train_total + n_val:required_clean]

    sc = common["synthesis"]
    kwargs = {
        "scale_divisor": float(sc["scale_divisor"]),
        "reference_channel_1based": int(sc["reference_channel_1based"]),
    }

    train_A, scaled_A = synthesize_pairs(clean_train, noise_A, **kwargs)
    train_B, scaled_B = synthesize_pairs(clean_train, noise_B, **kwargs)
    val, scaled_val = synthesize_pairs(clean_val, val_real, **kwargs)
    test, scaled_test = synthesize_pairs(clean_test, test_real, **kwargs)

    out_dir = ROOT / "outputs" / "04_ablation_datasets" / scfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    save_dat(out_dir / "train_A_measured_resampling.dat", train_A)
    save_dat(out_dir / "train_B_wasserstein_aug.dat", train_B)
    save_dat(out_dir / "val_real_only.dat", val)
    save_dat(out_dir / "test_real_only.dat", test)

    pd.DataFrame({
        "resample_position": np.arange(N),
        "source_train_index_0based": resample_idx,
    }).to_csv(out_dir / "A_resample_indices.csv", index=False)

    checks = {
        "scenario": args.scenario,
        "real_train_noise": N,
        "generated_noise": len(generated),
        "train_A": len(train_A),
        "train_B": len(train_B),
        "val": len(val),
        "test": len(test),
        "AB_first_half_max_diff": float(np.max(np.abs(noise_A[:N] - noise_B[:N]))),
        "AB_clean_label_max_diff": float(np.max(np.abs(train_A[:, 40:] - train_B[:, 40:]))),
        "A_scaled_noise_std": float(np.std(scaled_A)),
        "B_scaled_noise_std": float(np.std(scaled_B)),
        "Val_scaled_noise_std": float(np.std(scaled_val)),
        "Test_scaled_noise_std": float(np.std(scaled_test)),
    }

    pd.DataFrame([checks]).to_csv(out_dir / "construction_summary.csv", index=False)
    print(pd.DataFrame([checks]).to_string(index=False))


if __name__ == "__main__":
    main()
