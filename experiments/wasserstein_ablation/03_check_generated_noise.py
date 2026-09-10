from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_scenario_config, load_measured_noise_80col


def corr_mae(a, b):
    ca = np.corrcoef(a, rowvar=False)
    cb = np.corrcoef(b, rowvar=False)
    mask = ~np.eye(ca.shape[0], dtype=bool)
    return float(np.mean(np.abs(ca[mask] - cb[mask])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    scfg = load_scenario_config(args.scenario, ROOT)
    real_file = ROOT / "outputs" / "01_split_real_noise" / scfg["slug"] / "train_real_noise.dat"
    gen_file = ROOT / "outputs" / "02_wasserstein_generation" / scfg["slug"] / "generated_noise.dat"

    _, real = load_measured_noise_80col(real_file)
    _, gen = load_measured_noise_80col(gen_file)

    per_ch_real = np.std(real, axis=0)
    per_ch_gen = np.std(gen, axis=0)

    summary = {
        "scenario": args.scenario,
        "real_global_std": float(np.std(real)),
        "generated_global_std": float(np.std(gen)),
        "global_std_ratio_gen_over_real": float(np.std(gen) / np.std(real)),
        "median_channel_std_ratio": float(np.median(per_ch_gen / per_ch_real)),
        "correlation_matrix_MAE": corr_mae(real, gen),
        "flattened_wasserstein_distance": float(
            wasserstein_distance(real.ravel(), gen.ravel())
        ),
    }

    out_dir = ROOT / "outputs" / "03_generator_quality" / scfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "generator_quality_summary.csv", index=False)

    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
