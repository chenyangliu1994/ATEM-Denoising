from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_scenario_config
from src.training.rh_trainer import train_rh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    parser.add_argument("--group", required=True, choices=["A","B"])
    args = parser.parse_args()

    common = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)

    data_dir = ROOT / "outputs" / "04_ablation_datasets" / scfg["slug"]
    if args.group == "A":
        train_file = data_dir / "train_A_measured_resampling.dat"
        group_name = "A_measured_resampling"
    else:
        train_file = data_dir / "train_B_wasserstein_aug.dat"
        group_name = "B_wasserstein_aug"

    val_file = data_dir / "val_real_only.dat"
    out_dir = ROOT / "outputs" / "05_denoiser_ablation" / scfg["slug"] / group_name

    model_path, summary = train_rh(
        train_file=train_file,
        val_file=val_file,
        out_dir=out_dir,
        scenario_id=args.scenario,
        group=group_name,
        cfg=common,
    )

    print("\nFinished:", model_path)
    print(summary)


if __name__ == "__main__":
    main()
