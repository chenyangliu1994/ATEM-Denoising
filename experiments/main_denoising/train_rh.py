from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config
from src.training.rh_trainer import train_rh


def main():
    parser = argparse.ArgumentParser(
        description="Train the fixed ATEM denoiser with the two-stage R-H protocol."
    )
    parser.add_argument("--train", required=True, help="80-column noisy40|clean40 training file")
    parser.add_argument("--val", required=True, help="80-column noisy40|clean40 validation file")
    parser.add_argument("--out", default="outputs/main_denoising")
    parser.add_argument("--name", default="main")
    args = parser.parse_args()

    cfg = load_common_config(ROOT)

    train_file = Path(args.train)
    val_file = Path(args.val)
    if not train_file.is_absolute():
        train_file = ROOT / train_file
    if not val_file.is_absolute():
        val_file = ROOT / val_file

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    model_path, summary = train_rh(
        train_file=train_file,
        val_file=val_file,
        out_dir=out_dir,
        scenario_id=args.name,
        group="main",
        cfg=cfg,
    )

    print("Final model:", model_path)
    print(summary)


if __name__ == "__main__":
    main()
