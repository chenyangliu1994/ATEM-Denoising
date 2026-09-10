from pathlib import Path
import sys
import argparse
import csv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_scenario_config


def count_nonempty_lines(path):
    n = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def split_file(source, out_dir, train_ratio=0.80, val_ratio=0.10):
    total = count_nonempty_lines(source)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train_real_noise.dat",
        "val": out_dir / "val_real_noise.dat",
        "test": out_dir / "test_real_noise.dat",
    }

    with Path(source).open("r", encoding="utf-8", errors="ignore") as fin, \
         paths["train"].open("w", encoding="utf-8") as ftrain, \
         paths["val"].open("w", encoding="utf-8") as fval, \
         paths["test"].open("w", encoding="utf-8") as ftest:

        idx = 0
        for line in fin:
            if not line.strip():
                continue
            if idx < n_train:
                ftrain.write(line)
            elif idx < n_train + n_val:
                fval.write(line)
            else:
                ftest.write(line)
            idx += 1

    return total, n_train, n_val, n_test, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    common = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)

    source = ROOT / scfg["measured_noise_file"]
    if not source.exists():
        raise FileNotFoundError(source)

    out_dir = ROOT / "outputs" / "01_split_real_noise" / scfg["slug"]
    total, n_train, n_val, n_test, paths = split_file(
        source,
        out_dir,
        float(common["split"]["train_ratio"]),
        float(common["split"]["val_ratio"]),
    )

    if total != int(scfg["expected_total_rows"]):
        raise ValueError(
            f"{args.scenario}: expected {scfg['expected_total_rows']} rows, got {total}"
        )

    manifest = out_dir / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scenario","total","train","val","test"])
        w.writerow([args.scenario,total,n_train,n_val,n_test])

    print(f"{args.scenario}: Total={total}, Train={n_train}, Val={n_val}, Test={n_test}")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
