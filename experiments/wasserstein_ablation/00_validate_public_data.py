from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_scenario_config, load_measured_noise_80col, load_clean_40col


def main():
    common = load_common_config(ROOT)
    clean_path = ROOT / common["clean_response_file"]

    clean = load_clean_40col(clean_path)
    print("Clean responses:", clean_path)
    print("Shape:", clean.shape)
    if clean.shape != (320000, 40):
        raise ValueError("Expected exact ablation clean matrix: 320000 x 40.")

    for sid in ["E1", "E2", "E3", "E4"]:
        cfg = load_scenario_config(sid, ROOT)
        p = ROOT / cfg["measured_noise_file"]
        time, noise = load_measured_noise_80col(p)
        print(f"{sid}: {p.name} -> {noise.shape}")
        if noise.shape != (int(cfg["expected_total_rows"]), 40):
            raise ValueError(f"{sid}: unexpected row count.")

    print("\nAll public input files passed structural checks.")


if __name__ == "__main__":
    main()
