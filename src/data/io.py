from pathlib import Path
import numpy as np
import pandas as pd
import yaml


SCENARIO_CONFIG_FILES = {
    "E1": "E1_shielded_16stack.yaml",
    "E2": "E2_shielded_single.yaml",
    "E3": "E3_urban_roadside_16stack.yaml",
    "E4": "E4_urban_roadside_single.yaml",
}


def repo_root():
    return Path(__file__).resolve().parents[2]


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_common_config(root=None):
    root = Path(root) if root else repo_root()
    return load_yaml(root / "configs" / "common.yaml")


def load_scenario_config(scenario_id, root=None):
    scenario_id = scenario_id.upper()
    if scenario_id not in SCENARIO_CONFIG_FILES:
        raise ValueError(f"Unknown scenario {scenario_id}; choose E1, E2, E3, or E4.")
    root = Path(root) if root else repo_root()
    return load_yaml(root / "configs" / SCENARIO_CONFIG_FILES[scenario_id])


def load_measured_noise_80col(path, dtype=np.float64):
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 80:
        raise ValueError(f"{path}: expected 80 columns, got {df.shape[1]}")
    time = df.iloc[:, :40].to_numpy(dtype=dtype)
    noise = df.iloc[:, 40:80].to_numpy(dtype=dtype)
    if not np.isfinite(time).all() or not np.isfinite(noise).all():
        raise ValueError(f"{path}: NaN/Inf detected.")
    return time, noise


def load_clean_40col(path, dtype=np.float64):
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 40:
        raise ValueError(f"{path}: expected 40 columns, got {df.shape[1]}")
    arr = df.to_numpy(dtype=dtype)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: NaN/Inf detected.")
    return arr


def load_supervised_80col(path, dtype=np.float64):
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 80:
        raise ValueError(f"{path}: expected 80 columns, got {df.shape[1]}")
    noisy = df.iloc[:, :40].to_numpy(dtype=dtype)
    clean = df.iloc[:, 40:80].to_numpy(dtype=dtype)
    if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
        raise ValueError(f"{path}: NaN/Inf detected.")
    return noisy, clean


def save_dat(path, arr, fmt="%.12e"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, fmt=fmt)
