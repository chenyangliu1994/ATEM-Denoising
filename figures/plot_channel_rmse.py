from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import load_scenario_config
from src.evaluation.metrics import per_channel_rmse


SCENARIOS = [
    ("E1", "(a) Shielded room, 16-stack"),
    ("E2", "(b) Shielded room, single"),
    ("E3", "(c) Urban roadside, 16-stack"),
    ("E4", "(d) Urban roadside, single"),
]

plt.rcParams["font.family"] = "Times New Roman"
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharex=True)
axes = axes.ravel()

for ax, (sid, title) in zip(axes, SCENARIOS):
    scfg = load_scenario_config(sid, ROOT)
    base = ROOT / "outputs" / "06_test_ablation" / scfg["slug"]

    A = np.loadtxt(base / "test_predictions_A.dat")
    B = np.loadtxt(base / "test_predictions_B.dat")

    pred_A, clean_A = A[:, 40:80], A[:, 80:120]
    pred_B, clean_B = B[:, 40:80], B[:, 80:120]

    if not np.array_equal(clean_A, clean_B):
        raise RuntimeError(f"{sid}: A/B targets differ.")

    x = np.arange(1, 41)
    ax.plot(x, per_channel_rmse(pred_A, clean_A), marker="o", markevery=2, label="Measured-noise resampling")
    ax.plot(x, per_channel_rmse(pred_B, clean_B), marker="s", markevery=2, label="Wasserstein augmentation")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.axvline(2.5, linestyle=":", linewidth=0.8)
    ax.axvline(30.5, linestyle=":", linewidth=0.8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.20)

axes[0].set_ylabel("RMSE (V)")
axes[2].set_ylabel("RMSE (V)")
axes[2].set_xlabel("Time channel")
axes[3].set_xlabel("Time channel")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=[0,0,1,0.95])

out = ROOT / "results" / "wasserstein_ablation" / "Fig_per_channel_RMSE.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.show()
