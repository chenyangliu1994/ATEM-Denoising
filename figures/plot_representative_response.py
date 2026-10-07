from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import load_scenario_config

SCENARIOS = ["E1", "E4"]
plt.rcParams["font.family"] = "Times New Roman"

for sid in SCENARIOS:
    scfg = load_scenario_config(sid, ROOT)
    base = ROOT / "outputs" / "06_test_ablation" / scfg["slug"]
    A = np.loadtxt(base / "test_predictions_A.dat")
    B = np.loadtxt(base / "test_predictions_B.dat")

    noisy = A[:, :40]
    pred_A = A[:, 40:80]
    clean = A[:, 80:120]
    pred_B = B[:, 40:80]

    input_rmse = np.sqrt(np.mean((noisy-clean)**2, axis=1))
    med = np.median(input_rmse)
    idx = int(np.argmin(np.abs(input_rmse-med)))

    x = np.arange(1, 41)
    scale = 1e12

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(x, noisy[idx]*scale, "--", label="Noisy input")
    ax.plot(x, clean[idx]*scale, "-", label="Clean target")
    ax.plot(x, pred_A[idx]*scale, "-.", marker="o", markevery=3, label="Measured-noise resampling")
    ax.plot(x, pred_B[idx]*scale, ":", marker="s", markevery=3, label="Wasserstein augmentation")

    # Signed logarithmic axis preserves negative late-channel values.
    ax.set_yscale("symlog", linthresh=0.05, base=10)
    ax.set_xlabel("Time channel")
    ax.set_ylabel(r"Response ($\times 10^{-12}$ V)")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.20)

    out = ROOT / "results" / "wasserstein_ablation" / f"Fig_representative_{sid}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.show()
