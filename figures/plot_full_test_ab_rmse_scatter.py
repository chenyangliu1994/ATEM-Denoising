# -*- coding: utf-8 -*-
"""
SCI figure: full-test paired A/B per-sample RMSE scatter.

Each point is one held-out test sample.
x-axis: RMSE of A (measured-noise resampling)
y-axis: RMSE of B (Wasserstein augmentation)

Interpretation:
    point below y=x  -> B has lower RMSE (better)
    point above y=x  -> A has lower RMSE (better)

All test samples are used. No sample selection.

Reads:
    06_test_ablation/<scenario>/test_predictions_A.dat
    06_test_ablation/<scenario>/test_predictions_B.dat

Each file:
    noisy40 | prediction40 | clean40

Outputs:
    07_figures/Fig13_FullTest_AB_RMSE_Scatter.png
    07_figures/Fig13_FullTest_AB_RMSE_Scatter.pdf
    07_figures/Fig13_FullTest_AB_RMSE_Scatter.svg
    07_figures/Fig13_full_test_scatter_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "07_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("E1_屏蔽室_16次叠加", "(a) Shielded room – 16-stack"),
    ("E2_屏蔽室_1次叠加", "(b) Shielded room – single acquisition"),
    ("E3_城市道路旁_16次叠加", "(c) Urban roadside – 16-stack"),
    ("E4_城市道路旁_1次叠加", "(d) Urban roadside – single acquisition"),
]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

LABEL_SIZE = 10.5
TICK_SIZE = 9
TITLE_SIZE = 10.5
ANNOT_SIZE = 9.3


def load_prediction_file(path):
    arr = np.loadtxt(path)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != 120:
        raise ValueError(
            f"{path}: expected 120 columns "
            "(noisy40 | pred40 | clean40)"
        )

    noisy = arr[:, 0:40]
    pred = arr[:, 40:80]
    clean = arr[:, 80:120]
    return noisy, pred, clean


def row_rmse(pred, clean):
    return np.sqrt(np.mean((pred - clean) ** 2, axis=1))


summary_rows = []

fig, axes = plt.subplots(
    2, 2,
    figsize=(8.6, 7.4),
    constrained_layout=True
)

for ax, (scenario, title) in zip(axes.ravel(), SCENARIOS):

    base = ROOT / "06_test_ablation" / scenario

    noisy_a, pred_a, clean_a = load_prediction_file(
        base / "test_predictions_A.dat"
    )
    noisy_b, pred_b, clean_b = load_prediction_file(
        base / "test_predictions_B.dat"
    )

    if not np.array_equal(noisy_a, noisy_b):
        raise RuntimeError(f"{scenario}: A/B noisy inputs differ.")

    if not np.array_equal(clean_a, clean_b):
        raise RuntimeError(f"{scenario}: A/B clean targets differ.")

    rmse_a = row_rmse(pred_a, clean_a)
    rmse_b = row_rmse(pred_b, clean_b)

    valid = (
        np.isfinite(rmse_a)
        & np.isfinite(rmse_b)
        & (rmse_a > 0)
        & (rmse_b > 0)
    )

    rmse_a = rmse_a[valid]
    rmse_b = rmse_b[valid]

    b_better = rmse_b < rmse_a
    pct_b_better = 100.0 * np.mean(b_better)

    global_a = float(np.sqrt(np.mean((pred_a - clean_a) ** 2)))
    global_b = float(np.sqrt(np.mean((pred_b - clean_b) ** 2)))
    global_reduction = 100.0 * (global_a - global_b) / global_a

    # Determine common limits within this panel.
    low = min(rmse_a.min(), rmse_b.min())
    high = max(rmse_a.max(), rmse_b.max())

    # Slight margin in log space.
    low_plot = 10 ** (np.floor(np.log10(low)) - 0.05)
    high_plot = 10 ** (np.ceil(np.log10(high)) + 0.05)

    ax.scatter(
        rmse_a,
        rmse_b,
        s=8,
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )

    # Equality line: B = A.
    ax.plot(
        [low_plot, high_plot],
        [low_plot, high_plot],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low_plot, high_plot)
    ax.set_ylim(low_plot, high_plot)

    # Equal geometric scale makes the y=x comparison visually honest.
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=TITLE_SIZE, pad=6)

    ax.text(
        0.04,
        0.94,
        f"B lower RMSE: {pct_b_better:.1f}%\n"
        f"Overall RMSE reduction: {global_reduction:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.28",
            fill=False,
            linewidth=0.6
        ),
    )

    # Direction labels.
    ax.text(
        0.70, 0.10,
        "B better",
        transform=ax.transAxes,
        fontsize=8.5,
        rotation=0,
        ha="center",
        va="center",
    )
    ax.text(
        0.20, 0.78,
        "A better",
        transform=ax.transAxes,
        fontsize=8.5,
        rotation=0,
        ha="center",
        va="center",
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.4,
        alpha=0.18,
    )

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    summary_rows.append({
        "Scenario": scenario,
        "N_test": len(rmse_a),
        "B_lower_RMSE_percent": pct_b_better,
        "Overall_RMSE_A": global_a,
        "Overall_RMSE_B": global_b,
        "Overall_RMSE_reduction_percent": global_reduction,
        "Median_RMSE_A": float(np.median(rmse_a)),
        "Median_RMSE_B": float(np.median(rmse_b)),
    })

# Shared labels
fig.supxlabel(
    "Per-sample RMSE of measured-noise resampling model (A)",
    fontsize=LABEL_SIZE
)
fig.supylabel(
    "Per-sample RMSE of Wasserstein-augmentation model (B)",
    fontsize=LABEL_SIZE
)

stem = OUT_DIR / "Fig13_FullTest_AB_RMSE_Scatter"

fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight")
fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
fig.savefig(str(stem) + ".svg", bbox_inches="tight")

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "Fig13_full_test_scatter_summary.csv"
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

print("=" * 96)
print("FULL-TEST PAIRED A/B RMSE SCATTER")
print("=" * 96)
print(summary_df.to_string(index=False))
print()
print("Saved:")
print(str(stem) + ".png")
print(str(stem) + ".pdf")
print(str(stem) + ".svg")
print(summary_csv)
print("=" * 96)

plt.show()
