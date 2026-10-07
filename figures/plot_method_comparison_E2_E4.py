from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "method_comparison" / "method_comparison_summary.csv"

df = pd.read_csv(CSV)
methods = ["Wavelet", "TEMDnet", "TEM1Dformer_reimplementation", "FCED_H512", "FCED_H512_SRCG"]

for scenario in ["E2", "E4"]:
    sub = df[(df["Scenario"] == scenario) & (df["Method"].isin(methods))].copy()
    sub = sub.set_index("Method").reindex(methods).reset_index()
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, sub["RMSE_reduction_vs_Raw_pct"])
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Wavelet", "TEMDnet", "TEM1Dformer", "FC-ED", "FC-ED+SRCG"], rotation=18)
    ax.set_ylabel("RMSE reduction vs. raw (%)")
    ax.set_title(f"{scenario}: held-out measured-noise test")
    fig.tight_layout()
    out = ROOT / "figures" / f"method_comparison_{scenario}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)
