from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "wasserstein_ablation" / "comparison_summary.csv")

labels = [
    "E1  Shielded room, 16-stack",
    "E2  Shielded room, single",
    "E3  Urban roadside, 16-stack",
    "E4  Urban roadside, single",
]

plt.rcParams["font.family"] = "Times New Roman"
y = np.arange(4)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

vals = df["RMSE_improvement_pct"].to_numpy()
for yi, v in zip(y, vals):
    axes[0].hlines(yi, 0, v)
    axes[0].scatter(v, yi, s=55)
    axes[0].text(v+1.2, yi, f"{v:.1f}%", va="center")
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels)
axes[0].set_xlabel("RMSE reduction (%)")
axes[0].set_title("(a) Overall RMSE improvement")
axes[0].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.2)

vals = df["SNR_gain_dB"].to_numpy()
for yi, v in zip(y, vals):
    axes[1].hlines(yi, 0, v)
    axes[1].scatter(v, yi, s=55)
    axes[1].text(v+0.15, yi, f"+{v:.2f} dB", va="center")
axes[1].set_yticks(y)
axes[1].set_yticklabels([])
axes[1].set_xlabel("SNR gain (dB)")
axes[1].set_title("(b) Overall SNR improvement")
axes[1].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.2)

fig.tight_layout()
out = ROOT / "results" / "wasserstein_ablation" / "Fig_overall_benefit.png"
fig.savefig(out, dpi=600, bbox_inches="tight")
plt.show()
