from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "wasserstein_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, text, dashed=False, fs=10.2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        fill=False,
        linewidth=1.2,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, label=None):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.02, label, ha="center", fontsize=9)


box(0.04, 0.68, 0.16, 0.16, "Measured noise\n(training subset)")
box(0.27, 0.68, 0.18, 0.16, "Wasserstein\nnoise generator")
box(0.52, 0.68, 0.16, 0.16, "Generated\nnoise vectors")
arrow(0.20, 0.76, 0.27, 0.76, "training")
arrow(0.45, 0.76, 0.52, 0.76)

box(0.04, 0.38, 0.16, 0.16, "Measured noise\n(training subset)")
box(0.27, 0.38, 0.18, 0.16, "Noise pool for B\n(real + generated)")
box(0.04, 0.10, 0.16, 0.16, "Clean ATEM\nforward responses")

box(
    0.52, 0.34, 0.20, 0.20,
    "Synthesis of noisy-clean pairs\n"
    r"$x_i=s_i+(n_i/20)s_{22}$" "\n"
    r"$y_i=s_i$",
    fs=9.2,
)
box(0.79, 0.34, 0.17, 0.20, "Denoising model\ntraining")

arrow(0.20, 0.46, 0.27, 0.46)
arrow(0.60, 0.68, 0.39, 0.54)
arrow(0.45, 0.46, 0.52, 0.44)
arrow(0.20, 0.18, 0.52, 0.38)
arrow(0.72, 0.44, 0.79, 0.44)

box(
    0.52, 0.08, 0.20, 0.14,
    "Held-out measured noise\n(validation / test)",
    dashed=True, fs=9.1,
)
box(
    0.79, 0.08, 0.17, 0.14,
    "Generalization\nevaluation",
    dashed=True, fs=9.1,
)
arrow(0.72, 0.15, 0.79, 0.15)

ax.text(
    0.50, 0.015,
    "Validation and test noise are never used for generator or denoiser training.",
    ha="center", va="bottom", fontsize=9.0,
)

fig.tight_layout()

for ext in ["png", "pdf", "svg"]:
    out = OUT_DIR / f"Fig_Wasserstein_augmentation_workflow.{ext}"
    if ext == "png":
        fig.savefig(out, dpi=600, bbox_inches="tight")
    else:
        fig.savefig(out, bbox_inches="tight")

plt.show()
