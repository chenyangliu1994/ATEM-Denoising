# ATEM-Denoising

Reproducible code and data organization for airborne transient electromagnetic (ATEM) denoising using physics-based clean responses, independently measured noise, and Wasserstein generative augmentation with a gradient-norm penalty.

This repository contains the **denoising and Wasserstein-augmentation ablation component** of the study. The model-generation and forward-simulation workflow is available in [ATEM-Forward-Model-Library](https://github.com/chenyangliu1994/ATEM-Forward-Model-Library). The clean-response subset required for the controlled denoising ablation is included here.

## What is included

- Fixed 40-channel fully connected encoder-decoder denoiser.
- Bounded channel gating (BCG).
- Two-stage relative-to-hybrid (R-H) training protocol.
- Chronological measured-noise split: 80% train / 10% validation / 10% test.
- Wasserstein noise generator using the archived gradient-norm penalty implementation.
- Matched-size A/B augmentation ablation:
  - **A:** measured-noise resampling control.
  - **B:** Wasserstein generative augmentation.
- Common held-out measured-noise test evaluation.
- Scripts for reproducing the main ablation figures, including the full-test paired A/B RMSE scatter figure.

## Important implementation note

The archived generative implementation uses `alpha = torch.randn(...)` inside the gradient-norm penalty rather than the canonical `alpha ~ U(0,1)` interpolation used in standard WGAN-GP.

Therefore, this repository refers to the method as:

**Wasserstein generative augmentation with gradient-norm penalty**

rather than claiming exact equivalence to canonical WGAN-GP.

## Repository structure

```text
ATEM-Denoising/
├─ configs/
├─ data/
│  ├─ clean_responses/
│  └─ measured_noise/
├─ experiments/
│  ├─ synthetic_combined_noise/
│  └─ wasserstein_ablation/
├─ figures/
│  ├─ draw_augmentation_workflow.py
│  ├─ plot_channel_rmse.py
│  ├─ plot_overall_benefit.py
│  └─ plot_full_test_ab_rmse_scatter.py
├─ outputs/
├─ results/
│  └─ wasserstein_ablation/
├─ src/
│  ├─ augmentation/
│  ├─ data/
│  ├─ evaluation/
│  ├─ losses/
│  ├─ models/
│  └─ training/
├─ environment.yml
├─ requirements.txt
└─ README.md
```

## Data convention

The controlled ablation uses the following data layouts.

### Clean responses

`N × 40`

Each row is one 40-channel physics-based clean ATEM response.

The public clean-response file used by the present ablation contains **320,000 × 40** values and is stored under:

```text
data/clean_responses/
```

### Measured-noise files

`N × 80`

- columns 1–40: time channels
- columns 41–80: measured noise

### Supervised denoising files

`N × 80`

- columns 1–40: noisy response
- columns 41–80: clean response

All four measured-noise scenarios use the same synthesis rule:

$$
x_i=s_i+\frac{n_i}{20}s_{22}, \qquad y_i=s_i
$$

where `s_i` is the clean ATEM response at channel `i`, `n_i` is the corresponding measured/generated noise value, and `s_22` denotes the 22nd clean-response channel used as the common amplitude reference.

## Three representative method-comparison conditions

The following three conditions have been selected for the planned comparison with other denoising methods. Their data-construction code is organized in this repository:

| Condition | Data construction | Location |
|---|---|---|
| Synthetic combined noise (`d=10`) | Generate 40-channel clean forward responses and add atmospheric-pulse, Gaussian, and harmonic noise; save 40 noisy values + 40 clean values per row. | [`experiments/synthetic_combined_noise/`](experiments/synthetic_combined_noise/README.md) |
| E2 — shielded room, single acquisition | Split independently measured noise chronologically and construct paired training, validation, and test data. | `experiments/wasserstein_ablation/01_split_measured_noise.py` and `04_build_ablation_dataset.py` (use `--scenario E2`) |
| E4 — urban roadside, single acquisition | Use the same measured-noise splitting and paired-data construction workflow for the roadside single-acquisition records. | `experiments/wasserstein_ablation/01_split_measured_noise.py` and `04_build_ablation_dataset.py` (use `--scenario E4`) |

The synthetic combined-noise script preserves the historical generation procedure; the generated synthetic dataset is **not included** in this repository. The E2/E4 data-construction scripts also support the separate four-scenario Wasserstein A/B ablation described below. These three data-construction entry points are available, but a **unified train/validation/test protocol and cross-method benchmark results for all three conditions have not yet been added**. Do not interpret the existing A/B ablation results as a comparison with external denoising methods.

## Four measured-noise scenarios

| ID | Scenario | Total measured-noise rows | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| E1 | Shielded room, 16-stack | 14,171 | 11,336 | 1,417 | 1,418 |
| E2 | Shielded room, single acquisition | 2,883 | 2,306 | 288 | 289 |
| E3 | Urban roadside, 16-stack | 62,539 | 50,031 | 6,253 | 6,255 |
| E4 | Urban roadside, single acquisition | 44,847 | 35,877 | 4,484 | 4,486 |

The urban-roadside measurements were collected in the actual local electromagnetic environment. An operating wireless router was present nearby. This is reported only as an environmental condition and is **not** treated as a proven causal noise source.

## Matched augmentation ablation

For a scenario with `N` measured training-noise vectors:

### Group A — measured-noise resampling control

- first `N`: original measured training noise
- second `N`: bootstrap resampling with replacement from the same measured training pool

### Group B — Wasserstein augmentation

- first `N`: the same original measured training noise
- second `N`: `N` newly generated noise vectors

A and B therefore contain exactly the same number of training samples and use the same clean-response sequence.

Validation and test sets contain **only held-out measured noise** and are never used to train either the Wasserstein generator or the denoising model.

## Quick start

Create the environment:

```bash
conda env create -f environment.yml
conda activate atem-denoising
```

or install from:

```bash
pip install -r requirements.txt
```

Place the public data according to `data/README.md`, then run one scenario, for example E1:

```bash
python experiments/wasserstein_ablation/01_split_measured_noise.py --scenario E1
python experiments/wasserstein_ablation/02_train_wasserstein_generator.py --scenario E1
python experiments/wasserstein_ablation/03_check_generated_noise.py --scenario E1
python experiments/wasserstein_ablation/04_build_ablation_dataset.py --scenario E1
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group A
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group B
python experiments/wasserstein_ablation/06_evaluate_common_test.py --scenario E1
```

Repeat for E2–E4.

The final common-test evaluation scripts do **not** update model parameters.

## Reproducing the main ablation figures

The figure scripts are stored in:

```text
figures/
```

The full-test paired A/B RMSE scatter figure is generated with:

```bash
python figures/plot_full_test_ab_rmse_scatter.py
```

For each held-out test sample:

- x-axis: per-sample RMSE of Group A
- y-axis: per-sample RMSE of Group B
- points below the equality line `y = x`: lower RMSE for Wasserstein augmentation
- points above the equality line: lower RMSE for measured-noise resampling

This figure uses **all held-out test samples** and does not rely on selected representative examples.

## Reproduced test results

The manuscript-level A/B test metrics are provided in:

```text
results/wasserstein_ablation/test_metrics_all_scenarios.csv
results/wasserstein_ablation/comparison_summary.csv
```

Additional generator-quality summaries are provided in:

```text
results/wasserstein_ablation/generator_quality_summary.csv
```

## Code and data availability

Repository:

https://github.com/chenyangliu1994/ATEM-Denoising

The repository contains the denoising code, Wasserstein-augmentation ablation workflow, measured-noise data, the clean-response subset used by the controlled ablation, configuration files, figure scripts, and summary results.

## License

MIT License.
