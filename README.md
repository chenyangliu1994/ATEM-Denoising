# ATEM-Denoising

Reproducible code and data organization for airborne transient electromagnetic
(ATEM) denoising using physics-based clean responses, independently measured
noise, and Wasserstein generative augmentation with gradient-norm penalty.

This repository contains the **denoising component** of the study. The
physics-based generation of theoretical/clean ATEM responses can be released
as a separate repository or module later.

## What is included

- Fixed 40-channel fully connected encoder-decoder denoiser.
- Bounded channel gating (BCG).
- Two-stage R-H training protocol.
- Measured-noise chronological split: 80% train / 10% validation / 10% test.
- Wasserstein noise generator with the archived gradient-norm penalty
  implementation.
- Matched A/B augmentation ablation:
  - **A: measured-noise resampling control**
  - **B: Wasserstein generative augmentation**
- Common held-out measured-noise test evaluation.
- Scripts for reproducing the main ablation figures.

## Important implementation note

The archived generative implementation uses

```python
alpha = torch.randn(...)
```

inside the gradient-norm penalty rather than the canonical
`alpha ~ U(0,1)` interpolation used in standard WGAN-GP. Therefore, this
repository refers to the method as:

> **Wasserstein generative augmentation with gradient-norm penalty**

rather than claiming exact equivalence to canonical WGAN-GP.

## Data convention

The controlled ablation uses:

- Clean responses: `N x 40`
- Measured-noise files: `N x 80`
  - columns 1-40: time channels
  - columns 41-80: measured noise
- Supervised denoising files: `N x 80`
  - columns 1-40: noisy response
  - columns 41-80: clean response

All four scenarios use the same synthesis rule:

\[
x_i = s_i + \frac{n_i}{20}s_{22}, \qquad y_i=s_i.
\]

## Four measured-noise scenarios

| ID | Scenario | Total measured-noise rows | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| E1 | Shielded room, 16-stack | 14,171 | 11,336 | 1,417 | 1,418 |
| E2 | Shielded room, single acquisition | 2,883 | 2,306 | 288 | 289 |
| E3 | Urban roadside, 16-stack | 62,539 | 50,031 | 6,253 | 6,255 |
| E4 | Urban roadside, single acquisition | 44,847 | 35,877 | 4,484 | 4,486 |

The urban-roadside measurements were collected in the actual local
electromagnetic environment; an operating wireless router was present nearby.
This is reported as an environmental condition only and is **not** treated as
a proven causal noise source.

## Matched augmentation ablation

For a scenario with `N` measured training-noise vectors:

**Group A — measured-noise resampling control**

- first `N`: original measured training noise
- second `N`: bootstrap resampling with replacement from the same training pool

**Group B — Wasserstein augmentation**

- first `N`: the same original measured training noise
- second `N`: `N` newly generated noise vectors

A and B therefore contain exactly the same number of training samples and use
the same clean-response sequence. Validation and test sets contain only
held-out measured noise.

## Quick start

Create an environment:

```bash
conda env create -f environment.yml
conda activate atem-denoising
```

or:

```bash
pip install -r requirements.txt
```

Place the public data according to `data/README.md`, then run one scenario:

```bash
python experiments/wasserstein_ablation/01_split_measured_noise.py --scenario E1
python experiments/wasserstein_ablation/02_train_wasserstein_generator.py --scenario E1
python experiments/wasserstein_ablation/03_check_generated_noise.py --scenario E1
python experiments/wasserstein_ablation/04_build_ablation_dataset.py --scenario E1
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group A
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group B
python experiments/wasserstein_ablation/06_evaluate_common_test.py --scenario E1
```

Repeat for E2-E4.

The final test scripts never update model parameters.

## Reproduced test results

The manuscript-level A/B test metrics are provided in:

```text
results/wasserstein_ablation/test_metrics_all_scenarios.csv
results/wasserstein_ablation/comparison_summary.csv
```

## License

MIT License. See `LICENSE`.
