# ATEM-Denoising

Code, data, and experiment records for 40-channel airborne transient electromagnetic (ATEM) response denoising. The current model uses a fully connected encoder-decoder (FC-ED) followed by a selective residual correction gate (SRCG).

## Method

FC-ED first produces a candidate response \(z\). SRCG then estimates a channel-wise factor \(\alpha\) that controls how much of the FC-ED correction is retained:

$$
\hat{y}=x+\alpha\odot(z-x), \qquad 0\leq\alpha\leq1.
$$

The auxiliary target used in Stage 2 is

$$
\alpha^*=
\mathrm{clip}\left(
\frac{(s-x)(z-x)}
{(z-x)^2+\epsilon},
0,1
\right).
$$

The auxiliary term is used with a small weight (0.05) to regularize the gate. In practical terms, FC-ED proposes a correction and SRCG controls its amplitude at each time channel.

### Training settings

- input channels: 40
- FC-ED hidden dimension: 512
- SRCG hidden dimension: 64
- batch size: 32
- scale: `1e19`
- random seed: 2024
- Stage 1: 220 epochs, Adam, `lr=1e-3`, Relative Error loss
- Stage 1 checkpoint: minimum validation `Relative + 2*NRMSE`
- Stage 2: 100 epochs, FC-ED frozen, Adam, `lr=1e-5`
- Stage 2 objective: `Hybrid + 0.05*SmoothL1(alpha, alpha_star)`

The test set is not used for training or model selection.

## Measured-noise datasets

| ID | Acquisition condition | Total | Train | Validation | Test |
|---|---|---:|---:|---:|---:|
| E1 | Shielded room, 16-stack | 14,171 | 11,336 | 1,417 | 1,418 |
| E2 | Shielded room, single acquisition | 2,883 | 2,306 | 288 | 289 |
| E3 | Urban roadside, 16-stack | 62,539 | 50,031 | 6,253 | 6,255 |
| E4 | Urban roadside, single acquisition | 44,847 | 35,877 | 4,484 | 4,486 |

Supervised denoising files contain 80 columns: the first 40 columns are noisy responses and the last 40 columns are the corresponding clean responses.

The synthesis rule is

$$
x_i=s_i+\frac{n_i}{20}s_{22}, \qquad y=s.
$$

The source data used to reconstruct the experiments are stored under:

```text
data/
├─ clean_responses/
│  └─ clean_responses_ablation_320000x40.dat
└─ measured_noise/
   ├─ E1_shielded_16stack.dat
   ├─ E2_shielded_single.dat
   ├─ E3_urban_roadside_16stack.dat
   └─ E4_urban_roadside_single.dat
```

The `.dat` files are tracked with Git LFS.

## Comparison with other denoising methods

The cross-method comparison uses the two single-acquisition cases, E2 and E4.

Methods:

- Raw input
- Wavelet: SWT-sym6, level 3, MAD + universal soft threshold
- TEMDnet
- TEM1Dformer
- FC-ED (H512)
- FC-ED + SRCG

The TEM1Dformer implementation in this repository is a paper-guided reimplementation because official source code was not available.

All trainable methods are evaluated on the same held-out measured-noise test set within each scenario. Wavelet is applied directly to the same test responses.

### E2 — shielded room, single acquisition

| Method | RMSE | SNR (dB) | RMSE reduction vs. Raw | Harmful rate |
|---|---:|---:|---:|---:|
| Raw | 1.726e-11 | 37.989 | 0.00% | — |
| Wavelet | 1.847e-11 | 37.404 | -6.97% | 87.89% |
| TEMDnet | 1.808e-11 | 37.590 | -4.71% | 99.65% |
| TEM1Dformer (reimpl.) | 3.737e-11 | 31.281 | -116.48% | 53.29% |
| FC-ED H512 | 1.709e-11 | 38.076 | 0.99% | 86.85% |
| **FC-ED H512 + SRCG** | **1.081e-11** | **42.058** | **37.40%** | **24.91%** |

### E4 — urban roadside, single acquisition

| Method | RMSE | SNR (dB) | RMSE reduction vs. Raw | Harmful rate |
|---|---:|---:|---:|---:|
| Raw | 3.196e-11 | 33.888 | 0.00% | — |
| Wavelet | 3.557e-11 | 32.959 | -11.29% | 92.58% |
| TEMDnet | 2.901e-11 | 34.729 | 9.23% | 73.34% |
| TEM1Dformer (reimpl.) | 2.550e-11 | 35.849 | 20.22% | 55.82% |
| FC-ED H512 | 3.421e-11 | 33.298 | -7.02% | 24.74% |
| **FC-ED H512 + SRCG** | **1.831e-11** | **38.726** | **42.71%** | **4.86%** |

The harmful rate is the fraction of test samples for which processing increases RMSE relative to the raw input.

Full tables are available in:

```text
results/method_comparison/
```

## Wasserstein augmentation ablation

For each scenario, the WGAN and No-WGAN groups contain the same number of training samples and use the same clean-response sequence.

- **No-WGAN same-N**: measured training noise + bootstrap resampling from the same measured-noise pool
- **WGAN Full**: the same measured training noise + generated noise

Validation and test data contain held-out measured noise only.

| Scenario | No-WGAN RMSE | WGAN RMSE | WGAN RMSE change vs. No-WGAN | WGAN SNR change | Test samples |
|---|---:|---:|---:|---:|---:|
| E1 | 8.946e-13 | 9.940e-13 | -11.12% | -0.916 dB | 1,418 |
| E2 | 1.015e-11 | 1.081e-11 | -6.45% | -0.543 dB | 289 |
| E3 | 7.833e-12 | 6.803e-12 | +13.15% | +1.225 dB | 6,255 |
| E4 | 3.782e-11 | 1.831e-11 | +51.58% | +6.299 dB | 4,486 |

A positive RMSE change indicates lower RMSE with WGAN augmentation. Direct measured-noise resampling performs better in E1 and E2, whereas WGAN augmentation performs better in E3 and E4.

The amount of measured training noise also differs substantially among the four acquisition conditions. The observed differences therefore should not be attributed to environmental complexity alone.

Full ablation outputs are available in:

```text
results/wasserstein_ablation/
results/paperfinal_raw/
```

## Synthetic combined-noise experiments

The repository retains the earlier synthetic combined-noise scripts used during method development. The main analysis here focuses on measured-noise cases.

## Repository structure

```text
configs/
data/
experiments/
  paperfinal_reference/
  synthetic_combined_noise/
  wasserstein_ablation/
figures/
results/
  method_comparison/
  paperfinal_raw/
  wasserstein_ablation/
src/
  augmentation/
  data/
  evaluation/
  losses/
  models/
  training/
```

## Running the augmentation ablation

For E1:

```bash
python experiments/wasserstein_ablation/01_split_measured_noise.py --scenario E1
python experiments/wasserstein_ablation/02_train_wasserstein_generator.py --scenario E1
python experiments/wasserstein_ablation/03_check_generated_noise.py --scenario E1
python experiments/wasserstein_ablation/04_build_ablation_dataset.py --scenario E1
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group A
python experiments/wasserstein_ablation/05_train_denoiser.py --scenario E1 --group B
python experiments/wasserstein_ablation/06_evaluate_common_test.py --scenario E1
```

Use `--scenario E2`, `E3`, or `E4` for the other cases.

## Note on the Wasserstein generator

The archived generator uses a normally distributed interpolation coefficient inside the gradient-norm penalty. Standard WGAN-GP uses a uniform interpolation coefficient. For this reason, the repository refers to this part as **Wasserstein generative augmentation with a gradient-norm penalty** rather than canonical WGAN-GP.

## Data and checkpoints

The source clean-response and measured-noise data required to reconstruct the experiments are included through Git LFS.

Generated training/validation/test files, model checkpoints, prediction files, and large intermediate arrays are not included because they can be regenerated from the source data and scripts.

See `data/README.md` and `DATA_UPLOAD_GUIDE.md`.

## Repository

https://github.com/chenyangliu1994/ATEM-Denoising

## License

MIT License.
