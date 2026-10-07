# ATEM-Denoising

Code and experiment records for 40-channel airborne transient electromagnetic
(ATEM) response denoising. The current model uses a fully connected
encoder-decoder (FC-ED) followed by a selective residual correction gate
(SRCG).

## Method

FC-ED first produces a candidate response \(z\). SRCG then estimates a
channel-wise factor \(\alpha\) that controls how much of the FC-ED correction
is used:

\[
\hat y=x+\alpha\odot(z-x), \qquad 0\leq\alpha\leq1.
\]

The auxiliary target used during Stage 2 is

\[
\alpha^*=\mathrm{clip}\left(
\frac{(s-x)(z-x)}
{(z-x)^2+\epsilon},
0,1
\right).
\]

It is used with a small weight (0.05) to regularize the gate. In practical
terms, FC-ED proposes the correction and SRCG controls its amplitude at each
time channel.

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

Supervised denoising files contain 80 columns: the first 40 columns are noisy
responses and the last 40 columns are the corresponding clean responses.

The synthesis rule is

\[
x_i=s_i+\frac{n_i}{20}s_{22}, \qquad y=s.
\]

## Wasserstein augmentation ablation

For each scenario, the WGAN and No-WGAN groups have the same number of
training samples and use the same clean-response sequence.

- **No-WGAN same-N**: measured training noise + bootstrap resampling from the
  same measured-noise pool.
- **WGAN Full**: the same measured training noise + generated noise.

Validation and test data contain held-out measured noise only.

The result varies with acquisition condition. Direct measured-noise resampling
is better in E1 and E2, while Wasserstein augmentation is better in E3 and E4.
Because the amount of measured training noise also differs substantially among
the four cases, this difference cannot be assigned to environmental complexity
alone.

Summary files:

```text
results/wasserstein_ablation/current_overall_ablation_summary.csv
results/paperfinal_raw/
```

## Comparison with other denoising methods

The cross-method comparison uses the two single-acquisition cases, E2 and E4.
The following methods are included:

- Raw input
- Wavelet: SWT-sym6, level 3, MAD + universal soft threshold
- TEMDnet
- TEM1Dformer
- FC-ED (H512)
- FC-ED + SRCG

The TEM1Dformer code in this repository is a paper-guided reimplementation;
official source code was not available.

For the trainable methods, E2 and E4 use the same WGAN-augmented training set,
validation set, and held-out measured-noise test set. Wavelet is applied
directly to the same test responses.

Results are provided in:

```text
results/method_comparison/E2/
results/method_comparison/E4/
results/method_comparison/method_comparison_summary.csv
results/method_comparison/method_comparison_harmful_rates.csv
```

In addition to RMSE/NRMSE/SNR, the repository reports the sample-level harmful
modification rate, i.e. the fraction of test samples for which processing
increases RMSE relative to the raw input.

## Synthetic combined-noise experiments

The repository also retains the earlier synthetic combined-noise scripts.
Those experiments were useful during method development, but the main analysis
here uses measured-noise cases.

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

The archived generator uses a normally distributed interpolation coefficient
inside the gradient-norm penalty. Standard WGAN-GP uses a uniform interpolation
coefficient. For this reason, the repository refers to this part as
**Wasserstein generative augmentation with a gradient-norm penalty** rather
than canonical WGAN-GP.

## Data and checkpoints

Large training/validation/test `.dat` files, model checkpoints, and intermediate
arrays are not included in the GitHub repository. The compact CSV/JSON results
and plotting data are kept with the code.

See `data/README.md` and `UPLOAD_CHECKLIST.md`.

## Repository

https://github.com/chenyangliu1994/ATEM-Denoising

## License

MIT License.
