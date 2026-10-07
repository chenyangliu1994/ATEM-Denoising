# Wasserstein augmentation ablation

This directory contains the common workflow used for E1-E4.

For each acquisition condition:

1. Split measured noise chronologically into 80% train, 10% validation, and
   10% test.
2. Train the Wasserstein noise generator on the training split only.
3. Build two equal-size denoising training sets:
   - A / No-WGAN: measured noise + bootstrap-resampled measured noise.
   - B / WGAN: the same measured noise + generated noise.
4. Use the same clean-response rows in A and B.
5. Train the same H512 FC-ED + SRCG model for both groups.
6. Train Stage 1 for 220 epochs and Stage 2 for 100 epochs.
7. Evaluate both models on the same held-out measured-noise test set.

The Stage 2 alpha-target term has weight 0.05.

## Results

E1/E2 favor direct measured-noise resampling, whereas E3/E4 favor Wasserstein
augmentation. The amount of measured training noise is also different across
the four conditions, so these results should not be explained by environment
type alone.

## Example

```bash
python 01_split_measured_noise.py --scenario E1
python 02_train_wasserstein_generator.py --scenario E1
python 03_check_generated_noise.py --scenario E1
python 04_build_ablation_dataset.py --scenario E1
python 05_train_denoiser.py --scenario E1 --group A
python 05_train_denoiser.py --scenario E1 --group B
python 06_evaluate_common_test.py --scenario E1
```
