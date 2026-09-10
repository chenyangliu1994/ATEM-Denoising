# Wasserstein augmentation ablation

This directory reproduces the matched A/B experiment used in the manuscript.

## Protocol

For each scenario:

1. Keep the original measured-noise acquisition order.
2. Split continuously into 80% train, 10% validation, 10% test.
3. Train the Wasserstein generator using **training measured noise only**.
4. Build equal-size A/B training sets:
   - A = measured + bootstrap-resampled measured
   - B = the same measured + generated
5. Use identical clean-response rows for A and B.
6. Use the same synthesis rule in all scenarios:
   `x_i = s_i + (n_i / 20) * s_22`
7. Train A and B with the same two-stage R-H protocol.
8. Evaluate both once on the same held-out measured-noise test set.

## Run

```bash
python 00_validate_public_data.py
python 01_split_measured_noise.py --scenario E1
python 02_train_wasserstein_generator.py --scenario E1
python 03_check_generated_noise.py --scenario E1
python 04_build_ablation_dataset.py --scenario E1
python 05_train_denoiser.py --scenario E1 --group A
python 05_train_denoiser.py --scenario E1 --group B
python 06_evaluate_common_test.py --scenario E1
```
