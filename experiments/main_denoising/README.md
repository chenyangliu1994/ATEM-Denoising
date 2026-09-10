# Main denoising

The same fixed model and R-H trainer used in the controlled ablation are
available as generic entry points.

Train any prepared `noisy40 | clean40` dataset:

```bash
python experiments/main_denoising/train_rh.py   --train path/to/train.dat   --val path/to/val.dat   --out outputs/main_denoising
```

Evaluate a saved checkpoint:

```bash
python experiments/main_denoising/evaluate.py   --test path/to/test.dat   --checkpoint outputs/main_denoising/final_RH_model.pth
```

The manuscript's matched measured-noise augmentation comparison is under:

```text
experiments/wasserstein_ablation/
```
