# Synthetic combined-noise data

This folder preserves the original combined-noise dataset generation script used in the historical experiment. It creates a 40-channel clean forward response and adds atmospheric-pulse, Gaussian, and harmonic components to obtain a noisy response.

The historical setting is referred to as `d=10` in the manuscript. This is the original script, not a newly calibrated or re-run dataset generator. Run it only after installing its dependencies, including a compatible SimPEG version:

```bash
python experiments/synthetic_combined_noise/generate_combined_noise.py
```

Run from the repository root. The original script writes `zaosheng.txt` there: columns 1–40 are noisy responses and columns 41–80 are the corresponding clean responses. Its default requested sample count is 500,000, subject to the original multiprocessing allocation. The generated dataset is **not included** in this patch; do not commit the generated `zaosheng.txt` or `tmp_data_out_*.txt` files to Git.

For the three-condition method comparison, use a separately fixed train/validation/test split and identical evaluation samples across methods. This historical generation script does not define that split. For model generation and clean forward responses as a standalone workflow, see [ATEM-Forward-Model-Library](https://github.com/chenyangliu1994/ATEM-Forward-Model-Library).
