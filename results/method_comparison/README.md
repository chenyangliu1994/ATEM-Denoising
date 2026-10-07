# E2/E4 denoising-method comparison

The comparison covers:

- Raw
- Wavelet
- TEMDnet
- TEM1Dformer
- FC-ED H512
- FC-ED H512 + SRCG

All trainable methods are evaluated on the same held-out measured-noise test
set within each scenario. The TEM1Dformer implementation is a paper-guided
reimplementation because official source code was not available.

`method_comparison_summary.csv` contains the overall RMSE, MAE, NRMSE and SNR
results. `method_comparison_harmful_rates.csv` contains the sample-level harmful
modification rates.
