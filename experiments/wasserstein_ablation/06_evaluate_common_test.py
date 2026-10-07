from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_scenario_config, load_supervised_80col
from src.models.denoiser import FCED, FCED_SRCG


def rmse(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def overall_metrics(pred, clean):
    pred = np.asarray(pred, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    err = pred - clean
    signal = float(np.sum(clean ** 2))
    noise = float(np.sum(err ** 2))
    return {
        "RMSE": rmse(err),
        "MAE": float(np.mean(np.abs(err))),
        "NRMSE": float(np.sqrt(noise / max(signal, np.finfo(float).tiny))),
        "SNR_dB": float(np.inf if noise == 0 else 10.0*np.log10(signal/noise)),
    }


def load_final_model(stage1_path, stage2_path, device, cfg):
    dcfg = cfg["denoiser"]
    ck1 = torch.load(stage1_path, map_location=device, weights_only=False)
    ck2 = torch.load(stage2_path, map_location=device, weights_only=False)

    input_dim = int(ck2.get("input_dim", dcfg["input_dim"]))
    hidden_dim = int(ck2.get("hidden_dim", dcfg["hidden_dim"]))
    srcg_hidden = int(ck2.get("srcg_hidden", dcfg["srcg_hidden_dim"]))

    stage1 = FCED(input_dim, hidden_dim).to(device)
    stage1.load_state_dict(ck1["model_state_dict"], strict=True)
    stage1.eval()

    final = FCED_SRCG(FCED(input_dim, hidden_dim), input_dim, srcg_hidden).to(device)
    final.load_state_dict(ck2["model_state_dict"], strict=True)
    final.eval()
    return stage1, final


@torch.no_grad()
def predict(model, X, scale, device, batch=256):
    ds = TensorDataset(torch.from_numpy((X*scale).astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    out = []
    for (xb,) in loader:
        out.append((model(xb.to(device)).cpu().numpy()/scale).astype(np.float64))
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    cfg = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)
    scale = float(cfg["denoiser"]["scale"])

    data_dir = ROOT / "outputs" / "04_ablation_datasets" / scfg["slug"]
    noisy, clean = load_supervised_80col(data_dir / "test_real_only.dat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    preds = {}
    sample_rmse = {}

    for group, label in [("A_measured_resampling", "NoWGAN"), ("B_wasserstein_aug", "WGAN")]:
        mdir = ROOT / "outputs" / "05_denoiser_ablation" / scfg["slug"] / group
        s1, final = load_final_model(
            mdir / "FCED_stage1_H512_E220_best.pth",
            mdir / "FCED_H512_SRCG_continuous_best.pth",
            device, cfg,
        )
        pred = predict(final, noisy, scale, device)
        preds[label] = pred
        sample_rmse[label] = np.sqrt(np.mean((pred-clean)**2, axis=1))
        rows.append({"Scenario": args.scenario, "Variant": label, **overall_metrics(pred, clean)})

    out_dir = ROOT / "outputs" / "06_test_ablation" / scfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_metrics_A_vs_B.csv", index=False)

    np.savetxt(out_dir / "test_predictions_A.dat",
               np.hstack([noisy, preds["NoWGAN"], clean]), fmt="%.12e")
    np.savetxt(out_dir / "test_predictions_B.dat",
               np.hstack([noisy, preds["WGAN"], clean]), fmt="%.12e")

    A = df[df["Variant"]=="NoWGAN"].iloc[0]
    B = df[df["Variant"]=="WGAN"].iloc[0]
    better = float(np.mean(sample_rmse["WGAN"] < sample_rmse["NoWGAN"]) * 100.0)
    summary = {
        "Scenario": args.scenario,
        "NoWGAN_RMSE": A["RMSE"],
        "WGAN_RMSE": B["RMSE"],
        "WGAN_RMSE_reduction_vs_NoWGAN_pct": (A["RMSE"]-B["RMSE"])/A["RMSE"]*100.0,
        "NoWGAN_NRMSE": A["NRMSE"],
        "WGAN_NRMSE": B["NRMSE"],
        "NoWGAN_SNR_dB": A["SNR_dB"],
        "WGAN_SNR_dB": B["SNR_dB"],
        "WGAN_SNR_gain_vs_NoWGAN_dB": B["SNR_dB"]-A["SNR_dB"],
        "WGAN_lower_sample_RMSE_pct": better,
        "N_test": len(noisy),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "comparison_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
