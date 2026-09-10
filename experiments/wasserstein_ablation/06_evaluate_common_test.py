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
from src.models.denoiser import ATEMDenoiser
from src.evaluation.metrics import compute_metrics


def load_checkpoint_model(path, device, common):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dcfg = common["denoiser"]

    model = ATEMDenoiser(
        input_dim=int(ckpt.get("input_dim", dcfg["input_dim"])),
        hidden_dim=int(ckpt.get("hidden_dim", dcfg["hidden_dim"])),
        gate_hidden_dim=int(ckpt.get("gate_hidden_dim", dcfg["gate_hidden_dim"])),
        max_delta=float(ckpt.get("max_delta", dcfg["max_gate_delta"])),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, batch_size, device):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    out = []
    for (xb,) in loader:
        out.append(model(xb.to(device)).cpu().numpy().astype(np.float64))
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    common = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)
    dcfg = common["denoiser"]

    scale = float(dcfg["scale"])
    batch_size = 256

    data_dir = ROOT / "outputs" / "04_ablation_datasets" / scfg["slug"]
    noisy, clean = load_supervised_80col(data_dir / "test_real_only.dat")

    A_model_path = (
        ROOT / "outputs" / "05_denoiser_ablation" / scfg["slug"]
        / "A_measured_resampling" / "final_RH_model.pth"
    )
    B_model_path = (
        ROOT / "outputs" / "05_denoiser_ablation" / scfg["slug"]
        / "B_wasserstein_aug" / "final_RH_model.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_A = load_checkpoint_model(A_model_path, device, common)
    model_B = load_checkpoint_model(B_model_path, device, common)

    pred_A = predict(model_A, noisy * scale, batch_size, device) / scale
    pred_B = predict(model_B, noisy * scale, batch_size, device) / scale

    eps_phys = float(dcfg["eps_relative_scaled"]) / scale
    mA = compute_metrics(pred_A, clean, eps_relative_physical=eps_phys)
    mB = compute_metrics(pred_B, clean, eps_relative_physical=eps_phys)

    out_dir = ROOT / "outputs" / "06_test_ablation" / scfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        {"Scenario": args.scenario, "Group": "A_measured_resampling", **mA},
        {"Scenario": args.scenario, "Group": "B_wasserstein_aug", **mB},
    ])
    metrics_df.to_csv(out_dir / "test_metrics_A_vs_B.csv", index=False)

    np.savetxt(
        out_dir / "test_predictions_A.dat",
        np.hstack([noisy, pred_A, clean]),
        fmt="%.12e",
    )
    np.savetxt(
        out_dir / "test_predictions_B.dat",
        np.hstack([noisy, pred_B, clean]),
        fmt="%.12e",
    )

    improvement = {
        "RMSE_improvement_pct": (mA["RMSE"] - mB["RMSE"]) / mA["RMSE"] * 100,
        "MAE_improvement_pct": (mA["MAE"] - mB["MAE"]) / mA["MAE"] * 100,
        "MeanRelative_improvement_pct": (mA["MeanRelative"] - mB["MeanRelative"]) / mA["MeanRelative"] * 100,
        "Early2_RMSE_improvement_pct": (mA["Early2_RMSE"] - mB["Early2_RMSE"]) / mA["Early2_RMSE"] * 100,
        "Middle2_29_RMSE_improvement_pct": (mA["Middle2_29_RMSE"] - mB["Middle2_29_RMSE"]) / mA["Middle2_29_RMSE"] * 100,
        "Late10_RMSE_improvement_pct": (mA["Late10_RMSE"] - mB["Late10_RMSE"]) / mA["Late10_RMSE"] * 100,
        "SNR_gain_dB": mB["SNR_dB"] - mA["SNR_dB"],
    }
    pd.DataFrame([{"Scenario": args.scenario, **improvement}]).to_csv(
        out_dir / "comparison_summary.csv", index=False
    )

    print("\nA:")
    print(pd.Series(mA).to_string())
    print("\nB:")
    print(pd.Series(mB).to_string())
    print("\nB vs A:")
    print(pd.Series(improvement).to_string())


if __name__ == "__main__":
    main()
