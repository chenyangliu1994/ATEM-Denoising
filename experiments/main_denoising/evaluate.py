from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_supervised_80col
from src.models.denoiser import ATEMDenoiser
from src.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/main_denoising/evaluation")
    args = parser.parse_args()

    cfg = load_common_config(ROOT)
    dcfg = cfg["denoiser"]
    scale = float(dcfg["scale"])

    test_path = Path(args.test)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out)

    if not test_path.is_absolute():
        test_path = ROOT / test_path
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy, clean = load_supervised_80col(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ATEMDenoiser(
        input_dim=int(ckpt.get("input_dim", dcfg["input_dim"])),
        hidden_dim=int(ckpt.get("hidden_dim", dcfg["hidden_dim"])),
        gate_hidden_dim=int(ckpt.get("gate_hidden_dim", dcfg["gate_hidden_dim"])),
        max_delta=float(ckpt.get("max_delta", dcfg["max_gate_delta"])),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X = torch.tensor(noisy * scale, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=256, shuffle=False)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy().astype(np.float64))
    pred = np.vstack(preds) / scale

    eps_phys = float(dcfg["eps_relative_scaled"]) / scale
    metrics = compute_metrics(pred, clean, eps_relative_physical=eps_phys)

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    np.savetxt(
        out_dir / "predictions.dat",
        np.hstack([noisy, pred, clean]),
        fmt="%.12e",
    )

    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
