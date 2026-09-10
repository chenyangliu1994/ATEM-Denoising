from pathlib import Path
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.io import load_common_config, load_scenario_config, load_measured_noise_80col, save_dat
from src.augmentation.wasserstein_generator import Generator, Critic, archived_gradient_norm_penalty


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=["E1","E2","E3","E4"])
    args = parser.parse_args()

    common = load_common_config(ROOT)
    scfg = load_scenario_config(args.scenario, ROOT)
    wcfg = common["wasserstein"]
    seed = int(common["seed"])
    set_seed(seed)

    split_dir = ROOT / "outputs" / "01_split_real_noise" / scfg["slug"]
    input_file = split_dir / "train_real_noise.dat"

    out_dir = ROOT / "outputs" / "02_wasserstein_generation" / scfg["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    time, noise = load_measured_noise_80col(input_file, dtype=np.float32)
    if not np.allclose(time, time[0], rtol=1e-6, atol=1e-12):
        raise ValueError("Time channels are not identical across measured-noise rows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_data = torch.tensor(noise, dtype=torch.float32, device=device)

    G = Generator(
        latent_dim=int(wcfg["latent_dim"]),
        hidden_dim=int(wcfg["hidden_dim"]),
        output_dim=int(wcfg["output_dim"]),
    ).to(device)

    D = Critic(
        input_dim=int(wcfg["output_dim"]),
        hidden_dim=int(wcfg["hidden_dim"]),
    ).to(device)

    lr = float(wcfg["learning_rate"])
    betas = (float(wcfg["beta1"]), float(wcfg["beta2"]))
    opt_g = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_d = optim.Adam(D.parameters(), lr=lr, betas=betas)

    batch_size = int(wcfg["batch_size"])
    epochs = int(wcfg["epochs"])
    latent_dim = int(wcfg["latent_dim"])
    lambda_gp = float(wcfg["lambda_gp"])

    hist_g, hist_d = [], []
    n_real = len(real_data)

    for epoch in range(1, epochs + 1):
        epoch_g, epoch_d = [], []

        # Archived implementation: original order, no shuffle, D:G = 1:1.
        for start in range(0, n_real, batch_size):
            real = real_data[start:min(start + batch_size, n_real)]
            batch_n = len(real)

            # Critic update.
            z = torch.randn(batch_n, latent_dim, device=device)
            fake = G(z)

            opt_d.zero_grad()
            score_real = D(real)
            score_fake = D(fake.detach())
            gp = archived_gradient_norm_penalty(D, real, fake.detach())

            loss_d = -score_real.mean() + score_fake.mean() + lambda_gp * gp
            loss_d.backward()
            opt_d.step()

            dval = float(loss_d.item())
            hist_d.append(dval)
            epoch_d.append(dval)

            # Generator update reuses the same z, matching the archived code.
            opt_g.zero_grad()
            fake_g = G(z)
            loss_g = -D(fake_g).mean()
            loss_g.backward()
            opt_g.step()

            gval = float(loss_g.item())
            hist_g.append(gval)
            epoch_g.append(gval)

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"D={np.mean(epoch_d):.6f} G={np.mean(epoch_g):.6f}"
        )

    torch.save(G.state_dict(), out_dir / "generator.pth")
    torch.save(D.state_dict(), out_dir / "critic.pth")
    pd.DataFrame({"loss": hist_g}).to_csv(out_dir / "loss_generator.csv", index=False)
    pd.DataFrame({"loss": hist_d}).to_csv(out_dir / "loss_critic.csv", index=False)

    # Generate exactly N new noise vectors.
    # Keep the 4096-row generation batching used in the formal experiment.
    G.eval()
    generated_parts = []
    with torch.no_grad():
        gen_batch_size = 4096
        for start in range(0, n_real, gen_batch_size):
            current_n = min(gen_batch_size, n_real - start)
            z = torch.randn(current_n, latent_dim, device=device)
            generated_parts.append(
                G(z).cpu().numpy().astype(np.float64)
            )
    generated = np.vstack(generated_parts)

    # Preserve the same 80-column convention: time40 | noise40.
    generated_80 = np.hstack([
        np.tile(time[0:1].astype(np.float64), (n_real, 1)),
        generated,
    ])
    save_dat(out_dir / "generated_noise.dat", generated_80)

    pd.DataFrame([{
        "scenario": args.scenario,
        "seed": seed,
        "measured_train_rows": n_real,
        "generated_rows": n_real,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lambda_gp": lambda_gp,
        "alpha_distribution": "Normal(0,1)",
        "D_to_G_ratio": "1:1",
        "generator_latent_reused_for_G_update": True,
    }]).to_csv(out_dir / "generation_summary.csv", index=False)

    print("Generated:", out_dir / "generated_noise.dat")


if __name__ == "__main__":
    main()
