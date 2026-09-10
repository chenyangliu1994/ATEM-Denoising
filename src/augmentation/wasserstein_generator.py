import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=128, output_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def archived_gradient_norm_penalty(critic, real, fake):
    """
    Gradient-norm penalty used by the archived implementation.

    IMPORTANT:
        alpha ~ N(0,1), implemented with torch.randn.

    This intentionally preserves the archived research implementation and is
    not claimed to be identical to canonical WGAN-GP interpolation.
    """
    alpha = torch.randn(
        (real.size(0), 1),
        device=real.device,
    ).expand_as(real)

    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)

    score = critic(interpolated)
    grad_outputs = torch.ones_like(score)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1.0) ** 2).mean()
