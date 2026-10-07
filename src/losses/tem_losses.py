import torch


def relative_loss(pred: torch.Tensor, target: torch.Tensor, eps_relative_scaled: float = 1.0):
    denom = torch.where(
        torch.abs(target) < eps_relative_scaled,
        torch.full_like(target, eps_relative_scaled),
        torch.abs(target),
    )
    return torch.mean(torch.abs(pred - target) / denom)


def hybrid_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_nrmse: float = 2.0,
    eps_relative_scaled: float = 1.0,
    eps_nrmse: float = 1e-12,
):
    err = pred - target
    rel = relative_loss(pred, target, eps_relative_scaled)
    nrmse = torch.sqrt(
        torch.mean(err ** 2) / (torch.mean(target ** 2) + eps_nrmse)
    )
    hybrid = rel + lambda_nrmse * nrmse
    return hybrid, rel, nrmse


@torch.no_grad()
def continuous_alpha_star(
    x: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
):
    d = z - x
    alpha_star = ((target - x) * d) / (d * d + eps)
    return torch.clamp(alpha_star, 0.0, 1.0)
