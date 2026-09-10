import torch
import torch.nn as nn


class RelativeLoss(nn.Module):
    def __init__(self, eps_relative_scaled=1.0):
        super().__init__()
        self.eps = eps_relative_scaled

    def forward(self, y_pred, y_true):
        denom = torch.where(
            torch.abs(y_true) < self.eps,
            torch.full_like(y_true, self.eps),
            torch.abs(y_true),
        )
        return torch.mean(torch.abs(y_pred - y_true) / denom)


class HybridTEMLoss(nn.Module):
    def __init__(
        self,
        lambda_nrmse=2.0,
        eps_relative_scaled=1.0,
        eps_nrmse=1e-12,
    ):
        super().__init__()
        self.lambda_nrmse = lambda_nrmse
        self.eps_relative_scaled = eps_relative_scaled
        self.eps_nrmse = eps_nrmse

    def components(self, y_pred, y_true):
        err = y_pred - y_true

        denom = torch.where(
            torch.abs(y_true) < self.eps_relative_scaled,
            torch.full_like(y_true, self.eps_relative_scaled),
            torch.abs(y_true),
        )
        relative = torch.mean(torch.abs(err) / denom)

        nrmse = torch.sqrt(
            torch.mean(err ** 2)
            / (torch.mean(y_true ** 2) + self.eps_nrmse)
        )
        return relative, nrmse

    def forward(self, y_pred, y_true):
        relative, nrmse = self.components(y_pred, y_true)
        total = relative + self.lambda_nrmse * nrmse
        return total, relative, nrmse
