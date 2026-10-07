import torch
import torch.nn as nn


class ThreeLinearBlock(nn.Module):
    """Three fully connected layers with ReLU after each layer."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x


class FCED(nn.Module):
    """
    Paper-final fully connected encoder-decoder.

    Encoder: 40 -> H -> H -> H
    Decoder: H -> H -> H -> 40
    Paper-final H = 512.
    """
    def __init__(self, input_dim: int = 40, hidden_dim: int = 512):
        super().__init__()
        self.encoder = ThreeLinearBlock(input_dim, hidden_dim, hidden_dim)
        self.decoder = ThreeLinearBlock(hidden_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SelectiveResidualCorrectionGate(nn.Module):
    """
    SRCG learns how much of the FC-ED candidate correction should be accepted.

        z = FCED(x)
        d = z - x
        alpha = SRCG(x, z, d)
        y_hat = x + alpha * d

    The gate is channel-wise and bounded to [0, 1] by sigmoid.
    """
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        # sigmoid(2.1972246) ~= 0.90
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.1972246)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-12)
        x_n = x / (rms + 1e-12)
        z_n = z / (rms + 1e-12)
        d_n = (z - x) / (rms + 1e-12)
        feat = torch.cat([x_n, z_n, d_n], dim=1)
        h = torch.relu(self.fc1(feat))
        return torch.sigmoid(self.fc2(h))


class FCED_SRCG(nn.Module):
    def __init__(self, backbone: FCED, input_dim: int = 40, srcg_hidden: int = 64):
        super().__init__()
        self.backbone = backbone
        self.srcg = SelectiveResidualCorrectionGate(input_dim, srcg_hidden)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        z = self.backbone(x)
        alpha = self.srcg(x, z)
        pred = x + alpha * (z - x)
        if return_aux:
            return pred, z, alpha
        return pred
