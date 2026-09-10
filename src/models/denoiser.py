import torch
import torch.nn as nn


class ThreeLinearBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x


class BoundedChannelGate(nn.Module):
    """
    BCG:
        RMS-normalized 40 -> 64 -> 40 gate network
        g = 1 + max_delta * tanh(z)

    With max_delta=0.10, gate values are bounded approximately in [0.9, 1.1].
    """
    def __init__(self, input_dim=40, gate_hidden_dim=64, max_delta=0.10):
        super().__init__()
        self.max_delta = max_delta
        self.fc1 = nn.Linear(input_dim, gate_hidden_dim)
        self.fc2 = nn.Linear(gate_hidden_dim, input_dim)

        # Identity initialization: z=0 -> g=1.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-12)
        x_norm = x / (rms + 1e-12)
        h = torch.relu(self.fc1(x_norm))
        logits = self.fc2(h)
        gates = 1.0 + self.max_delta * torch.tanh(logits)
        return x * gates, gates


class ATEMDenoiser(nn.Module):
    """
    Fixed 40-channel fully connected encoder-decoder with BCG.

    Encoder: 40 -> 128 -> 128 -> 128
    Decoder: 128 -> 128 -> 128 -> 40
    Final activation: ReLU
    """
    def __init__(
        self,
        input_dim=40,
        hidden_dim=128,
        gate_hidden_dim=64,
        max_delta=0.10,
    ):
        super().__init__()
        self.gate = BoundedChannelGate(
            input_dim=input_dim,
            gate_hidden_dim=gate_hidden_dim,
            max_delta=max_delta,
        )
        self.encoder = ThreeLinearBlock(input_dim, hidden_dim, hidden_dim)
        self.decoder = ThreeLinearBlock(hidden_dim, hidden_dim, input_dim)

    def forward(self, x, return_gates=False):
        x_gated, gates = self.gate(x)
        y = self.decoder(self.encoder(x_gated))
        if return_gates:
            return y, gates
        return y
