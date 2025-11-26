import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Chomps off extra padding added for causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN residual block that preserves sequence length."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 1×1 projection for residual if channel mismatch
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Fully working TCN for time-series forecasting.
    Input:  (B, T, C)
    Output: (B,)  — predicts log-price next day
    """
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        in_ch = input_size

        for i in range(num_levels):
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (B, T, C) → convert to (B, C, T)
        x = x.transpose(1, 2)
        y = self.network(x)

        # global average pooling over time dimension
        y = y.mean(dim=2)

        out = self.fc(y).squeeze(-1)
        return out

    # -------- Save & load checkpoints --------
    def save(self, path, extra=None):
        import torch
        import os
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {"model": self.state_dict(), "extra": extra}
        torch.save(ckpt, path)

    @staticmethod
    def load(path, map_location="cpu", **kwargs):
        ckpt = torch.load(path, map_location=map_location)
        model = TCNModel(**kwargs)
        model.load_state_dict(ckpt["model"])
        return model
