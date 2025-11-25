"""
Temporal Convolutional Network (TCN) for single-step forecasting.
Paper reference: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
This implementation uses causal dilated convolutions with residual blocks.

Input: x (batch, seq_len, n_features)
Output: y_hat (batch,)

GPU-ready: move model to device via model.to(device)
"""
from typing import List, Optional
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        num_channels: list of out channels for each temporal block, e.g. [64, 64, 128]
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        layers = []
        num_levels = len(num_channels)
        in_channels = input_size
        for i in range(num_levels):
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        # Final projection: global pooling over time then linear to output
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(num_channels[-1], 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_size)
        TCN expects (batch, channels, seq_len)
        returns (batch,)
        """
        assert x.dim() == 3
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.tcn(x)  # (batch, channels, seq_len)
        out = self.fc(y).squeeze(-1)  # (batch,)
        return out

    def save(self, path: str, extra: dict = None):
        state = {"model_state_dict": self.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None, **model_kwargs):
        map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
