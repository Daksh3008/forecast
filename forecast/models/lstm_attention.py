"""
Upgraded LSTM + Multi-Head Attention model.

Features:
- Input projection
- Positional encoding
- Multi-layer LSTM (configurable)
- Self-attention over time (MultiheadAttention)
- LayerNorm and residual connections
- MLP head with configurable fc_hidden
- save()/load() helpers that persist model kwargs in checkpoint for reproducible loading
- optional return-based training support (prediction can be configured to predict absolute price
  or log-return / return).
"""

from __future__ import annotations
import math
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        L = x.size(1)
        return x + self.pe[:, :L]


class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
        fc_hidden: int = 128,
        input_proj: Optional[int] = None,
        seq_len: int = 60,
        predict_return: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden=fc_hidden,
            input_proj=input_proj,
            seq_len=seq_len,
            predict_return=predict_return,
        )

        self.device = device
        self.input_proj_dim = input_proj or hidden_size
        # input projection
        self.input_proj = nn.Linear(input_size, self.input_proj_dim)

        # positional encoding
        self.pos_enc = PositionalEncoding(self.input_proj_dim, max_len=seq_len)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_proj_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        attn_dim = hidden_size * (2 if bidirectional else 1)
        # project LSTM outputs to attn dim if needed
        if attn_dim != self.input_proj_dim:
            self.proj_for_attn = nn.Linear(attn_dim, self.input_proj_dim)
        else:
            self.proj_for_attn = nn.Identity()

        # Multihead attention expects (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) with batch_first
        self.self_attn = nn.MultiheadAttention(embed_dim=self.input_proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        # LayerNorm + residual
        self.ln_attn = nn.LayerNorm(self.input_proj_dim)
        self.ln_post = nn.LayerNorm(self.input_proj_dim)

        # MLP head
        self.fc = nn.Sequential(
            nn.Linear(self.input_proj_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.predict_return = predict_return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch,) scalar output
        """
        # input projection
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        # LSTM
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden*directions)
        attn_in = self.proj_for_attn(lstm_out)  # (batch, seq_len, d_model)

        # Self-attention (batch_first=True)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)  # (batch, seq_len, d_model)
        # residual + norm
        h = self.ln_attn(attn_in + self.dropout(attn_out))
        # pooling: use last timestep + mean pooling combined
        last = h[:, -1, :]  # (batch, d_model)
        mean = h.mean(dim=1)
        pooled = self.ln_post(last + mean)
        out = self.fc(pooled).squeeze(-1)
        return out

    # -------------------------
    # Save / Load helpers
    # -------------------------
    def save(self, path: str, extra: Optional[Dict[str, Any]] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "model_kwargs": self.kwargs,
            "extra": extra or {}
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None, **override_kwargs):
        payload = torch.load(path, map_location=map_location)
        saved_kwargs = payload.get("model_kwargs", {})
        saved_kwargs.update(override_kwargs or {})
        model = cls(**saved_kwargs)
        model.load_state_dict(payload["model_state_dict"])
        return model
