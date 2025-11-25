"""
LSTM + Multi-Head Attention model for single-step forecasting.
Input: x (batch, seq_len, n_features)
Output: y_hat (batch,) or (batch, 1)

GPU-ready: supply device string ("cuda" or "cpu") to constructor or call .to(device).
Includes save/load helpers that persist model state_dict + config.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Dict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # last odd column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x: torch.Tensor):
        # x: (seq_len, batch, d_model)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x


class LSTMAttentionModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        attn_embed_dim: Optional[int] = None,
        dropout: float = 0.1,
        bidirectional: bool = False,
        fc_hidden: int = 64,
        device: Optional[str] = None,
    ):
        """
        Params:
            input_size: number of features per timestep
            hidden_size: LSTM hidden size
            n_layers: number of LSTM layers
            n_heads: heads for MultiHeadAttention
            attn_embed_dim: embedding dim for attention (defaults to hidden_size)
            dropout: dropout between layers
            bidirectional: if True use bidirectional LSTM
            fc_hidden: hidden units for final MLP
            device: "cuda" or "cpu" (optional)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=2048)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        attn_dim = attn_embed_dim or hidden_size
        # If bidirectional, project to attn_dim
        self.attn_in_proj = nn.Linear(hidden_size * self.num_directions, attn_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=n_heads, dropout=dropout, batch_first=False)

        self.fc = nn.Sequential(
            nn.Linear(attn_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_size)
        returns: (batch,) or (batch, 1)
        """
        assert x.dim() == 3
        batch, seq_len, _ = x.shape
        x = self.input_projection(x)  # (batch, seq_len, hidden)
        x = x.transpose(0, 1)  # (seq_len, batch, hidden) for pos encoder & MHA
        x = self.pos_encoder(x)  # add positional encoding

        x_lstm_in = x.transpose(0, 1)  # (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x_lstm_in)  # (batch, seq_len, hidden * num_directions)

        # prepare for attention: (seq_len, batch, embed)
        # If bidirectional, lstm_out last dim = hidden*2, else hidden
        attn_in = self.attn_in_proj(lstm_out)  # (batch, seq_len, attn_dim)
        attn_in = attn_in.transpose(0, 1)  # (seq_len, batch, attn_dim)

        # self-attention across time
        attn_out, attn_weights = self.multihead_attn(attn_in, attn_in, attn_in, need_weights=True)
        # attn_out: (seq_len, batch, attn_dim). We'll pool across time (e.g., take last time or mean)
        # Use attention-weighted pooling: take the attention output at the last time step
        last = attn_out[-1]  # (batch, attn_dim)

        out = self.fc(last).squeeze(-1)  # (batch,)
        return out

    def save(self, path: str, extra: Optional[Dict] = None):
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
