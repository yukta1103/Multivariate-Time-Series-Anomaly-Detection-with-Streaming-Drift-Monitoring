"""Anomaly Transformer (Xu et al., ICLR 2022), reimplemented.

The idea the paper turns on: a normal timestep's attention is dominated by its
immediate neighbours, so its learned attention ("series association") is close
to a Gaussian centred on itself ("prior association"). An anomalous timestep
cannot confine its attention locally -- it has to attend to the whole window to
be reconstructed. The gap between the two, the *association discrepancy*, is a
sharper anomaly signal than reconstruction error alone.

A minimax schedule stops the model collapsing the discrepancy to zero: the
prior branch is pulled toward the series, while the series branch is pushed
away from the prior.

Memory notes -- the association tensors are ``(B, H, L, L)`` and dominate VRAM
(~20x everything parametric at L=100). Three deviations from the reference
implementation, all numerically identical, all purely to cut that term:

1. ``distances`` is kept as a ``(L, L)`` buffer and broadcast, rather than
   ``.repeat()``-ed to ``(B, H, L, L)``. Saves one full tensor per layer.
2. ``distances**2`` is precomputed once at init instead of per forward pass.
   Saves a second.
3. The prior is normalized once per layer in the loss (see ``losses.py``)
   rather than the reference's four times.

The association matrices are forced to fp32 even under autocast: sigma can get
as small as ~1.1e-5, and ``1 / (sqrt(2*pi) * sigma)`` then reaches ~3.6e4,
which is uncomfortably close to the fp16 ceiling of 65504.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal positions. No learnable parameters."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Circular-padded conv over the feature axis, lifting c_in -> d_model."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        self.tokenConv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1,
            padding_mode="circular", bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class AnomalyAttention(nn.Module):
    """Self-attention that also emits a learnable Gaussian prior association."""

    def __init__(self, win_size: int, attention_dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        idx = torch.arange(win_size, dtype=torch.float32)
        distances = (idx[:, None] - idx[None, :]).abs()
        # Only the square is ever used; precompute it so the forward pass never
        # materialises a (B, H, L, L) intermediate just to square distances.
        self.register_buffer("distances_sq", distances.pow(2), persistent=False)

    def forward(self, queries, keys, values, sigma):
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)

        # Big matmul: let autocast run this in fp16 where available.
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores

        # Softmax in fp32 -- autocast promotes it anyway, but be explicit since
        # `series` is a probability distribution the KL loss will take logs of.
        series = self.dropout(torch.softmax(attn.float(), dim=-1))

        # --- Gaussian prior association, fp32 throughout (see module docstring)
        sigma = sigma.transpose(1, 2).float()          # (B, L, H) -> (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3.0, sigma) - 1.0            # sigma in (~1.1e-5, 2)
        sigma = sigma.unsqueeze(-1)                    # (B, H, L, 1), broadcasts
        prior = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(
            -self.distances_sq / (2 * sigma * sigma)   # (1,1,L,L) / (B,H,L,1)
        )

        v = torch.einsum("bhls,bshd->blhd", series.to(values.dtype), values)
        return v.contiguous(), series, prior


class AttentionLayer(nn.Module):
    def __init__(self, win_size: int, d_model: int, n_heads: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        d_keys = d_model // n_heads
        self.inner_attention = AnomalyAttention(win_size, dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads

        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(queries).view(B, L, H)

        out, series, prior = self.inner_attention(q, k, v, sigma)
        return self.out_projection(out.view(B, L, -1)), series, prior


class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int,
                 dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        new_x, series, prior = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), series, prior


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer], norm_layer=None) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        series_list, prior_list = [], []
        for layer in self.attn_layers:
            x, series, prior = layer(x)
            series_list.append(series)
            prior_list.append(prior)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list


class AnomalyTransformer(nn.Module):
    """Config A: d_model=512, n_heads=8, e_layers=3, d_ff=512, win_size=100."""

    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(win_size, d_model, n_heads, dropout),
                    d_model, d_ff, dropout, activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """Returns ``(reconstruction, series_list, prior_list)``."""
        enc, series_list, prior_list = self.encoder(self.embedding(x))
        return self.projection(enc), series_list, prior_list

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
