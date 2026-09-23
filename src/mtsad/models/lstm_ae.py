"""LSTM autoencoder baseline.

The number the Anomaly Transformer has to beat. Classic repeat-vector design:
encode the window to the encoder's final hidden state, tile that across the
window, decode it back. Anomaly score is plain per-timestep reconstruction
error -- no association discrepancy, which is exactly the ablation that
isolates what the paper's mechanism contributes.

``forward`` returns ``(reconstruction, [], [])`` so this model is a drop-in
for ``AnomalyTransformer`` in the shared training loop and the shared scoring
path. Empty association lists make the training loop skip the minimax term.
That sharing is the point: the comparison has to be an artifact of the models,
not of two different eval implementations.

Default ``hidden_size=384`` puts it at ~4.2 M parameters against the Anomaly
Transformer's 4.8 M, so neither side wins on capacity alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=lstm_dropout,
        )
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor):
        """``x``: (B, L, F). Returns ``(reconstruction, [], [])``."""
        _, (h, _) = self.encoder(x)
        latent = h[-1]                                    # (B, hidden)
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded), [], []

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
