"""Turn a trained model into a per-timestep anomaly score over the test set.

Both models go through this one function. The Anomaly Transformer's scores get
weighted by the association discrepancy; the LSTM-AE's are plain
reconstruction error, because it has no associations to weight by. That
difference *is* the thing Phase 3 measures, so everything around it is shared.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.losses import anomaly_score


@torch.no_grad()
def score_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    temperature: float = 50.0,
    use_association: bool = True,
) -> np.ndarray:
    """Per-timestep anomaly scores, concatenated in time order.

    The loader must be unshuffled with non-overlapping windows, so window i
    covers timesteps ``[i * L, (i + 1) * L)`` and concatenating yields one
    score per timestep.
    """
    model.eval()
    chunks: list[np.ndarray] = []

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            rec, series, prior = model(x)

        # Per-timestep reconstruction error, fp32. (B, L)
        err = (rec.float() - x).pow(2).mean(dim=-1)

        if use_association and series:
            score = anomaly_score(err, series, prior, temperature=temperature)
        else:
            score = err

        chunks.append(score.float().cpu().numpy())

    return np.concatenate(chunks, axis=0).reshape(-1)


def align_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim labels to the timesteps the windows actually cover.

    Non-overlapping windows of length L cover ``floor(T / L) * L`` timesteps,
    so the final ``T mod L`` are unscored -- 17 of 427,617 on SMAP, 29 of
    73,729 on MSL. Both models drop the identical tail, so the comparison is
    unaffected; returning the pair explicitly keeps the truncation visible
    rather than letting a silent broadcast hide it.
    """
    n = len(scores)
    if n > len(labels):
        raise ValueError(
            f"got {n} scores for only {len(labels)} labels; the loader is "
            f"probably overlapping or shuffled"
        )
    return scores, np.asarray(labels[:n])
