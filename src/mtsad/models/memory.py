"""Measure real VRAM for one train step, and pick a batch size that fits.

The analytic estimate for Config A was 2.4-3.0 GB. This module checks that
against ``torch.cuda.max_memory_allocated()`` rather than trusting it, and
falls back to Plan B (batch 32 + grad accumulation x2) automatically if the
measurement comes in over budget or OOMs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .anomaly_transformer import AnomalyTransformer
from .losses import anomaly_score, association_discrepancy, minimax_loss

GB = 1024**3


@dataclass
class MemoryProbe:
    batch_size: int
    ok: bool
    train_alloc_gb: float
    train_reserved_gb: float
    eval_alloc_gb: float
    total_vram_gb: float
    oom: bool = False

    def __str__(self) -> str:
        if self.oom:
            return f"  batch {self.batch_size:>4}  OOM"
        return (
            f"  batch {self.batch_size:>4}  train {self.train_alloc_gb:5.2f} GB "
            f"alloc / {self.train_reserved_gb:5.2f} GB reserved   "
            f"eval {self.eval_alloc_gb:5.2f} GB   "
            f"{'OK' if self.ok else 'OVER BUDGET'}"
        )


def probe_batch(
    batch_size: int,
    win_size: int,
    n_features: int,
    device: torch.device,
    budget_gb: float = 6.5,
    amp: bool = True,
    k: float = 3.0,
    amp_dtype: torch.dtype = torch.bfloat16,
    **model_kwargs,
) -> MemoryProbe:
    """Run one full train step plus one eval step and report peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    total = torch.cuda.get_device_properties(device).total_memory / GB

    try:
        model = AnomalyTransformer(
            win_size=win_size, enc_in=n_features, c_out=n_features, **model_kwargs
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
        x = torch.randn(batch_size, win_size, n_features, device=device)

        # ---- one training step ----
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            rec, series, prior = model(x)
            rec_loss = torch.nn.functional.mse_loss(rec.float(), x)
        terms = association_discrepancy(series, prior)
        total_loss, _, _ = minimax_loss(rec_loss, terms, k=k)
        scaler.scale(total_loss).backward()
        scaler.step(opt)
        scaler.update()

        train_alloc = torch.cuda.max_memory_allocated(device) / GB
        train_reserved = torch.cuda.max_memory_reserved(device) / GB

        # ---- one eval step (scoring path, which Phases 3-5 use) ----
        del rec, series, prior, terms, total_loss, rec_loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                rec, series, prior = model(x)
            err = (rec.float() - x).pow(2).mean(-1)
            _ = anomaly_score(err, series, prior)
        eval_alloc = torch.cuda.max_memory_allocated(device) / GB

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return MemoryProbe(batch_size, False, 0.0, 0.0, 0.0, total, oom=True)

    finally:
        for name in ("model", "opt", "scaler", "x"):
            if name in dir():
                pass
        torch.cuda.empty_cache()

    return MemoryProbe(
        batch_size=batch_size,
        ok=train_alloc <= budget_gb,
        train_alloc_gb=train_alloc,
        train_reserved_gb=train_reserved,
        eval_alloc_gb=eval_alloc,
        total_vram_gb=total,
    )


def select_batch_size(
    preferred: int,
    win_size: int,
    n_features: int,
    device: torch.device,
    budget_gb: float = 6.5,
    amp: bool = True,
    **model_kwargs,
) -> tuple[int, int, list[MemoryProbe]]:
    """Probe ``preferred``; fall back to Plan B if it OOMs or exceeds budget.

    Returns ``(batch_size, grad_accum_steps, probes)``. The fallback halves the
    batch and doubles accumulation, so the effective batch is unchanged.
    """
    probes = [
        probe_batch(
            preferred, win_size, n_features, device, budget_gb, amp, **model_kwargs
        )
    ]
    if probes[-1].ok:
        return preferred, 1, probes

    batch, accum = preferred, 1
    while batch > 8:
        batch //= 2
        accum *= 2
        probes.append(
            probe_batch(
                batch, win_size, n_features, device, budget_gb, amp, **model_kwargs
            )
        )
        if probes[-1].ok:
            return batch, accum, probes

    raise RuntimeError(
        "no batch size >= 8 fits the memory budget; reduce win_size or n_heads"
    )

