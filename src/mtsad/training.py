"""Shared training loop for the Anomaly Transformer and the LSTM-AE baseline.

One loop serves both. A model returning empty association lists (the LSTM-AE)
trains on reconstruction alone; one returning populated lists gets the minimax
objective. Same split, same early stopping, same everything else -- so the
Phase 3 comparison measures the models rather than two eval harnesses.

Mixed precision: attention and the FFN run under bf16 autocast, while the
association discrepancy is forced to fp32 inside
``losses.association_discrepancy`` (it takes ``log(p + 1e-4)`` of near-zero
probabilities). bf16 rather than fp16 because SMAP/MSL standardize to z-scores
as extreme as 328.9, which overflows fp16's 65504 ceiling once squared -- see
``TrainConfig.amp_dtype``.

``nan_check_steps`` defaults to every step, not a warmup window: the original
500-step guard expired before the first NaN appeared and the run completed
"successfully" with a corrupt checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.nn as nn
from .models.losses import association_discrepancy, minimax_loss


class AssociationLossNaN(RuntimeError):
    """Raised when the fp32-guarded KL terms still go non-finite."""


@dataclass
class TrainConfig:
    epochs: int = 3                 # reference uses 3 for SMAP/MSL
    lr: float = 1e-4
    k: float = 3.0                  # association-discrepancy weight
    batch_size: int = 64
    grad_accum: int = 1
    amp: bool = True

    # bf16, not fp16. SMAP/MSL standardize to extreme z-scores -- feature 10
    # is a command flag that fires ~1 step in 108k, so its z reaches 328.9,
    # and 9 of 25 SMAP features exceed |z| > 50. fp16 overflows at 65504, so
    # squaring anything past ~256 gives inf, and internal activations from
    # those spikes produced intermittent NaN from ~step 500 onward. bf16 has
    # fp32's 8-bit exponent, so the range is a non-issue; on sm_89 it runs at
    # the same speed as fp16. The reference trains in fp32 and never hits this.
    amp_dtype: str = "bf16"         # "bf16" | "fp16"

    grad_clip: float | None = 1.0
    patience: int = 3

    # Always on, not a warmup window. The original 500-step guard expired
    # before the first NaN and the run "succeeded" with a corrupt checkpoint.
    # The check is torch.isfinite on four scalars -- unmeasurable next to a
    # forward pass. Set a positive value to limit it to that many steps.
    nan_check_steps: int = -1       # -1 = every step

    lr_halve_each_epoch: bool = True
    log_every: int = 200
    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[self.amp_dtype]

    @property
    def needs_grad_scaler(self) -> bool:
        """Loss scaling exists to stop fp16 gradients underflowing. bf16 has
        the exponent range to not need it."""
        return self.amp and self.amp_dtype == "fp16"


@dataclass
class EpochStats:
    epoch: int
    train_rec: float
    train_series: float
    train_prior: float
    val_rec: float
    val_objective: float
    seconds: float
    lr: float
    peak_gb: float


@dataclass
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("inf")
    checkpoint: Path | None = None
    stopped_early: bool = False


def _finite_or_raise(step: int, **named: torch.Tensor) -> None:
    bad = [n for n, t in named.items() if not torch.isfinite(t).all()]
    if bad:
        detail = ", ".join(f"{n}={named[n].item():.4g}" for n in bad)
        raise AssociationLossNaN(
            f"non-finite loss at step {step}: {detail}.\n"
            f"The association KL is already forced to fp32, so if only "
            f"'rec_loss' is bad the cause is the forward pass overflowing: "
            f"SMAP/MSL contain features whose z-scores reach ~330, which "
            f"exceeds fp16 range once squared. Use amp_dtype='bf16' (the "
            f"default) or set WindowConfig(clip_sigma=...) to bound the input."
        )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    global_step: int = 0,
) -> tuple[float, float, float, int]:
    """One pass. Training when ``optimizer`` is given, else evaluation."""
    train = optimizer is not None
    model.train(train)

    rec_sum = series_sum = prior_sum = 0.0
    n = 0

    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", dtype=cfg.torch_dtype, enabled=cfg.amp):
                rec, series, prior = model(x)
                # Reconstruction in fp32: it feeds the same objective as the
                # fp32 KL terms and mixing precisions across a sum is asking
                # for silent underflow.
                rec_loss = F.mse_loss(rec.float(), x)

            if series:
                terms = association_discrepancy(series, prior)
                total, series_scalar, prior_scalar = minimax_loss(
                    rec_loss, terms, k=cfg.k
                )
            else:
                # Reconstruction-only models (the LSTM-AE baseline) return
                # empty association lists. Same loop, same split, same early
                # stopping -- so the Phase 3 comparison cannot be an artifact
                # of two different training harnesses.
                total = rec_loss
                series_scalar = prior_scalar = torch.zeros((), device=x.device)

            if cfg.nan_check_steps < 0 or global_step < cfg.nan_check_steps:
                _finite_or_raise(
                    global_step,
                    rec_loss=rec_loss,
                    series_term=series_scalar,
                    prior_term=prior_scalar,
                    total=total,
                )

        if train:
            assert optimizer is not None and scaler is not None
            scaler.scale(total / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                if cfg.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if cfg.log_every and global_step % cfg.log_every == 0:
                # series_scalar and prior_scalar are always numerically equal:
                # both are the symmetrised KL, which is symmetric in its two
                # arguments. They differ only in where detach() sits, i.e. in
                # which branch receives gradient. Log one.
                print(
                    f"      step {global_step:>6}  rec {rec_loss.item():.5f}  "
                    f"assoc {series_scalar.item():+.5f}",
                    flush=True,
                )

        bs = x.size(0)
        rec_sum += rec_loss.item() * bs
        series_sum += series_scalar.item() * bs
        prior_sum += prior_scalar.item() * bs
        n += bs

    return rec_sum / n, series_sum / n, prior_sum / n, global_step


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    tag: str = "model",
) -> TrainResult:
    """Train with the minimax objective, early stopping on the val objective."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.needs_grad_scaler)

    result = TrainResult()
    bad_epochs = 0
    global_step = 0

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result.checkpoint = checkpoint_dir / f"{tag}.pt"

    for epoch in range(cfg.epochs):
        if cfg.lr_halve_each_epoch and epoch > 0:
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * (0.5**epoch)
        lr_now = optimizer.param_groups[0]["lr"]

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        tr_rec, tr_series, tr_prior, global_step = _run_epoch(
            model, train_loader, cfg, device, optimizer, scaler, global_step
        )
        va_rec, va_series, _, _ = _run_epoch(model, val_loader, cfg, device)

        # Early-stop on the maximize-phase objective, matching the reference.
        va_obj = va_rec - cfg.k * va_series

        stats = EpochStats(
            epoch=epoch,
            train_rec=tr_rec,
            train_series=tr_series,
            train_prior=tr_prior,
            val_rec=va_rec,
            val_objective=va_obj,
            seconds=time.perf_counter() - t0,
            lr=lr_now,
            peak_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
        )
        result.history.append(stats)

        print(
            f"  epoch {epoch + 1}/{cfg.epochs}  "
            f"train rec {tr_rec:.5f}  series {tr_series:+.5f}  "
            f"prior {tr_prior:+.5f}  |  val rec {va_rec:.5f}  "
            f"obj {va_obj:+.5f}  |  {stats.seconds:.0f}s  "
            f"lr {lr_now:.2e}  peak {stats.peak_gb:.2f} GB",
            flush=True,
        )

        if va_obj < result.best_val:
            result.best_val, result.best_epoch, bad_epochs = va_obj, epoch, 0
            if result.checkpoint is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_objective": va_obj,
                        "config": cfg.__dict__,
                    },
                    result.checkpoint,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop: no improvement for {cfg.patience} epochs")
                result.stopped_early = True
                break

    if result.checkpoint is not None and result.checkpoint.exists():
        model.load_state_dict(
            torch.load(
                result.checkpoint, map_location=device, weights_only=True
            )["model"]
        )

    return result

