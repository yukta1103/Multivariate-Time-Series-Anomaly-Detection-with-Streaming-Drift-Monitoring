"""Association discrepancy and the minimax objective.

Two deliberate departures from the reference implementation, both of which
leave the computed gradients unchanged:

**Hoisted prior normalization.** The reference evaluates
``prior / prior.sum(-1, keepdim=True)`` four separate times per layer, and the
logs of ``series`` and the normalized prior four times each. Each of those is a
full ``(B, H, L, L)`` tensor kept alive for backward. Computing each once per
layer and reusing it drops the live-tensor count per layer from ~30 to ~20 --
roughly a third of total training VRAM at L=100.

**Single backward pass.** The reference calls
``loss1.backward(retain_graph=True)`` then ``loss2.backward()``. Because
gradients accumulate, that is identical to one backward on ``loss1 + loss2``,
and it avoids ``retain_graph=True`` holding the entire graph alive for a second
traversal. ``minimax_loss`` returns that sum directly.

Note ``loss1 + loss2 = 2 * rec + k * (prior_term - series_term)``: the
reconstruction gradient is applied *twice* relative to k. That is a quirk of
the reference, not of the paper, but we keep it so our numbers stay comparable
to published results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# The reference uses 1e-4 inside the logs, not a smaller epsilon. It is load
# bearing: these are length-L distributions whose tails legitimately reach zero,
# and a tighter epsilon drives log() toward -inf and destabilises the minimax.
KL_EPS = 1e-4

# Floor for log(rec_err) in log-space scoring. A perfectly reconstructed
# timestep gives rec_err == 0.0 exactly -- rare, but reachable early in
# training and on the constant-valued command channels that make up most of
# SMAP/MSL. Unguarded, log(0) = -inf, which would reintroduce the underflow
# failure one step earlier in the pipeline.
#
# Applied as clamp_min rather than `rec_err + eps` deliberately: clamping
# leaves every legitimate value bit-identical and only rewrites the
# pathological ones, whereas adding eps perturbs the entire score vector and
# would shift the ranking slightly. 1e-30 maps to log == -69.08 and sits well
# clear of float32's smallest normal (1.18e-38).
REC_ERR_FLOOR = 1e-30


def kl_per_step(p: torch.Tensor, q: torch.Tensor, eps: float = KL_EPS):
    """KL(p || q) per timestep, averaged over heads.

    ``p``, ``q``: ``(B, H, L, L)`` -- distributions over the last axis.
    Returns ``(B, L)``.
    """
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1).mean(1)


@dataclass
class DiscrepancyTerms:
    """Both directions of the association discrepancy, as ``(B, L)`` maps.

    ``series_term`` carries gradient into the series branch only (prior
    detached); ``prior_term`` carries gradient into the prior branch only
    (series detached). That separation is what makes the minimax work.
    """

    series_term: torch.Tensor
    prior_term: torch.Tensor


def association_discrepancy(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    eps: float = KL_EPS,
) -> DiscrepancyTerms:
    """Symmetrised KL between series and prior associations, averaged over layers.

    Computed in fp32 regardless of autocast: these are logs of near-zero
    probabilities and fp16 here produces NaN within a few hundred steps.
    """
    n = len(series_list)
    if n == 0 or n != len(prior_list):
        raise ValueError(
            f"need matching non-empty series/prior lists, got {len(series_list)} "
            f"and {len(prior_list)}"
        )

    series_total: torch.Tensor | None = None
    prior_total: torch.Tensor | None = None

    for series, prior in zip(series_list, prior_list):
        series = series.float()
        prior = prior.float()

        # --- hoisted: one normalization, one log of each, reused four ways ---
        prior_norm = prior / prior.sum(-1, keepdim=True).clamp_min(eps)
        log_s = torch.log(series + eps)
        log_p = torch.log(prior_norm + eps)

        s_d, log_s_d = series.detach(), log_s.detach()
        p_d, log_p_d = prior_norm.detach(), log_p.detach()

        # Gradient reaches `series` only.
        s_term = (series * (log_s - log_p_d)).sum(-1).mean(1) + (
            p_d * (log_p_d - log_s)
        ).sum(-1).mean(1)

        # Gradient reaches `prior` only.
        p_term = (prior_norm * (log_p - log_s_d)).sum(-1).mean(1) + (
            s_d * (log_s_d - log_p)
        ).sum(-1).mean(1)

        series_total = s_term if series_total is None else series_total + s_term
        prior_total = p_term if prior_total is None else prior_total + p_term

    return DiscrepancyTerms(series_total / n, prior_total / n)


def minimax_loss(
    rec_loss: torch.Tensor, terms: DiscrepancyTerms, k: float = 3.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The combined minimax objective.

    Equivalent to the reference's two backward passes:
        loss1 = rec - k * series_term   (maximize phase: push series off prior)
        loss2 = rec + k * prior_term    (minimize phase: pull prior onto series)
    summed into one graph traversal.

    Returns ``(total, series_scalar, prior_scalar)``.
    """
    series_scalar = terms.series_term.mean()
    prior_scalar = terms.prior_term.mean()
    total = 2.0 * rec_loss - k * series_scalar + k * prior_scalar
    return total, series_scalar, prior_scalar


def anomaly_score(
    rec_err: torch.Tensor,
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    temperature: float = 50.0,
    eps: float = KL_EPS,
    log_space: bool = True,
    rec_err_floor: float = REC_ERR_FLOOR,
) -> torch.Tensor:
    """Per-timestep anomaly score: discrepancy-weighted reconstruction error.

    ``rec_err``: ``(B, L)`` per-timestep reconstruction error.
    Returns ``(B, L)``.

    A timestep whose series association is *far* from its Gaussian prior gets a
    large softmax weight, amplifying its reconstruction error. This is the
    paper's criterion -- reconstruction error alone is the baseline we beat in
    Phase 3.

    ``log_space=True`` returns ``log(weight) + log(rec_err)`` instead of their
    product. This is a strictly monotone transform, so every ranking metric
    (ROC/PR AUC, any threshold sweep) is unchanged in exact arithmetic -- but
    it avoids a severe float32 underflow. The logits here span hundreds of
    nats (temperature 50 x a KL of ~16, summed over 3 layers), so the softmax
    saturates and the product underflows: measured on SMAP, **70.06% of scores
    collapsed to exactly 0.0**, which destroyed the ranking among them and
    capped attainable recall at 0.33. Log space keeps those points ordered.
    """
    series_loss: torch.Tensor | None = None
    prior_loss: torch.Tensor | None = None

    for series, prior in zip(series_list, prior_list):
        series = series.float()
        prior = prior.float()
        prior_norm = prior / prior.sum(-1, keepdim=True).clamp_min(eps)

        s = kl_per_step(series, prior_norm, eps) * temperature
        p = kl_per_step(prior_norm, series, eps) * temperature
        series_loss = s if series_loss is None else series_loss + s
        prior_loss = p if prior_loss is None else prior_loss + p

    logits = -series_loss - prior_loss
    if log_space:
        # log(softmax(logits)) + log(rec_err). See REC_ERR_FLOOR: a perfectly
        # reconstructed timestep would otherwise make this -inf.
        return torch.log_softmax(logits, dim=-1) + torch.log(
            rec_err.clamp_min(rec_err_floor)
        )
    return torch.softmax(logits, dim=-1) * rec_err
