"""Phase 2 tests.

The two that matter most are ``test_single_backward_matches_double_backward``
and ``test_hoisted_normalization_matches_naive``: both optimizations in
``losses.py`` are justified by a claim of numerical identity with the
reference, so the claim is tested rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.models.anomaly_transformer import AnomalyTransformer
from mtsad.models.losses import (
    anomaly_score,
    association_discrepancy,
    kl_per_step,
    minimax_loss,
)
from mtsad.training import AssociationLossNaN, TrainConfig, _finite_or_raise

WIN, FEAT, BATCH = 16, 5, 2
SMALL = dict(d_model=32, n_heads=4, e_layers=2, d_ff=32)


def make_model(**kw):
    torch.manual_seed(0)
    return AnomalyTransformer(WIN, FEAT, FEAT, **{**SMALL, **kw})


def make_batch():
    torch.manual_seed(1)
    return torch.randn(BATCH, WIN, FEAT)


# ---------------------------------------------------------------- model ----


def test_forward_shapes():
    model = make_model()
    rec, series, prior = model(make_batch())
    assert rec.shape == (BATCH, WIN, FEAT)
    assert len(series) == len(prior) == SMALL["e_layers"]
    for s, p in zip(series, prior):
        assert s.shape == (BATCH, SMALL["n_heads"], WIN, WIN)
        assert p.shape == (BATCH, SMALL["n_heads"], WIN, WIN)


def test_series_rows_are_distributions():
    _, series, _ = make_model()(make_batch())
    for s in series:
        assert torch.allclose(s.sum(-1), torch.ones_like(s.sum(-1)), atol=1e-5)


def test_prior_peaks_on_the_diagonal_and_never_sums_to_zero():
    """The prior underflows to exactly 0 at long lags, and that is fine.

    exp(-d^2 / 2*sigma^2) hits fp32 zero once d/sigma is large enough, so the
    matrix is non-negative rather than strictly positive -- which is why the
    KL takes log(p + 1e-4) rather than log(p). What must hold is that the
    diagonal stays positive, since that is what keeps every row sum away from
    zero and the normalization well defined.
    """
    _, _, prior = make_model()(make_batch())
    diag = torch.arange(WIN)
    for p in prior:
        assert (p >= 0).all() and torch.isfinite(p).all()
        assert (p[..., diag, diag] > 0).all(), "diagonal must stay positive"
        assert (p.sum(-1) > 0).all(), "row sums feed a division"
        # Distance 0 is the mode of a zero-centred Gaussian.
        assert (p.argmax(-1) == diag).all()


def test_parameter_count_matches_analytic_formula():
    d, h, layers, ff, c = 512, 8, 3, 512, 25
    model = AnomalyTransformer(100, c, c, d_model=d, n_heads=h,
                               e_layers=layers, d_ff=ff)
    embed = 3 * c * d
    attn = 4 * (d * d + d) + (d * h + h)
    feed = (d * ff + ff) + (ff * d + d)
    norms = 2 * 2 * d
    expected = embed + layers * (attn + feed + norms) + 2 * d + (d * c + c)
    assert model.n_parameters() == expected == 4_798_513


def test_prior_uses_broadcast_not_materialised_distances():
    """distances_sq stays (L, L); materialising it to (B,H,L,L) was the leak."""
    model = make_model()
    buf = model.encoder.attn_layers[0].attention.inner_attention.distances_sq
    assert buf.shape == (WIN, WIN)
    assert torch.equal(buf[0], torch.arange(WIN, dtype=torch.float32) ** 2)


# ----------------------------------------------------------------- loss ----


def test_symmetrised_kl_makes_both_terms_numerically_equal():
    _, series, prior = make_model()(make_batch())
    terms = association_discrepancy(series, prior)
    # Equal values, different gradient paths. Guards against "fixing" a
    # non-bug by making them differ.
    assert torch.allclose(terms.series_term, terms.prior_term, atol=1e-6)


def test_gradient_routing_is_separated_by_detach():
    model = make_model()
    _, series, prior = model(make_batch())
    for s in series:
        s.retain_grad()
    for p in prior:
        p.retain_grad()
    terms = association_discrepancy(series, prior)
    terms.series_term.mean().backward(retain_graph=True)
    # series_term must not push gradient into the prior branch.
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in prior)
    assert any(s.grad is not None and torch.count_nonzero(s.grad) > 0
               for s in series)


def _naive_discrepancy(series_list, prior_list, eps=1e-4):
    """Reference form: renormalise and re-log on every one of the four uses."""
    st = pt = None
    for series, prior in zip(series_list, prior_list):
        series, prior = series.float(), prior.float()
        n = lambda: prior / prior.sum(-1, keepdim=True).clamp_min(eps)  # noqa: E731
        s = kl_per_step(series, n().detach(), eps) + kl_per_step(
            n().detach(), series, eps
        )
        p = kl_per_step(n(), series.detach(), eps) + kl_per_step(
            series.detach(), n(), eps
        )
        st = s if st is None else st + s
        pt = p if pt is None else pt + p
    k = len(series_list)
    return st / k, pt / k


def test_hoisted_normalization_matches_naive():
    """One normalization + one log, reused, must equal recomputing four times."""
    _, series, prior = make_model()(make_batch())
    ours = association_discrepancy(series, prior)
    ref_s, ref_p = _naive_discrepancy(series, prior)
    assert torch.allclose(ours.series_term, ref_s, atol=1e-6)
    assert torch.allclose(ours.prior_term, ref_p, atol=1e-6)


def test_single_backward_matches_double_backward():
    """Our one-pass minimax must equal the reference's two backward calls."""
    x = make_batch()
    k = 3.0

    def grads(single: bool):
        model = make_model()
        model.zero_grad(set_to_none=True)
        rec, series, prior = model(x)
        rec_loss = torch.nn.functional.mse_loss(rec, x)
        terms = association_discrepancy(series, prior)
        if single:
            total, _, _ = minimax_loss(rec_loss, terms, k=k)
            total.backward()
        else:
            loss1 = rec_loss - k * terms.series_term.mean()
            loss2 = rec_loss + k * terms.prior_term.mean()
            loss1.backward(retain_graph=True)
            loss2.backward()
        return [
            p.grad.detach().clone() if p.grad is not None else None
            for p in model.parameters()
        ]

    for a, b in zip(grads(True), grads(False)):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6), "minimax forms diverged"


def test_minimax_applies_reconstruction_twice():
    """loss1 + loss2 carries 2*rec -- a reference quirk we keep deliberately."""
    rec = torch.tensor(1.0, requires_grad=True)
    terms = type("T", (), {"series_term": torch.zeros(1), "prior_term": torch.zeros(1)})
    total, _, _ = minimax_loss(rec, terms, k=3.0)
    assert total.item() == pytest.approx(2.0)


def test_association_discrepancy_rejects_mismatched_lists():
    with pytest.raises(ValueError):
        association_discrepancy([], [])
    with pytest.raises(ValueError):
        association_discrepancy([torch.rand(1, 2, 4, 4)], [])


def test_anomaly_score_shape_and_weighting():
    _, series, prior = make_model()(make_batch())
    err = torch.rand(BATCH, WIN)
    score = anomaly_score(err, series, prior)
    assert score.shape == (BATCH, WIN)
    assert torch.isfinite(score).all()


def test_kl_is_zero_for_identical_distributions():
    p = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    assert torch.allclose(kl_per_step(p, p), torch.zeros(2, 8), atol=1e-6)


# ------------------------------------------------------------ nan guard ----


def test_nan_guard_raises_on_non_finite():
    with pytest.raises(AssociationLossNaN, match="non-finite loss at step 7"):
        _finite_or_raise(7, series_term=torch.tensor(float("nan")))


def test_nan_guard_passes_on_finite():
    _finite_or_raise(0, a=torch.tensor(1.0), b=torch.tensor(-2.0))


def test_nan_guard_runs_every_step_not_just_a_warmup():
    """A 500-step warmup expired before the first NaN and the run "passed"."""
    assert TrainConfig().nan_check_steps == -1


def test_default_amp_dtype_is_bf16_with_no_grad_scaler():
    """fp16 overflowed on SMAP's ~330-sigma command spikes; bf16 does not."""
    cfg = TrainConfig()
    assert cfg.amp_dtype == "bf16"
    assert cfg.torch_dtype is torch.bfloat16
    assert cfg.needs_grad_scaler is False
    assert TrainConfig(amp_dtype="fp16").needs_grad_scaler is True


def test_fp16_overflows_on_the_observed_smap_magnitude():
    """SMAP feature 10 reaches z = 328.9; squaring that exceeds fp16 range."""
    z = torch.tensor([328.9])
    assert torch.isinf(z.half() ** 2).item(), "fp16 should overflow here"
    assert torch.isfinite(z.bfloat16().float() ** 2).item()


# --------------------------------------------------------- fp32 KL claim ----


def test_kl_in_fp16_loses_precision_that_fp32_keeps():
    """Why the KL is forced to fp32: log() of small probabilities in fp16.

    Not a NaN demo -- it shows the fp16 path is measurably wrong on the exact
    kind of near-zero probabilities the prior produces at long lags.
    """
    p = torch.full((1, 1, 4, 4), 1e-7)
    p = p / p.sum(-1, keepdim=True)
    q = torch.softmax(torch.randn(1, 1, 4, 4) * 10, dim=-1)
    exact = kl_per_step(p.double(), q.double())
    f32 = kl_per_step(p.float(), q.float())
    f16 = kl_per_step(p.half(), q.half()).float()
    assert (f32.double() - exact).abs().max() < (f16.double() - exact).abs().max()
