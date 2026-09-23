"""Phase 3 tests.

``test_point_adjust_matches_reference_loop`` is the load-bearing one: our
segment-based implementation must agree with the literal loop from the
reference solver on random inputs, because every adjusted number we report
depends on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.eval.metrics import (
    all_positive_baseline,
    best_threshold,
    evaluate,
    evaluate_at_threshold,
    point_adjust,
    random_baseline,
    segments,
    threshold_at_quantile,
)
from mtsad.eval.scoring import align_labels


def _reference_point_adjust(y_true, y_pred):
    """Verbatim transcription of the reference solver's loop."""
    y_true = np.asarray(y_true).astype(int).tolist()
    y_pred = np.asarray(y_pred).astype(int).copy().tolist()
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if y_true[j] == 0:
                    break
                y_pred[j] = 1
            for j in range(i, len(y_true)):
                if y_true[j] == 0:
                    break
                y_pred[j] = 1
        elif y_true[i] == 0:
            anomaly_state = False
        if anomaly_state:
            y_pred[i] = 1
    return np.array(y_pred, dtype=np.int8)


# ------------------------------------------------------------- segments ----


def test_segments_finds_half_open_runs():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert segments(y) == [(1, 3), (5, 6), (7, 10)]


def test_segments_handles_edges():
    assert segments(np.array([1, 1, 0])) == [(0, 2)]
    assert segments(np.array([0, 1, 1])) == [(1, 3)]
    assert segments(np.array([1, 1, 1])) == [(0, 3)]
    assert segments(np.zeros(5)) == []


# ------------------------------------------------------- point adjustment ----


def test_point_adjust_expands_a_hit_segment():
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    np.testing.assert_array_equal(
        point_adjust(y_true, y_pred), [0, 1, 1, 1, 0]
    )


def test_point_adjust_leaves_a_missed_segment_alone():
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    np.testing.assert_array_equal(point_adjust(y_true, y_pred), np.zeros(5))


def test_point_adjust_keeps_false_positives():
    """Adjustment must not forgive FPs outside true segments."""
    y_true = np.array([0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    out = point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(out, [1, 0, 1, 1, 0, 1])


@pytest.mark.parametrize("seed", range(25))
def test_point_adjust_matches_reference_loop(seed):
    rng = np.random.default_rng(seed)
    n = 200
    y_true = (rng.random(n) < 0.2).astype(np.int8)
    y_pred = (rng.random(n) < 0.3).astype(np.int8)
    np.testing.assert_array_equal(
        point_adjust(y_true, y_pred), _reference_point_adjust(y_true, y_pred)
    )


def test_point_adjust_does_not_mutate_input():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    original = y_pred.copy()
    point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(y_pred, original)


# -------------------------------------------------------------- metrics ----


def test_all_positive_raw_f1_matches_closed_form():
    y = np.zeros(1000, dtype=np.int8)
    y[:100] = 1  # p = 0.1
    res = all_positive_baseline(y)
    p = 0.1
    assert res.raw.f1 == pytest.approx(2 * p / (1 + p))
    assert res.raw.recall == pytest.approx(1.0)
    # Everything flagged means every segment is hit, so adjustment gives 1.0.
    assert res.adjusted.f1 == pytest.approx(2 * p / (1 + p))


def test_adjustment_never_lowers_f1_on_a_reasonable_scorer():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.15).astype(np.int8)
    scores = rng.random(2000) + y * 0.3
    res = evaluate("t", y, scores, n_thresholds=200)
    assert res.adjusted.f1 >= res.raw.f1


def test_auc_is_computed_before_adjustment():
    """AUC must depend only on the continuous scores and labels."""
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(np.int8)
    scores = rng.random(500)
    a = evaluate("a", y, scores, n_thresholds=50)
    b = evaluate("b", y, scores, n_thresholds=500)
    # Changing the threshold grid changes F1 resolution but cannot move AUC.
    assert a.roc_auc == pytest.approx(b.roc_auc)
    assert a.pr_auc == pytest.approx(b.pr_auc)


def test_random_scorer_has_chance_auc_but_near_perfect_adjusted_f1():
    """The Kim et al. (2022) result, reproduced on SMAP-shaped labels.

    A scorer carrying zero information sits at chance ROC-AUC, yet point
    adjustment lifts its F1 into the range papers report as state of the art.
    """
    rng = np.random.default_rng(3)
    n = 20000
    y = np.zeros(n, dtype=np.int8)
    # 18 segments of 140 steps -> ~12.6% positive, matching SMAP's 12.79%.
    for s in rng.choice(np.arange(0, n - 140, 1000), size=18, replace=False):
        y[s : s + 140] = 1
    assert 0.10 < y.mean() < 0.15, "fixture should mirror SMAP's positive rate"

    res = random_baseline(y, seed=0, n_thresholds=400)

    # No information: ranking is at chance.
    assert res.roc_auc == pytest.approx(0.5, abs=0.05)
    # Raw F1 cannot beat the all-positive floor by much.
    assert res.raw.f1 < 0.30
    # Yet adjusted F1 lands in "published result" territory.
    assert res.adjusted.f1 > 0.80, f"expected heavy inflation, got {res.adjusted.f1}"
    assert res.inflation > 3.0


def test_best_f1_never_loses_to_the_all_positive_floor():
    """All-positive is reachable at the lowest threshold, so it is a floor.

    Regression guard: with ">" instead of ">=", scores tied at the minimum
    were never all flagged. On SMAP 70% of scores tied at exactly 0.0 and the
    model scored 0.1950 against the trivial baseline's 0.2268.
    """
    rng = np.random.default_rng(11)
    y = (rng.random(4000) < 0.13).astype(np.int8)
    for scores in (
        rng.random(4000),                                  # continuous
        np.where(rng.random(4000) < 0.7, 0.0, rng.random(4000)),  # 70% tied
        np.zeros(4000),                                    # fully degenerate
    ):
        floor = all_positive_baseline(y).raw.f1
        got = best_threshold(y, scores, adjust=False, n_thresholds=300).f1
        assert got >= floor - 1e-9, f"{got} < all-positive floor {floor}"


def test_log_space_scoring_preserves_ranking():
    """log(w) + log(e) must rank identically to w * e, without underflowing."""
    import torch

    from mtsad.models.losses import anomaly_score

    torch.manual_seed(0)
    # Realistic shape: 8 heads, L=100, and a sharply peaked series against a
    # broad prior, which is what drives the KL to ~16 nats in practice.
    series = [torch.softmax(torch.randn(2, 8, 100, 100) * 6, dim=-1)
              for _ in range(3)]
    prior = [torch.rand(2, 8, 100, 100) + 0.05 for _ in range(3)]
    err = torch.rand(2, 100) + 0.01

    # Small temperature keeps the product representable, so the two agree.
    prod = anomaly_score(err, series, prior, temperature=0.02, log_space=False)
    logs = anomaly_score(err, series, prior, temperature=0.02, log_space=True)
    assert torch.equal(prod.flatten().argsort(), logs.flatten().argsort())

    # At the real temperature the product underflows; log space does not.
    prod50 = anomaly_score(err, series, prior, temperature=50.0, log_space=False)
    logs50 = anomaly_score(err, series, prior, temperature=50.0, log_space=True)
    assert (prod50 == 0).any(), "expected float32 underflow at temperature 50"
    assert torch.isfinite(logs50).all()
    assert len(torch.unique(logs50)) > len(torch.unique(prod50))


def _score_fixture(n_steps: int = 32):
    import torch

    torch.manual_seed(0)
    series = [torch.softmax(torch.randn(1, 4, n_steps, n_steps) * 4, dim=-1)
              for _ in range(3)]
    prior = [torch.rand(1, 4, n_steps, n_steps) + 0.05 for _ in range(3)]
    return series, prior


def test_log_space_survives_exactly_zero_reconstruction_error():
    """A perfect reconstruction gives rec_err == 0.0; log(0) would be -inf.

    Rare but reachable: early in training, and on the constant-valued command
    channels that make up most of SMAP/MSL. Without the floor this would
    reintroduce the underflow failure one stage earlier in the pipeline.
    """
    import torch

    from mtsad.models.losses import REC_ERR_FLOOR, anomaly_score

    series, prior = _score_fixture()
    err = torch.rand(1, 32)
    err[0, :5] = 0.0

    out = anomaly_score(err, series, prior, log_space=True)
    assert torch.isfinite(out).all(), "zero rec_err must not produce -inf"

    # Each zeroed entry must equal what it would be had rec_err been exactly
    # the floor. (The absolute value is log_softmax(logits) + log(floor), so
    # it is well below log(floor) alone -- comparing to log(floor) directly
    # would be wrong.)
    floored_input = err.clone()
    floored_input[0, :5] = REC_ERR_FLOOR
    assert torch.equal(out, anomaly_score(floored_input, series, prior,
                                          log_space=True))


def test_all_zero_reconstruction_error_stays_finite():
    """Degenerate extreme: every timestep reconstructed perfectly."""
    import torch

    from mtsad.models.losses import anomaly_score

    series, prior = _score_fixture()
    out = anomaly_score(torch.zeros(1, 32), series, prior, log_space=True)
    assert torch.isfinite(out).all()


def test_rec_err_floor_leaves_normal_values_unperturbed():
    """clamp_min, not `+ eps`: legitimate values must be bit-identical.

    Adding an epsilon would shift the whole score vector and perturb the
    ranking; clamping only rewrites the pathological entries.
    """
    import torch

    from mtsad.models.losses import anomaly_score

    series, prior = _score_fixture()
    err = torch.rand(1, 32) + 0.01  # all well above the floor
    floored = anomaly_score(err, series, prior, log_space=True)
    unfloored = anomaly_score(err, series, prior, log_space=True,
                              rec_err_floor=0.0)
    assert torch.equal(floored, unfloored)


def test_rec_err_floor_is_representable_in_float32():
    import torch

    from mtsad.models.losses import REC_ERR_FLOOR

    as_f32 = float(torch.tensor(REC_ERR_FLOOR, dtype=torch.float32))
    assert as_f32 > 0, "floor must not flush to zero in float32"
    assert as_f32 > torch.finfo(torch.float32).tiny, "floor must be normal"


def test_evaluate_rejects_shape_mismatch_and_nan():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate("x", y, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="non-finite"):
        evaluate("x", y, np.array([0.1, np.nan, 0.3, 0.4]))


def test_evaluate_at_fixed_threshold_agrees_with_sweep_bound():
    rng = np.random.default_rng(5)
    y = (rng.random(1000) < 0.1).astype(np.int8)
    scores = rng.random(1000) + y * 0.5
    best = best_threshold(y, scores, adjust=False, n_thresholds=300)
    raw, _ = evaluate_at_threshold(y, scores, best.threshold)
    assert raw.f1 == pytest.approx(best.f1)


def test_threshold_at_quantile_uses_only_reference_scores():
    ref = np.arange(100, dtype=np.float64)
    assert threshold_at_quantile(ref, 0.99) == pytest.approx(98.01)


# ------------------------------------------------------------- alignment ----


def test_align_labels_trims_the_unscored_tail():
    scores = np.zeros(400)
    labels = np.zeros(417)
    s, lab = align_labels(scores, labels)
    assert len(s) == len(lab) == 400


def test_align_labels_rejects_more_scores_than_labels():
    with pytest.raises(ValueError, match="overlapping or shuffled"):
        align_labels(np.zeros(500), np.zeros(400))
