"""Phase 4 tests: correctness of the detectors and of the harness measuring them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.drift.adwin import ADWIN, ADWINConfig
from mtsad.drift.base import RecalibrationState
from mtsad.drift.harness import (
    detection_delay,
    false_positive_rate,
    inject_shift,
    synthetic_clean_stream,
)
from mtsad.drift.kswin import KSWIN, KSWINConfig


def stationary(n=4000, seed=0, loc=0.0, scale=1.0):
    return np.random.default_rng(seed).normal(loc, scale, n)


# ----------------------------------------------------------------- ADWIN ----


def test_adwin_stays_quiet_on_a_stationary_stream():
    det = ADWIN(clock=1)
    assert not any(det.update(x) for x in stationary(3000))


def test_adwin_fires_on_a_mean_shift():
    det = ADWIN(clock=1)
    stream = np.concatenate([stationary(1000, 0), stationary(1000, 1, loc=5.0)])
    fired = [i for i, x in enumerate(stream) if det.update(x)]
    assert fired, "ADWIN missed an obvious mean shift"
    assert min(f for f in fired if f >= 1000) - 1000 < 200


def test_adwin_window_shrinks_after_a_change():
    det = ADWIN(clock=1)
    for x in stationary(1000, 0):
        det.update(x)
    wide = det.width
    for x in stationary(300, 1, loc=10.0):
        det.update(x)
    assert det.width < wide, "window should contract after drift"


def test_adwin_window_grows_while_stationary():
    det = ADWIN(clock=1)
    for x in stationary(200, 0):
        det.update(x)
    early = det.width
    for x in stationary(800, 1):
        det.update(x)
    assert det.width > early


def test_adwin_exponential_histogram_bounds_bucket_count():
    """Memory must stay O(log n), not O(n)."""
    det = ADWIN(clock=1)
    for x in stationary(20000, 3):
        det.update(x)
    assert len(det._buckets) < 200, f"bucket list grew to {len(det._buckets)}"
    assert det.width > 1000


def test_adwin_bucket_totals_stay_consistent():
    det = ADWIN(clock=1)
    for x in stationary(5000, 4):
        det.update(x)
    assert det._n == pytest.approx(sum(b[0] for b in det._buckets))
    assert det._total == pytest.approx(sum(b[1] for b in det._buckets), rel=1e-9)
    assert det.variance >= 0.0


def test_adwin_smaller_delta_is_more_conservative():
    stream = np.concatenate([stationary(600, 0), stationary(600, 1, loc=0.6)])
    loose = len(ADWIN(delta=0.2, clock=1).run(stream))
    tight = len(ADWIN(delta=1e-8, clock=1).run(stream))
    assert tight <= loose


def test_adwin_rejects_bad_config():
    with pytest.raises(ValueError):
        ADWIN(delta=0.0)
    with pytest.raises(ValueError):
        ADWIN(clock=0)
    with pytest.raises(TypeError):
        ADWIN(nonexistent_option=1)


def test_adwin_rejects_non_finite_input():
    with pytest.raises(ValueError, match="non-finite"):
        ADWIN().update(float("nan"))


def test_adwin_reset_clears_state():
    det = ADWIN(clock=1)
    for x in stationary(500):
        det.update(x)
    det.reset()
    assert det.width == 0 and det._buckets == []


def test_adwin_config_is_exposed_as_dict():
    det = ADWIN(delta=0.01)
    assert det.config["delta"] == 0.01
    assert set(ADWINConfig().__dict__) <= set(det.config)


# ----------------------------------------------------------------- KSWIN ----


def test_kswin_stays_quiet_on_a_stationary_stream():
    det = KSWIN(window_size=200, reference_size=500, alpha=1e-4, stride=50)
    assert not any(det.update(x) for x in stationary(5000))


def test_kswin_fires_on_a_distribution_shift():
    det = KSWIN(window_size=200, reference_size=500, alpha=1e-3, stride=25)
    stream = np.concatenate([stationary(2000, 0), stationary(2000, 1, loc=2.0)])
    fired = [i for i, x in enumerate(stream) if det.update(x)]
    assert any(f >= 2000 for f in fired)


def test_kswin_catches_variance_increase():
    clean = stationary(3000, 0, scale=1.0)
    shifted = stationary(3000, 1, loc=0.0, scale=6.0)  # same mean
    stream = np.concatenate([clean, shifted])

    ks = KSWIN(window_size=300, reference_size=1000, alpha=1e-3, stride=25)
    ks_fired = [i for i, x in enumerate(stream) if ks.update(x)]
    assert any(f >= 3000 for f in ks_fired), "KS should catch a spread change"


def test_adwin_is_blind_to_a_variance_decrease_but_kswin_is_not():
    """Documented limitation, and the reason both detectors ship.

    ADWIN cuts when two sub-window *means* diverge. Shrinking the spread
    while holding the mean makes sub-window means converge, so there is
    nothing for it to cut on. Confirmed on real SMAP and MSL scores: ADWIN
    MISSED a 'variance x0.33' injection on both, while KSWIN caught it in
    99 steps.
    """
    base = stationary(4000, 0, loc=0.0, scale=1.0)
    tail = base[2000:] * (1 / 3)  # same mean, one third the spread
    stream = np.concatenate([base[:2000], tail])

    adwin_fired = [
        e.index for e in ADWIN(clock=1).run(stream) if e.index >= 2000
    ]
    ks_fired = [
        e.index
        for e in KSWIN(window_size=300, reference_size=1000, alpha=1e-3,
                       stride=25).run(stream)
        if e.index >= 2000
    ]
    assert not adwin_fired, "ADWIN is expected to miss a pure variance drop"
    assert ks_fired, "KSWIN should catch it"


def test_kswin_uses_supplied_reference_without_calibrating():
    ref = stationary(500, 7)
    det = KSWIN(reference=ref, window_size=100, alpha=1e-3, stride=10)
    assert det.is_calibrated
    det.update(0.0)
    assert det.is_calibrated


def test_kswin_calibrates_from_stream_head_when_no_reference():
    det = KSWIN(window_size=50, reference_size=100, stride=10)
    assert not det.is_calibrated
    for x in stationary(100):
        det.update(x)
    assert det.is_calibrated


def test_kswin_cooldown_suppresses_alarm_storms():
    stream = np.concatenate([stationary(1000, 0), stationary(3000, 1, loc=3.0)])
    hot = len(KSWIN(window_size=200, reference_size=500, alpha=0.01,
                    stride=10, cooldown=0).run(stream))
    cool = len(KSWIN(window_size=200, reference_size=500, alpha=0.01,
                     stride=10, cooldown=500).run(stream))
    assert cool < hot


def test_kswin_rejects_bad_config():
    with pytest.raises(ValueError):
        KSWIN(alpha=0.0)
    with pytest.raises(ValueError):
        KSWIN(window_size=1)
    with pytest.raises(TypeError):
        KSWIN(bogus=3)


# --------------------------------------------------------------- harness ----


@pytest.mark.parametrize("kind", ["offset", "scale", "variance"])
def test_inject_shift_only_touches_the_tail(kind):
    base = stationary(1000)
    out = inject_shift(base, start=500, kind=kind, magnitude=3.0)
    np.testing.assert_array_equal(out[:500], base[:500])
    assert not np.allclose(out[500:], base[500:])


def test_inject_splice_uses_donor_distribution():
    base = stationary(1000, 0, loc=0.0)
    donor = stationary(1000, 1, loc=20.0)
    out = inject_shift(base, 500, kind="splice", donor=donor)
    assert out[500:].mean() > 15.0
    with pytest.raises(ValueError, match="donor"):
        inject_shift(base, 500, kind="splice")


def test_inject_shift_validates_arguments():
    with pytest.raises(ValueError, match="outside stream"):
        inject_shift(stationary(100), start=500)
    with pytest.raises(ValueError, match="unknown shift kind"):
        inject_shift(stationary(100), 50, kind="nope")


def test_detection_delay_reports_first_alarm_after_shift():
    stream = inject_shift(stationary(4000), 2000, "offset", 4.0)
    res = detection_delay(lambda: ADWIN(clock=1), stream, shift_at=2000)
    assert res.detected and res.delay is not None and res.delay >= 0
    assert res.first_alarm >= 2000


def test_detection_delay_reports_a_miss_rather_than_crashing():
    res = detection_delay(
        lambda: ADWIN(delta=1e-12, clock=1), stationary(2000), shift_at=1000
    )
    if not res.detected:
        assert res.delay is None and res.first_alarm is None


def test_false_positive_rate_on_synthetic_stationary_stream():
    clean = synthetic_clean_stream(stationary(2000), n=10000, seed=1)
    res = false_positive_rate(lambda: ADWIN(clock=1), clean)
    assert res.n_steps == 10000
    assert res.rate_per_1000 == pytest.approx(1000 * res.n_alarms / 10000)


def test_synthetic_clean_stream_is_stationary():
    ref = stationary(1000)
    out = synthetic_clean_stream(ref, 8000, seed=2)
    first, second = out[:4000], out[4000:]
    assert abs(first.mean() - second.mean()) < 0.2


# ------------------------------------------------------- recalibration ----


def test_recalibration_flag_latches_until_acknowledged():
    state = RecalibrationState(ADWIN(clock=1))
    stream = inject_shift(stationary(3000), 1500, "offset", 5.0)
    for x in stream:
        state.update(x)
    assert state.needs_recalibration
    assert state.events

    state.acknowledge()
    assert not state.needs_recalibration
    assert state.events, "history must survive an acknowledgement"


def test_recalibration_summary_is_serialisable():
    state = RecalibrationState(KSWIN(window_size=50, reference_size=100))
    for x in stationary(300):
        state.update(x)
    summary = state.summary()
    assert summary["detector"] == "KSWIN"
    assert "alpha" in summary["config"]
    assert summary["n_seen"] == 300
