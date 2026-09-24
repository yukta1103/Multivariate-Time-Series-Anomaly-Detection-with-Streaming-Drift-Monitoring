"""Phase 5 tests for the dashboard's non-UI logic.

Streamlit rendering is not unit testable, so everything that could silently
be wrong lives in ``mtsad.dashboard`` and is tested here: channel slicing,
the replay state machine, manual recalibration, and the reference lookups
that put measured Phase 3/4 numbers on screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.dashboard.data import FULL_STREAM, DashboardBundle, StreamSlice
from mtsad.dashboard.reference import (
    model_quality,
    nearest_point,
    raw_stream_alarm_rate,
    sweep_points,
)
from mtsad.dashboard.session import DetectorSettings, StreamSession

REPORTS = REPO / "reports"
BUNDLE = REPO / "data" / "processed" / "dashboard_smap.npz"

needs_bundle = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="run scripts/07_prepare_dashboard.py first",
)
needs_reports = pytest.mark.skipif(
    not (REPORTS / "phase4_drift.json").exists(),
    reason="run scripts/06_drift.py --sweep first",
)


def synthetic_slice(n=3000, shift_at=1500, seed=0) -> StreamSlice:
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, n)
    scores[shift_at:] += 6.0
    labels = np.zeros(n, dtype=np.int8)
    labels[2000:2100] = 1
    return StreamSlice("synthetic", rng.normal(0, 1, n), labels, scores, 0)


# ------------------------------------------------------------- slicing ----


@needs_bundle
def test_bundle_loads_and_slices_a_channel():
    b = DashboardBundle.load(BUNDLE)
    assert b.spacecraft == "SMAP"
    assert len(b.channel_ids) == 53
    s = b.slice_for(b.channel_ids[0], "at")
    assert len(s) == len(s.telemetry) == len(s.labels) > 0
    assert not s.is_full_stream


@needs_bundle
def test_channel_slices_partition_the_full_stream():
    b = DashboardBundle.load(BUNDLE)
    full = b.slice_for(FULL_STREAM, "at")
    total = sum(len(b.slice_for(c, "at")) for c in b.channel_ids)
    assert total == len(full), "channel slices must tile the whole stream"


@needs_bundle
def test_slice_offset_points_back_into_the_full_stream():
    b = DashboardBundle.load(BUNDLE)
    full = b.slice_for(FULL_STREAM, "at")
    cid = b.channel_ids[3]
    s = b.slice_for(cid, "at")
    np.testing.assert_array_equal(
        s.scores, full.scores[s.offset : s.offset + len(s)]
    )


@needs_bundle
def test_both_models_are_available_and_differ():
    b = DashboardBundle.load(BUNDLE)
    at = b.slice_for(b.channel_ids[0], "at")
    ae = b.slice_for(b.channel_ids[0], "ae")
    assert len(at) == len(ae)
    assert not np.allclose(at.scores, ae.scores)


@needs_bundle
def test_suggested_channels_all_contain_anomalies():
    b = DashboardBundle.load(BUNDLE)
    picks = b.channels_with_anomalies()
    assert picks, "the demo needs at least one channel with anomalies"
    for cid in picks:
        assert b.slice_for(cid, "at").labels.any()


@needs_bundle
def test_unknown_channel_and_model_raise():
    b = DashboardBundle.load(BUNDLE)
    with pytest.raises(KeyError):
        b.slice_for("NOPE-9", "at")
    with pytest.raises(KeyError):
        b.slice_for(b.channel_ids[0], "nope")


def test_anomaly_segment_count():
    s = synthetic_slice()
    assert s.n_anomaly_segments == 1


# ------------------------------------------------------------- session ----


def test_session_advances_and_finishes():
    s = StreamSession(synthetic_slice(n=500, shift_at=250))
    assert s.position == 0 and not s.finished
    s.advance(200)
    assert s.position == 200
    s.advance(10_000)
    assert s.finished and s.position == 500


def test_session_threshold_comes_from_calibration_window_only():
    slice_ = synthetic_slice()
    s = StreamSession(slice_, threshold_quantile=0.99)
    calib = slice_.scores[: s.calibration_size]
    assert s.threshold == pytest.approx(np.quantile(calib, 0.99))
    # The post-shift tail must not have influenced it.
    assert s.threshold < slice_.scores[-100:].mean()


def test_session_raises_drift_after_an_injected_shift():
    s = StreamSession(synthetic_slice(n=3000, shift_at=1500))
    s.advance(3000)
    assert s.needs_recalibration
    assert any(a.index >= 1500 for a in s.alarms)


def test_session_is_quiet_on_a_stationary_stream():
    rng = np.random.default_rng(1)
    n = 3000
    flat = StreamSlice("flat", rng.normal(0, 1, n), np.zeros(n, np.int8),
                       rng.normal(0, 1, n), 0)
    s = StreamSession(flat)
    s.advance(n)
    assert not s.needs_recalibration, f"spurious alarms: {s.alarms}"


def test_recalibration_clears_flag_and_moves_threshold():
    s = StreamSession(synthetic_slice(n=3000, shift_at=1500))
    s.advance(3000)
    before = s.threshold
    new = s.recalibrate(window=1000)
    assert not s.needs_recalibration
    assert s.n_recalibrations == 1
    assert new > before, "threshold should rise after an upward shift"


def test_recalibration_repoints_kswin_so_it_does_not_immediately_refire():
    """Without moving the KS reference, the stale baseline re-fires at once."""
    slice_ = synthetic_slice(n=6000, shift_at=1500)
    s = StreamSession(slice_)
    s.advance(3000)
    s.recalibrate(window=1000)
    s.advance(1000)
    ks_after = [a for a in s.alarms if a.detector == "KSWIN" and a.index >= 3000]
    assert not ks_after, "KSWIN kept firing against the old reference"


def test_changing_settings_rebuilds_and_restarts():
    s = StreamSession(synthetic_slice())
    s.advance(1000)
    assert s.position == 1000
    s.apply_settings(DetectorSettings(adwin_delta=0.05))
    assert s.position == 0 and s.alarms == []


def test_identical_settings_do_not_restart():
    s = StreamSession(synthetic_slice())
    s.advance(1000)
    s.apply_settings(DetectorSettings())
    assert s.position == 1000


def test_reset_returns_to_the_start():
    s = StreamSession(synthetic_slice())
    s.advance(2000)
    s.reset()
    assert s.position == 0 and not s.needs_recalibration and s.alarms == []


def test_visible_window_is_bounded_and_aligned():
    s = StreamSession(synthetic_slice(n=3000, shift_at=1500))
    s.advance(2500)
    v = s.visible(span=500)
    assert len(v["index"]) == 500
    assert v["index"][-1] == 2499
    for key in ("telemetry", "labels", "scores"):
        assert len(v[key]) == 500
    assert set(v["flagged"]).issubset(set(v["index"]))


def test_flagged_points_are_above_threshold():
    s = StreamSession(synthetic_slice(n=3000, shift_at=1500))
    s.advance(3000)
    v = s.visible(span=1000)
    assert (s.stream.scores[v["flagged"]] >= s.threshold).all()


def test_counts_split_by_detector():
    s = StreamSession(synthetic_slice(n=3000, shift_at=1500))
    s.advance(3000)
    c = s.counts()
    assert c["ADWIN"] + c["KSWIN"] == len(s.alarms)


# ----------------------------------------------------------- reference ----


@needs_reports
def test_model_quality_reads_measured_phase3_numbers():
    q = model_quality(REPORTS, "SMAP", "at")
    assert q.roc_auc == pytest.approx(0.5343, abs=1e-3)
    assert q.all_positive_f1 == pytest.approx(0.2268, abs=1e-3)
    assert q.beats_trivial
    assert q.verdict == "barely above chance"


@needs_reports
def test_below_chance_model_is_flagged_as_an_error():
    q = model_quality(REPORTS, "MSL", "at")
    assert q.roc_auc < 0.5
    assert not q.beats_chance
    assert q.verdict == "below chance"
    assert q.severity == "error"


@needs_reports
def test_sweep_points_are_sorted_and_carry_measurements():
    pts = sweep_points(REPORTS, "SMAP", "ADWIN")
    assert [p.value for p in pts] == sorted(p.value for p in pts)
    assert all(p.fp_per_1000 >= 0 for p in pts)


@needs_reports
def test_default_slider_positions_match_the_phase4_headline():
    """The figure shown next to the default slider must be the measured one."""
    assert nearest_point(
        sweep_points(REPORTS, "SMAP", "ADWIN"), 0.002
    ).fp_per_1000 == pytest.approx(0.660)
    assert nearest_point(
        sweep_points(REPORTS, "SMAP", "KSWIN"), 1e-3
    ).fp_per_1000 == pytest.approx(0.020)
    assert nearest_point(
        sweep_points(REPORTS, "MSL", "ADWIN"), 0.002
    ).fp_per_1000 == pytest.approx(0.230)


@needs_reports
def test_raw_stream_rate_is_far_above_the_stationary_rate():
    """Justifies the advanced-view warning text."""
    for sc in ("SMAP", "MSL"):
        raw = raw_stream_alarm_rate(REPORTS, sc, "ADWIN")
        stationary = nearest_point(
            sweep_points(REPORTS, sc, "ADWIN"), 0.002
        ).fp_per_1000
        assert raw > 50 * stationary
