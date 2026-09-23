"""Phase 1 regression tests.

Tests that need the dataset skip cleanly when ``data/`` is absent, so the suite
still runs on a fresh clone before ``scripts/01_download_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import Scaler, SlidingWindowDataset, WindowConfig, build_splits
from mtsad.data.telemanom import (
    EXCLUDED_CHANNELS,
    EXPECTED,
    build_concat,
    channel_ids,
    load_labels,
)
from mtsad.data.verify import verify_all

DATA = REPO / "data"
needs_data = pytest.mark.skipif(
    not (DATA / "raw" / "labeled_anomalies.csv").exists(),
    reason="dataset not downloaded; run scripts/01_download_data.py",
)
needs_reference = pytest.mark.skipif(
    not (DATA / "reference" / "SMAP_train.npy").exists(),
    reason="reference arrays not downloaded",
)


# --------------------------------------------------------------- scaler ----


def test_scaler_fits_on_train_only():
    train = np.array([[0.0], [2.0], [4.0]])
    s = Scaler("standard").fit(train)
    # Test data far from the train distribution must NOT re-centre the scaler.
    out = s.transform(np.array([[100.0]]))
    assert out[0, 0] > 50


def test_scaler_clamps_zero_variance():
    x = np.column_stack([np.arange(10.0), np.ones(10)])  # col 1 is constant
    s = Scaler("standard").fit(x)
    assert s.n_degenerate_ == 1
    out = s.transform(x)
    assert np.isfinite(out).all()
    assert (out[:, 1] == 0).all()


def test_scaler_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Scaler("robust")


def test_scaler_requires_fit():
    with pytest.raises(RuntimeError):
        Scaler("standard").transform(np.zeros((2, 2)))


# -------------------------------------------------------------- windows ----


def test_window_count_and_shapes():
    data = np.zeros((250, 3))
    ds = SlidingWindowDataset(data, None, win_size=100, stride=100)
    assert len(ds) == 2  # starts 0 and 100; 200..300 would overrun
    x, y = ds[0]
    assert tuple(x.shape) == (100, 3)
    assert tuple(y.shape) == (100,)


def test_windows_are_contiguous_and_ordered():
    data = np.arange(500, dtype=np.float64).reshape(500, 1)
    ds = SlidingWindowDataset(data, None, win_size=10, stride=10)
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert x0[0, 0] == 0 and x0[-1, 0] == 9
    assert x1[0, 0] == 10


def test_labels_are_aligned_to_their_window():
    data = np.zeros((300, 2))
    labels = np.zeros(300)
    labels[150:160] = 1
    ds = SlidingWindowDataset(data, labels, win_size=100, stride=100)
    assert ds[0][1].sum() == 0  # window 0..100
    assert ds[1][1].sum() == 10  # window 100..200 holds all ten


def test_window_longer_than_series_raises():
    with pytest.raises(ValueError):
        SlidingWindowDataset(np.zeros((10, 2)), None, win_size=100, stride=1)


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError):
        SlidingWindowDataset(np.zeros((100, 2)), np.zeros(99), win_size=10, stride=1)


# ------------------------------------------------------------ benchmark ----


@needs_data
def test_p2_is_duplicated_and_excluded():
    df = load_labels(DATA)
    dupes = df.loc[df["chan_id"].duplicated(), "chan_id"].tolist()
    assert dupes == ["P-2"], "the known csv duplicate changed"
    assert "P-2" not in channel_ids(DATA, "SMAP")
    assert "P-2" in channel_ids(DATA, "SMAP", literature_subset=False)
    assert EXCLUDED_CHANNELS == {"P-2"}


@needs_data
def test_channels_are_lexicographically_ordered():
    ids = channel_ids(DATA, "SMAP")
    assert ids == sorted(ids)
    assert ids.index("D-11") < ids.index("D-2")  # string sort, not numeric


@needs_data
@pytest.mark.parametrize("spacecraft", ["SMAP", "MSL"])
def test_concat_matches_published_shapes(spacecraft):
    train, test, labels, _, bounds = build_concat(DATA, spacecraft)
    exp = EXPECTED[spacecraft]
    assert train.shape == (exp["train"], exp["features"])
    assert test.shape == (exp["test"], exp["features"])
    assert labels.shape == (exp["test"],)
    assert len(bounds) - 1 == exp["channels"]


@needs_data
@needs_reference
def test_rebuild_is_bit_exact_against_reference():
    ok, lines = verify_all(DATA)
    assert ok, "\n".join(lines)


@needs_data
def test_labels_are_binary_and_nonempty():
    _, _, labels, _, _ = build_concat(DATA, "SMAP")
    assert set(np.unique(labels)) <= {0, 1}
    assert 0.05 < labels.mean() < 0.25


# --------------------------------------------------------------- splits ----


@needs_data
def test_val_split_is_per_channel_in_concat_mode():
    """Concat mode must split inside each channel, not slice the spliced tail.

    Slicing the tail holds out the last ~10 channels outright rather than
    later time, so train and val land on disjoint spacecraft subsystems. That
    showed up as a 23x train/val reconstruction gap during Phase 2 training.
    """
    cfg = WindowConfig(val_fraction=0.2)
    train_raw, _, _, bounds, _ = build_concat(DATA, "MSL")
    sp = build_splits(DATA, "MSL", cfg)

    # Per-channel split: totals are the sum of each channel's own 80/20.
    expected_val = sum(
        int((int(b) - int(a)) * 0.2) for a, b in zip(bounds[:-1], bounds[1:])
    )
    assert len(sp.val.data) == expected_val
    assert len(sp.train.data) + len(sp.val.data) == EXPECTED["MSL"]["train"]

    # A naive tail slice would give exactly int(N * 0.2); per-channel rounding
    # makes these differ, which is the cheapest proof the split is per-channel.
    assert expected_val != int(EXPECTED["MSL"]["train"] * 0.2)


@needs_data
def test_val_takes_the_tail_of_each_channel_not_the_head():
    """Val must be later-in-time within every channel."""
    cfg = WindowConfig(val_fraction=0.2, normalize="none")
    train_raw, _, _, bounds, _ = build_concat(DATA, "MSL")
    sp = build_splits(DATA, "MSL", cfg)

    a, b = int(bounds[0]), int(bounds[1])
    seg = train_raw[a:b]
    n_val = int(len(seg) * 0.2)
    np.testing.assert_allclose(
        sp.val.data[:n_val], seg[len(seg) - n_val :].astype(np.float32), rtol=1e-6
    )


@needs_data
def test_entity_mode_val_split_is_a_plain_chronological_tail():
    cfg = WindowConfig(val_fraction=0.2, win_size=50)
    sp = build_splits(DATA, "SMAP", cfg, mode="entity", chan_id="P-1")
    total = len(sp.train.data) + len(sp.val.data)
    assert len(sp.val.data) == int(total * 0.2)


@needs_data
def test_test_windows_are_non_overlapping_by_default():
    cfg = WindowConfig()
    assert cfg.stride_for("test") == cfg.win_size
    sp = build_splits(DATA, "MSL", cfg)
    # Non-overlapping => each window covers a distinct span of timesteps.
    assert len(sp.test) == (EXPECTED["MSL"]["test"] - cfg.win_size) // cfg.win_size + 1


@needs_data
def test_entity_mode_loads_single_channel():
    sp = build_splits(DATA, "SMAP", WindowConfig(win_size=50), mode="entity",
                      chan_id="P-1")
    assert sp.n_features == 25
    assert sp.test_bounds is None
    assert sp.test_labels.sum() > 0


@needs_data
def test_entity_mode_requires_chan_id():
    with pytest.raises(ValueError):
        build_splits(DATA, "SMAP", mode="entity")


@needs_data
def test_dropping_boundary_windows_shrinks_test_set():
    cfg_keep = WindowConfig(drop_boundary_windows=False)
    cfg_drop = WindowConfig(drop_boundary_windows=True)
    keep = build_splits(DATA, "MSL", cfg_keep)
    drop = build_splits(DATA, "MSL", cfg_drop)
    assert keep.n_boundary_windows > 0
    assert len(drop.test) == len(keep.test) - keep.n_boundary_windows
