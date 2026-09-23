"""Parse the raw NASA telemanom channels into the arrays the literature uses.

Two views of the same data:

``entity`` mode
    One channel at a time -- 55 SMAP channels or 27 MSL channels, each an
    independent multivariate series with its own anomaly spans. This is the
    methodologically clean view.

``concat`` mode
    All channels of a spacecraft spliced end-to-end into one long array. This
    is what Anomaly Transformer / OmniAnomaly / THOC actually evaluate on, so
    it is the only way to get numbers comparable to published tables. It is
    also a somewhat artificial construct -- unrelated channels are joined at
    hard boundaries -- which we quantify rather than hide (see
    ``count_boundary_windows``).

Raw layout, per channel, shape ``(n_timesteps, n_features)``:
    column 0      the telemetry value being monitored
    columns 1..n  one-hot encoded commands issued to the spacecraft
SMAP has 25 features, MSL has 55.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SPACECRAFT = ("SMAP", "MSL")

# Two quirks of the reference preprocessing, both established empirically by
# diffing our rebuild against the published arrays (see verify.py):
#
# 1. labeled_anomalies.csv ships a duplicate row for channel P-2. Concatenating
#    the csv naively splices P-2 in twice.
# 2. The published pipeline then drops P-2 altogether. So of the 55 SMAP rows
#    in the csv, 54 are unique and 53 survive into the benchmark. Dropping P-2
#    accounts for the row counts exactly:
#    train 138004 - 2821 = 135183 and test 435826 - 8209 = 427617.
#
# Channels are spliced in lexicographic chan_id order ("D-11" before "D-2"),
# not csv row order. csv order does not reproduce the reference arrays.
EXCLUDED_CHANNELS = frozenset({"P-2"})

# Published shapes, used as assertions rather than as inputs.
EXPECTED = {
    "SMAP": {"channels": 53, "features": 25, "train": 135183, "test": 427617},
    "MSL": {"channels": 27, "features": 55, "train": 58317, "test": 73729},
}


@dataclass(frozen=True)
class Channel:
    """One telemetry channel: its arrays and its labelled anomaly spans."""

    chan_id: str
    spacecraft: str
    train: np.ndarray  # (T_train, F)
    test: np.ndarray  # (T_test, F)
    anomaly_sequences: tuple[tuple[int, int], ...]  # inclusive [start, end]

    @property
    def n_features(self) -> int:
        return self.test.shape[1]

    def test_labels(self) -> np.ndarray:
        """Dense 0/1 label vector over the test array."""
        y = np.zeros(len(self.test), dtype=np.uint8)
        for start, end in self.anomaly_sequences:
            y[start : end + 1] = 1
        return y


def load_labels(root: Path) -> pd.DataFrame:
    """Read ``labeled_anomalies.csv`` with the two list columns parsed.

    ``anomaly_sequences`` is valid Python (``[[2149, 2349], ...]``) so
    ``literal_eval`` handles it. ``class`` is *not*: it holds bare unquoted
    tokens (``[contextual, point]``), so it needs splitting by hand.
    """
    df = pd.read_csv(Path(root) / "raw" / "labeled_anomalies.csv")
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    df["class"] = df["class"].apply(_parse_class_list)
    return df


def _parse_class_list(raw: str) -> list[str]:
    inner = str(raw).strip().strip("[]").strip()
    return [tok.strip() for tok in inner.split(",") if tok.strip()] if inner else []


def channel_ids(
    root: Path, spacecraft: str, literature_subset: bool = True
) -> list[str]:
    """Channel ids for one spacecraft, sorted lexicographically and deduplicated.

    Order is load-bearing: it is the order the published preprocessing splices
    in, so changing it silently desynchronises our arrays from the reference
    ones. ``verify.py`` checks this still holds.

    ``literature_subset=True`` drops :data:`EXCLUDED_CHANNELS` to match the
    published benchmark. Pass ``False`` in entity mode, where P-2 is a
    perfectly valid channel to evaluate on its own.
    """
    _check_spacecraft(spacecraft)
    df = load_labels(root)
    ids = df.loc[df["spacecraft"] == spacecraft, "chan_id"].drop_duplicates()
    if literature_subset:
        ids = ids[~ids.isin(EXCLUDED_CHANNELS)]
    return sorted(ids)


def load_channel(root: Path, chan_id: str, spacecraft: str | None = None) -> Channel:
    """Load one channel's train/test arrays and its anomaly spans."""
    root = Path(root)
    df = load_labels(root)
    row = df.loc[df["chan_id"] == chan_id]
    if row.empty:
        raise KeyError(f"unknown channel {chan_id!r}")
    row = row.iloc[0]

    train = np.load(root / "raw" / "train" / f"{chan_id}.npy")
    test = np.load(root / "raw" / "test" / f"{chan_id}.npy")

    # labeled_anomalies.csv carries the expected test length; a mismatch means
    # the download is corrupt or the mirror drifted from the original release.
    if len(test) != int(row["num_values"]):
        raise ValueError(
            f"{chan_id}: test has {len(test)} rows but labels declare "
            f"{int(row['num_values'])}"
        )

    spans = tuple((int(a), int(b)) for a, b in row["anomaly_sequences"])
    for a, b in spans:
        if not (0 <= a <= b < len(test)):
            raise ValueError(f"{chan_id}: anomaly span [{a}, {b}] out of range")

    return Channel(
        chan_id=chan_id,
        spacecraft=spacecraft or str(row["spacecraft"]),
        train=train,
        test=test,
        anomaly_sequences=spans,
    )


def load_spacecraft(
    root: Path, spacecraft: str, literature_subset: bool = True
) -> list[Channel]:
    """All channels of one spacecraft, in canonical order."""
    return [
        load_channel(root, cid, spacecraft)
        for cid in channel_ids(root, spacecraft, literature_subset)
    ]


@lru_cache(maxsize=4)
def _concat_cached(root_str: str, spacecraft: str) -> tuple[np.ndarray, ...]:
    root = Path(root_str)
    channels = load_spacecraft(root, spacecraft, literature_subset=True)

    train = np.concatenate([c.train for c in channels], axis=0)
    test = np.concatenate([c.test for c in channels], axis=0)
    labels = np.concatenate([c.test_labels() for c in channels], axis=0)

    # Offsets where each channel starts in the concatenated test array; windows
    # that straddle one of these join two unrelated spacecraft subsystems.
    test_bounds = np.cumsum([0] + [len(c.test) for c in channels])
    train_bounds = np.cumsum([0] + [len(c.train) for c in channels])

    return train, test, labels, train_bounds, test_bounds


def build_concat(root: Path, spacecraft: str, validate: bool = True):
    """Splice all channels of a spacecraft into the literature-standard arrays.

    Returns ``(train, test, test_labels, train_bounds, test_bounds)``.
    """
    _check_spacecraft(spacecraft)
    train, test, labels, train_bounds, test_bounds = _concat_cached(
        str(Path(root).resolve()), spacecraft
    )

    if validate:
        exp = EXPECTED[spacecraft]
        got = {
            "channels": len(train_bounds) - 1,
            "features": train.shape[1],
            "train": len(train),
            "test": len(test),
        }
        if got != exp:
            raise ValueError(
                f"{spacecraft} concatenation does not match the published "
                f"benchmark.\n  expected {exp}\n  got      {got}"
            )

    return train, test, labels, train_bounds, test_bounds


def count_boundary_windows(bounds: np.ndarray, win: int, stride: int) -> int:
    """How many sliding windows straddle a channel boundary.

    An artifact of ``concat`` mode: such a window contains the tail of one
    spacecraft subsystem and the head of an unrelated one.
    """
    total = int(bounds[-1])
    starts = np.arange(0, total - win + 1, stride)
    interior = bounds[1:-1]  # true joins, excluding the outer edges
    if len(interior) == 0:
        return 0
    # a window [s, s+win) straddles if any interior boundary falls strictly inside
    hit = (interior[None, :] > starts[:, None]) & (
        interior[None, :] < starts[:, None] + win
    )
    return int(hit.any(axis=1).sum())


def _check_spacecraft(name: str) -> None:
    if name not in SPACECRAFT:
        raise ValueError(f"spacecraft must be one of {SPACECRAFT}, got {name!r}")
