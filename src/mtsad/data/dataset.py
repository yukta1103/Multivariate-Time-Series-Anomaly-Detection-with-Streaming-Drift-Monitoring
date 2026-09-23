"""Windowing, normalization and splits for SMAP/MSL.

Memory note: SMAP train is 135k timesteps and a stride-1 sliding window over it
yields ~135k windows. Materialising those as an array would cost ~1.4 GB in
float32 for a dataset whose raw form is 14 MB. We therefore keep the flat
``(T, F)`` array and slice windows lazily in ``__getitem__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .telemanom import build_concat, load_channel


@dataclass(frozen=True)
class WindowConfig:
    """Windowing and normalization settings.

    Defaults reproduce the Anomaly Transformer setup.
    """

    win_size: int = 100
    train_stride: int = 1
    # None means "stride == win_size", i.e. non-overlapping test windows so
    # every test timestep receives exactly one anomaly score. This is what the
    # reference implementation's test loader does and it matters for Phase 3:
    # overlapping test windows would score some timesteps repeatedly and
    # quietly change the F1 denominator.
    test_stride: int | None = None
    normalize: str = "standard"  # "standard" | "minmax" | "none"
    val_fraction: float = 0.2
    drop_boundary_windows: bool = False

    # Optionally bound standardized values to +/- this many sigma.
    #
    # SMAP/MSL are mostly one-hot command flags, and the rare ones are close
    # to constant without being constant: SMAP feature 10 fires about once in
    # 108k steps, so std is 3.0e-03 and firing yields z = 328.9. Nine of 25
    # SMAP features exceed |z| > 50.
    #
    # Left None by default because the reference preprocessing does not clip
    # and we want comparable numbers; training stability is handled instead
    # by bf16, whose exponent range absorbs these fine. Set to e.g. 10.0 to
    # study the effect.
    clip_sigma: float | None = None

    def stride_for(self, split: str) -> int:
        if split == "train":
            return self.train_stride
        return self.test_stride or self.win_size


class Scaler:
    """Per-feature scaler, fit on training data only.

    Fitting on train alone is the point: several public repos fit the scaler on
    the concatenated train+test array, which leaks test statistics into the
    normalization and inflates results.

    Both SMAP and MSL are mostly one-hot command columns, many of which are
    constant within a given slice. Those have zero variance (or zero range), so
    the divisor is clamped to 1.0 rather than producing inf/nan.
    """

    def __init__(self, kind: str = "standard", clip_sigma: float | None = None) -> None:
        if kind not in ("standard", "minmax", "none"):
            raise ValueError(f"unknown normalize kind {kind!r}")
        self.kind = kind
        self.clip_sigma = clip_sigma
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_degenerate_: int = 0

    def fit(self, x: np.ndarray) -> "Scaler":
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "none":
            self.center_ = np.zeros(x.shape[1])
            self.scale_ = np.ones(x.shape[1])
            return self

        if self.kind == "standard":
            center, scale = x.mean(axis=0), x.std(axis=0)
        else:  # minmax -> [0, 1]
            center, scale = x.min(axis=0), x.max(axis=0) - x.min(axis=0)

        degenerate = scale <= 1e-12
        self.n_degenerate_ = int(degenerate.sum())
        scale = np.where(degenerate, 1.0, scale)

        self.center_, self.scale_ = center, scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler.fit must be called before transform")
        z = (np.asarray(x, dtype=np.float64) - self.center_) / self.scale_
        if self.clip_sigma is not None:
            z = np.clip(z, -self.clip_sigma, self.clip_sigma)
        return z.astype(np.float32)


class SlidingWindowDataset(Dataset):
    """Sliding windows over a flat ``(T, F)`` series.

    Yields ``(window, label_window)`` with shapes ``(win_size, F)`` and
    ``(win_size,)``. Labels are all-zero for training splits, which have no
    anomaly annotations in this benchmark.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None,
        win_size: int,
        stride: int,
        valid_starts: np.ndarray | None = None,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"expected (T, F) array, got shape {data.shape}")
        if len(data) < win_size:
            raise ValueError(
                f"series of length {len(data)} is shorter than win_size {win_size}"
            )

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.labels = (
            np.zeros(len(data), dtype=np.float32)
            if labels is None
            else np.asarray(labels, dtype=np.float32)
        )
        if len(self.labels) != len(self.data):
            raise ValueError(
                f"labels length {len(self.labels)} != data length {len(self.data)}"
            )

        self.win_size, self.stride = win_size, stride
        self.starts = (
            np.arange(0, len(data) - win_size + 1, stride)
            if valid_starts is None
            else np.asarray(valid_starts, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        e = s + self.win_size
        return (
            torch.from_numpy(self.data[s:e]),
            torch.from_numpy(self.labels[s:e]),
        )

    @property
    def n_features(self) -> int:
        return self.data.shape[1]


@dataclass
class Splits:
    """The three splits plus the metadata Phases 3-5 need."""

    train: SlidingWindowDataset
    val: SlidingWindowDataset
    test: SlidingWindowDataset
    scaler: Scaler
    n_features: int
    test_labels: np.ndarray  # dense per-timestep labels over the full test series
    test_bounds: np.ndarray | None  # channel boundaries, concat mode only
    n_boundary_windows: int


def _split_train_val(
    train_raw: np.ndarray,
    bounds: np.ndarray | None,
    val_fraction: float,
    win_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last ``val_fraction`` of time, chronologically.

    In concat mode this must be done *per channel*. Slicing the tail off the
    spliced array does not hold out later time -- it holds out the last ~10
    channels outright, so train and val end up on disjoint spacecraft
    subsystems and the val loss measures transfer to unseen hardware rather
    than fit. (Observed as a 23x train/val reconstruction gap before this
    fix.) Splitting inside each channel keeps every channel in both halves.

    Note the reference implementation sidesteps this by validating on the test
    set, which leaks labels; we do not copy that.
    """
    if bounds is None:
        n_val = int(len(train_raw) * val_fraction)
        return train_raw[: len(train_raw) - n_val], train_raw[-n_val:]

    tr_parts, va_parts = [], []
    too_short = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = train_raw[int(a) : int(b)]
        n_val = int(len(seg) * val_fraction)
        if len(seg) - n_val < win_size or n_val < win_size:
            too_short += 1
        tr_parts.append(seg[: len(seg) - n_val])
        va_parts.append(seg[len(seg) - n_val :])

    if too_short:
        import warnings

        warnings.warn(
            f"{too_short} channel(s) yield a train or val segment shorter than "
            f"win_size={win_size}; their data appears only inside windows that "
            f"straddle into an adjacent channel. Affects SMAP D-12 (62 val "
            f"steps) and MSL T-9 (87) at the defaults -- ~0.2% of val.",
            stacklevel=2,
        )

    return np.concatenate(tr_parts), np.concatenate(va_parts)


def build_splits(
    root: Path,
    spacecraft: str,
    cfg: WindowConfig | None = None,
    mode: str = "concat",
    chan_id: str | None = None,
) -> Splits:
    """Build train/val/test windowed datasets.

    ``mode="concat"`` reproduces the literature setup (all channels spliced).
    ``mode="entity"`` uses a single channel, named by ``chan_id``.
    """
    cfg = cfg or WindowConfig()
    root = Path(root)

    if mode == "concat":
        train_raw, test_raw, test_labels, train_bounds, test_bounds = build_concat(
            root, spacecraft
        )
    elif mode == "entity":
        if chan_id is None:
            raise ValueError("mode='entity' requires chan_id")
        ch = load_channel(root, chan_id, spacecraft)
        train_raw, test_raw, test_labels = ch.train, ch.test, ch.test_labels()
        train_bounds = test_bounds = None
    else:
        raise ValueError(f"mode must be 'concat' or 'entity', got {mode!r}")

    # Chronological validation split, per channel in concat mode. Never random:
    # a random split over a time series puts near-identical neighbouring
    # windows on both sides and leaks.
    tr_raw, va_raw = _split_train_val(
        train_raw, train_bounds, cfg.val_fraction, cfg.win_size
    )

    scaler = Scaler(cfg.normalize, cfg.clip_sigma).fit(tr_raw)
    tr, va, te = (scaler.transform(a) for a in (tr_raw, va_raw, test_raw))

    valid_starts = None
    n_boundary = 0
    if mode == "concat" and test_bounds is not None:
        stride = cfg.stride_for("test")
        starts = np.arange(0, len(te) - cfg.win_size + 1, stride)
        interior = test_bounds[1:-1]
        straddles = (
            (interior[None, :] > starts[:, None])
            & (interior[None, :] < starts[:, None] + cfg.win_size)
        ).any(axis=1)
        n_boundary = int(straddles.sum())
        if cfg.drop_boundary_windows:
            valid_starts = starts[~straddles]

    return Splits(
        train=SlidingWindowDataset(
            tr, None, cfg.win_size, cfg.stride_for("train")
        ),
        val=SlidingWindowDataset(va, None, cfg.win_size, cfg.stride_for("train")),
        test=SlidingWindowDataset(
            te, test_labels, cfg.win_size, cfg.stride_for("test"), valid_starts
        ),
        scaler=scaler,
        n_features=tr.shape[1],
        test_labels=np.asarray(test_labels),
        test_bounds=test_bounds,
        n_boundary_windows=n_boundary,
    )


def make_loaders(
    splits: Splits, batch_size: int = 64, num_workers: int = 0
) -> dict[str, DataLoader]:
    """DataLoaders with the shuffling convention each split needs.

    Test is never shuffled: Phases 4 and 5 replay it as a time-ordered stream.
    """
    common = dict(batch_size=batch_size, num_workers=num_workers, drop_last=False)
    return {
        "train": DataLoader(splits.train, shuffle=True, **common),
        "val": DataLoader(splits.val, shuffle=False, **common),
        "test": DataLoader(splits.test, shuffle=False, **common),
    }
