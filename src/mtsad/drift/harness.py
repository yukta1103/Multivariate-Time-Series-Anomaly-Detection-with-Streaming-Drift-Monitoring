"""Measuring drift detectors honestly: delay, false positives, and blindness.

Three things get measured, because reporting only the first would be
misleading:

1. **Detection delay** -- steps from an injected shift to the first alarm.
2. **False positive rate** on a stream with no injected shift. A detector
   tuned until it always catches the injection is worthless if it also fires
   constantly on clean data, so these two are reported together.
3. **Blindness to model quality** -- whether the detector behaves any
   differently on scores from a model that discriminates anomalies well
   versus one that does not. It does not, and the dashboard has to say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .base import DriftDetector, DriftEvent

ShiftKind = Literal["offset", "scale", "variance", "splice"]


# --------------------------------------------------------------- injection ---


def inject_shift(
    scores: np.ndarray,
    start: int,
    kind: ShiftKind = "offset",
    magnitude: float = 1.0,
    donor: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return a copy of ``scores`` with a distribution shift from ``start`` on.

    ``offset``    add ``magnitude`` standard deviations to the tail.
    ``scale``     multiply deviations from the mean by ``magnitude``.
    ``variance``  keep the mean, inflate the spread by ``magnitude``.
    ``splice``    replace the tail with samples drawn from ``donor``, which
                  is the most realistic case: a genuinely different channel.
    """
    out = np.asarray(scores, dtype=np.float64).copy()
    if not 0 <= start < len(out):
        raise ValueError(f"start {start} outside stream of length {len(out)}")

    tail = out[start:]
    mu = float(out[:start].mean()) if start else float(out.mean())
    sigma = float(out[:start].std()) if start else float(out.std())
    sigma = sigma if sigma > 0 else 1.0

    if kind == "offset":
        out[start:] = tail + magnitude * sigma
    elif kind == "scale":
        out[start:] = mu + (tail - mu) * magnitude
    elif kind == "variance":
        centred = tail - tail.mean()
        out[start:] = tail.mean() + centred * magnitude
    elif kind == "splice":
        if donor is None:
            raise ValueError("kind='splice' requires a donor array")
        rng = np.random.default_rng(seed)
        out[start:] = rng.choice(
            np.asarray(donor, dtype=np.float64), size=len(tail), replace=True
        )
    else:
        raise ValueError(f"unknown shift kind {kind!r}")

    return out


# ------------------------------------------------------------ measurement ---


@dataclass
class DelayResult:
    detected: bool
    delay: int | None          # steps from shift onset to first alarm
    first_alarm: int | None
    shift_at: int
    n_alarms_before_shift: int
    n_alarms_after_shift: int
    events: list[DriftEvent] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.detected:
            return f"  NOT DETECTED (shift at {self.shift_at:,})"
        return (
            f"  detected at {self.first_alarm:,}  "
            f"delay {self.delay:,} steps  "
            f"({self.n_alarms_before_shift} pre-shift alarms)"
        )


@dataclass
class FalsePositiveResult:
    n_alarms: int
    n_steps: int
    rate_per_1000: float
    mean_gap: float | None

    def __str__(self) -> str:
        return (
            f"  {self.n_alarms} alarms over {self.n_steps:,} clean steps  "
            f"= {self.rate_per_1000:.3f} per 1000"
        )


def detection_delay(
    make_detector: Callable[[], DriftDetector],
    stream: np.ndarray,
    shift_at: int,
) -> DelayResult:
    """First alarm at or after ``shift_at``, and how late it was."""
    events = make_detector().run(stream)
    before = [e for e in events if e.index < shift_at]
    after = [e for e in events if e.index >= shift_at]
    first = after[0].index if after else None
    return DelayResult(
        detected=first is not None,
        delay=None if first is None else first - shift_at,
        first_alarm=first,
        shift_at=shift_at,
        n_alarms_before_shift=len(before),
        n_alarms_after_shift=len(after),
        events=events,
    )


def false_positive_rate(
    make_detector: Callable[[], DriftDetector], clean_stream: np.ndarray
) -> FalsePositiveResult:
    """Alarms on a stream with no injected shift. Every one is spurious.

    Caveat worth keeping in mind when reading the number: real telemetry is
    not stationary, so an alarm here is only *definitely* a false positive
    relative to the injection. Use ``synthetic_clean_stream`` for a stream
    that is stationary by construction.
    """
    events = make_detector().run(clean_stream)
    n = len(clean_stream)
    gaps = np.diff([e.index for e in events]) if len(events) > 1 else None
    return FalsePositiveResult(
        n_alarms=len(events),
        n_steps=n,
        rate_per_1000=1000.0 * len(events) / n if n else 0.0,
        mean_gap=float(gaps.mean()) if gaps is not None and len(gaps) else None,
    )


def synthetic_clean_stream(
    reference: np.ndarray, n: int, seed: int = 0
) -> np.ndarray:
    """I.i.d. resample of ``reference``: stationary by construction.

    On this stream every alarm is unambiguously a false positive, which makes
    it the right place to quote a calibrated false-positive rate. The real
    score stream is not stationary and conflates the two.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(reference, dtype=np.float64), size=n, replace=True)


# ------------------------------------------------- model-quality blindness ---


@dataclass
class BlindnessResult:
    """Does the detector behave differently on a good vs. useless model?"""

    label_a: str
    label_b: str
    alarms_a: int
    alarms_b: int
    roc_a: float
    roc_b: float

    @property
    def indistinguishable(self) -> bool:
        larger = max(self.alarms_a, self.alarms_b, 1)
        return abs(self.alarms_a - self.alarms_b) / larger < 0.5


def blind_to_model_quality(
    make_detector: Callable[[], DriftDetector],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    roc_a: float,
    roc_b: float,
    label_a: str = "real model",
    label_b: str = "random scores",
) -> BlindnessResult:
    """Run the same detector on a real model's scores and on noise.

    The detector monitors the score *distribution*; it has no access to
    labels and therefore no way to know whether those scores mean anything.
    Quantifying that is the point -- a dashboard that shows a confident
    "drift detected" banner while the underlying model is at chance is
    actively misleading, and Phase 5 needs to surface model quality
    separately rather than letting the drift banner imply it.
    """
    return BlindnessResult(
        label_a=label_a,
        label_b=label_b,
        alarms_a=len(make_detector().run(scores_a)),
        alarms_b=len(make_detector().run(scores_b)),
        roc_a=roc_a,
        roc_b=roc_b,
    )
