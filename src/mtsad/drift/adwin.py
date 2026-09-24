"""ADWIN (Bifet & Gavalda, SDM 2007), adaptive windowing.

Keeps a window of recent values and, whenever the window can be split into an
older part and a newer part whose means differ by more than a Hoeffding-style
bound, drops the older part and reports drift. The window therefore sizes
itself: long while the stream is stationary, short right after a change.

Buckets are stored as an exponential histogram so memory is O(log n) rather
than O(n): at most ``max_buckets`` buckets of each size 2^k, oldest first.
Each bucket carries ``(size, total, total_sq)`` so a merge is an add and a
drop is a subtract.

The cut test, with variance (Section 3.2 of the paper):

    m       = 1 / (1/n0 + 1/n1)            harmonic mean of the two halves
    delta'  = delta / n
    eps_cut = sqrt((2/m) * var_W * ln(2/delta')) + (2/(3m)) * ln(2/delta')

and we cut when ``|mean0 - mean1| > eps_cut``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import DriftDetector, config_as_dict


@dataclass
class ADWINConfig:
    """Exposed for Phase 5's sensitivity controls."""

    delta: float = 0.002
    """Confidence. Smaller = more conservative = fewer false positives."""

    max_buckets: int = 5
    """Buckets per exponential-histogram row. Higher = finer cut points."""

    min_window_length: int = 5
    """Both halves must hold at least this many points to be testable."""

    grace_period: int = 10
    """Ignore the first few points while the window fills."""

    clock: int = 32
    """Only test for a cut every `clock` inserts.

    The standard value from the paper's implementation. It bounds detection
    delay from below -- an alarm can be up to ``clock - 1`` steps late purely
    from this quantisation, which is why reported delays should be read
    against it. Set to 1 for exact timing at higher cost.
    """

    cooldown: int = 0
    """Suppress further alarms for this many steps after one fires.

    ADWIN drops one bucket per cut and keeps cutting on subsequent steps, so a
    single conceptual change surfaces as a burst of alarms -- 101 of them in
    1700 steps on a single SMAP channel at delta=0.002. That is correct
    behaviour for the algorithm and useless on a dashboard.

    Defaults to 0, which is the per-crossing behaviour the Phase 4 false
    positive rates were measured under; leaving it there keeps those numbers
    comparable. The dashboard raises it to group a burst into one event.
    """


class ADWIN(DriftDetector):
    def __init__(self, config: ADWINConfig | None = None, **overrides) -> None:
        self.cfg = config or ADWINConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"unknown ADWIN option {key!r}")
            setattr(self.cfg, key, value)
        if not 0.0 < self.cfg.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.cfg.delta}")
        if self.cfg.clock < 1:
            raise ValueError("clock must be >= 1")
        self.reset()

    # ------------------------------------------------------------ state ----

    def reset(self) -> None:
        # Oldest first. Each entry: [size, total, total_sq].
        self._buckets: list[list[float]] = []
        self._n = 0.0
        self._total = 0.0
        self._total_sq = 0.0
        self._n_seen = 0
        self._suppress_until = -1
        self._last_statistic = float("nan")

    @property
    def config(self) -> dict:
        return config_as_dict(self.cfg)

    @property
    def width(self) -> int:
        """Current adaptive window length."""
        return int(self._n)

    @property
    def mean(self) -> float:
        return self._total / self._n if self._n else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        var = self._total_sq / self._n - (self._total / self._n) ** 2
        return max(var, 0.0)  # guard float cancellation

    # ------------------------------------------------------------ update ---

    def update(self, value: float) -> bool:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"ADWIN received a non-finite value: {value}")

        self._insert(value)
        self._n_seen += 1

        if self._n_seen < self.cfg.grace_period:
            return False
        if self._n_seen % self.cfg.clock != 0:
            return False

        # The window still shrinks during the refractory period -- only the
        # alarm is suppressed, so adaptation is unaffected.
        fired = self._shrink()
        if not fired:
            return False
        if self._n_seen <= self._suppress_until:
            return False
        self._suppress_until = self._n_seen + self.cfg.cooldown
        return True

    def _insert(self, value: float) -> None:
        self._buckets.append([1.0, value, value * value])
        self._n += 1.0
        self._total += value
        self._total_sq += value * value
        self._compress()

    def _compress(self) -> None:
        """Restore the invariant: at most max_buckets buckets of each size."""
        size = 1.0
        while True:
            same = [i for i, b in enumerate(self._buckets) if b[0] == size]
            if len(same) <= self.cfg.max_buckets:
                break
            # Merge the two OLDEST buckets of this size; they are adjacent
            # because compression proceeds smallest-size first.
            i, j = same[0], same[1]
            a, b = self._buckets[i], self._buckets[j]
            merged = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
            self._buckets[i : j + 1] = [merged]
            size *= 2.0

    def _shrink(self) -> bool:
        """Drop the oldest bucket while any split shows a significant gap."""
        detected = False
        changed = True
        while changed and len(self._buckets) >= 2:
            changed = False
            n0 = 0.0
            s0 = 0.0
            for i in range(len(self._buckets) - 1):
                n0 += self._buckets[i][0]
                s0 += self._buckets[i][1]
                n1 = self._n - n0
                s1 = self._total - s0
                if (
                    n0 < self.cfg.min_window_length
                    or n1 < self.cfg.min_window_length
                ):
                    continue
                if self._cut(n0, s0, n1, s1):
                    dropped = self._buckets.pop(0)
                    self._n -= dropped[0]
                    self._total -= dropped[1]
                    self._total_sq -= dropped[2]
                    detected = True
                    changed = True
                    break
        return detected

    def _cut(self, n0: float, s0: float, n1: float, s1: float) -> bool:
        diff = abs(s0 / n0 - s1 / n1)
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        delta_prime = self.cfg.delta / max(self._n, 1.0)
        log_term = math.log(2.0 / delta_prime)
        eps = math.sqrt(2.0 / m * self.variance * log_term) + (
            2.0 / (3.0 * m) * log_term
        )
        self._last_statistic = diff - eps
        return diff > eps
