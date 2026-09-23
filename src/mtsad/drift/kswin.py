"""Two-sample KS drift detection against a fixed reference window.

Complements ADWIN. ADWIN watches the *mean* and so is blind to a change that
preserves it (a variance-only shift, or a symmetric spread); the
Kolmogorov-Smirnov statistic compares whole empirical CDFs and catches those.
It is also the more interpretable of the two on a dashboard, since the
p-value and the reference window are directly displayable.

The reference is fixed by calibration -- normally the score distribution the
operating threshold was set on -- rather than rolling. A rolling reference
adapts to slow drift and then silently stops reporting it, which is precisely
the failure mode a "recalibrate your threshold" alarm exists to prevent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

from .base import DriftDetector, config_as_dict


@dataclass
class KSWINConfig:
    """Exposed for Phase 5's sensitivity controls."""

    window_size: int = 300
    """Recent scores compared against the reference."""

    reference_size: int = 1000
    """Calibration sample. Taken from the head of the stream if not supplied."""

    alpha: float = 0.001
    """KS p-value threshold. Smaller = more conservative."""

    stride: int = 25
    """Run the test every `stride` steps. Bounds detection delay from below
    and limits how many times the null is tested (each test is another
    chance at a false positive)."""

    cooldown: int = 0
    """Suppress further alarms for this many steps after one fires. 0 means
    report every crossing, which is what the false-positive measurement
    needs; the dashboard sets it higher to avoid alarm storms."""


class KSWIN(DriftDetector):
    def __init__(
        self,
        config: KSWINConfig | None = None,
        reference: np.ndarray | None = None,
        **overrides,
    ) -> None:
        self.cfg = config or KSWINConfig()
        for key, value in overrides.items():
            if not hasattr(self.cfg, key):
                raise TypeError(f"unknown KSWIN option {key!r}")
            setattr(self.cfg, key, value)
        if not 0.0 < self.cfg.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.cfg.alpha}")
        if self.cfg.window_size < 2 or self.cfg.reference_size < 2:
            raise ValueError("window_size and reference_size must be >= 2")
        if self.cfg.stride < 1:
            raise ValueError("stride must be >= 1")

        self._fixed_reference = (
            None if reference is None else np.asarray(reference, dtype=np.float64)
        )
        self.reset()

    # ------------------------------------------------------------ state ----

    def reset(self) -> None:
        self._recent: deque[float] = deque(maxlen=self.cfg.window_size)
        self._reference: np.ndarray | None = self._fixed_reference
        self._calibrating: list[float] = []
        self._n_seen = 0
        self._suppress_until = -1
        self._last_statistic = float("nan")
        self._last_p_value = float("nan")

    @property
    def config(self) -> dict:
        return config_as_dict(self.cfg)

    @property
    def last_p_value(self) -> float:
        return self._last_p_value

    @property
    def is_calibrated(self) -> bool:
        return self._reference is not None

    def set_reference(self, reference: np.ndarray) -> None:
        """Recalibrate against a new baseline (e.g. after an operator ack)."""
        self._reference = np.asarray(reference, dtype=np.float64)

    # ------------------------------------------------------------ update ---

    def update(self, value: float) -> bool:
        value = float(value)
        self._n_seen += 1

        # Fill the reference from the head of the stream if none was given.
        if self._reference is None:
            self._calibrating.append(value)
            if len(self._calibrating) >= self.cfg.reference_size:
                self._reference = np.asarray(self._calibrating, dtype=np.float64)
                self._calibrating = []
            return False

        self._recent.append(value)
        if len(self._recent) < self.cfg.window_size:
            return False
        if self._n_seen % self.cfg.stride != 0:
            return False
        if self._n_seen <= self._suppress_until:
            return False

        stat, p_value = ks_2samp(self._reference, np.fromiter(
            self._recent, dtype=np.float64, count=len(self._recent)
        ))
        self._last_statistic = float(stat)
        self._last_p_value = float(p_value)

        if p_value < self.cfg.alpha:
            self._suppress_until = self._n_seen + self.cfg.cooldown
            return True
        return False
