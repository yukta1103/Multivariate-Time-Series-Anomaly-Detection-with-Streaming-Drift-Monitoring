"""Replay state machine: position, detectors, threshold, recalibration.

Kept free of Streamlit so it can be unit tested. The dashboard holds one of
these in ``st.session_state`` and calls :meth:`StreamSession.advance`.

Recalibration is manual by design. Drift latches the flag; the operator
clicks, and only then is the threshold recomputed from the recent window.
Auto-recalibrating would hide the mechanism and, worse, would silently keep
chasing a degrading model's score distribution so the alarm never persists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..drift.adwin import ADWIN
from ..drift.kswin import KSWIN
from .data import StreamSlice


@dataclass
class DetectorSettings:
    """Live-adjustable knobs, mirroring the Phase 4 dataclass configs."""

    adwin_delta: float = 0.002
    adwin_clock: int = 1

    # ADWIN cuts one bucket per step after a change, so a single shift
    # surfaces as a burst (101 alarms in 1700 steps on SMAP A-1 at
    # delta=0.002). The refractory period groups a burst into one event so
    # the chart stays readable. Phase 4's measurements used 0.
    adwin_cooldown: int = 200

    ks_alpha: float = 1e-3
    ks_window: int = 300
    ks_stride: int = 25
    ks_cooldown: int = 200

    def key(self) -> tuple:
        """Identity for deciding whether detectors must be rebuilt."""
        return (self.adwin_delta, self.adwin_clock, self.adwin_cooldown,
                self.ks_alpha, self.ks_window, self.ks_stride,
                self.ks_cooldown)


@dataclass
class Alarm:
    index: int
    detector: str


@dataclass
class StreamSession:
    """Replays a slice, feeding both detectors and tracking the threshold."""

    stream: StreamSlice
    settings: DetectorSettings = field(default_factory=DetectorSettings)
    calibration_frac: float = 0.2
    threshold_quantile: float = 0.99

    position: int = 0
    threshold: float = 0.0
    needs_recalibration: bool = False
    alarms: list[Alarm] = field(default_factory=list)
    n_recalibrations: int = 0

    def __post_init__(self) -> None:
        self._build()

    # ------------------------------------------------------------ setup ----

    @property
    def calibration_size(self) -> int:
        n = max(int(len(self.stream) * self.calibration_frac), 2)
        return min(n, len(self.stream))

    def _build(self) -> None:
        """(Re)create detectors and reset the replay to the start."""
        s = self.settings
        calib = self.stream.scores[: self.calibration_size]
        self._adwin = ADWIN(delta=s.adwin_delta, clock=s.adwin_clock,
                            cooldown=s.adwin_cooldown)
        self._kswin = KSWIN(
            reference=calib, window_size=s.ks_window, alpha=s.ks_alpha,
            stride=s.ks_stride, cooldown=s.ks_cooldown,
        )
        self.position = 0
        self.alarms = []
        self.needs_recalibration = False
        self.n_recalibrations = 0
        self.threshold = float(np.quantile(calib, self.threshold_quantile))

    def apply_settings(self, settings: DetectorSettings) -> None:
        """Swap detector config. Replays from the start so the alarm history
        shown always corresponds to the settings currently displayed."""
        if settings.key() == self.settings.key():
            return
        self.settings = settings
        self._build()

    def reset(self) -> None:
        self._build()

    # ---------------------------------------------------------- playback ----

    @property
    def finished(self) -> bool:
        return self.position >= len(self.stream)

    def advance(self, n: int = 1) -> list[Alarm]:
        """Consume up to ``n`` timesteps. Returns alarms raised in this call."""
        raised: list[Alarm] = []
        for _ in range(n):
            if self.finished:
                break
            value = float(self.stream.scores[self.position])
            if self._adwin.update(value):
                raised.append(Alarm(self.position, "ADWIN"))
            if self._kswin.update(value):
                raised.append(Alarm(self.position, "KSWIN"))
            self.position += 1

        if raised:
            self.alarms.extend(raised)
            self.needs_recalibration = True
        return raised

    # ----------------------------------------------------- recalibration ----

    def recalibrate(self, window: int = 1000) -> float:
        """Operator action: reset the threshold from the recent window.

        Also re-points KSWIN's reference at that window, otherwise it keeps
        comparing against the stale baseline and re-fires immediately.
        """
        lo = max(0, self.position - window)
        recent = self.stream.scores[lo : max(self.position, lo + 2)]
        self.threshold = float(np.quantile(recent, self.threshold_quantile))
        self._kswin.set_reference(recent)
        self.needs_recalibration = False
        self.n_recalibrations += 1
        return self.threshold

    # ----------------------------------------------------------- views -----

    def visible(self, span: int = 2000) -> dict[str, np.ndarray]:
        """The trailing window the charts render."""
        lo = max(0, self.position - span)
        hi = self.position
        idx = np.arange(lo, hi)
        scores = self.stream.scores[lo:hi]
        return {
            "index": idx,
            "telemetry": self.stream.telemetry[lo:hi],
            "labels": self.stream.labels[lo:hi],
            "scores": scores,
            "flagged": idx[scores >= self.threshold] if len(idx) else idx,
        }

    def alarms_in(self, lo: int, hi: int) -> list[Alarm]:
        return [a for a in self.alarms if lo <= a.index < hi]

    @property
    def n_flagged(self) -> int:
        seen = self.stream.scores[: self.position]
        return int((seen >= self.threshold).sum()) if len(seen) else 0

    def counts(self) -> dict[str, int]:
        return {
            "ADWIN": sum(a.detector == "ADWIN" for a in self.alarms),
            "KSWIN": sum(a.detector == "KSWIN" for a in self.alarms),
        }

    @property
    def adwin_width(self) -> int:
        return self._adwin.width

    @property
    def kswin_p_value(self) -> float:
        return self._kswin.last_p_value
