"""Common interface for drift detectors.

Everything is a dataclass config rather than a constructor argument so Phase
5's dashboard can rebuild a detector from live slider values without knowing
which concrete class it is holding.

A note on what these detectors actually do, which matters for how the
dashboard presents them: they monitor the distribution of *anomaly scores*.
They answer "has the score distribution moved?", not "is the model still
correct?". Those come apart badly when the model is weak -- see
``harness.blind_to_model_quality``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class DriftEvent:
    index: int              # stream position where the alarm fired
    statistic: float        # detector-specific magnitude
    detail: str = ""


class DriftDetector(ABC):
    """Streaming detector: feed one score at a time, get an alarm or not."""

    @abstractmethod
    def update(self, value: float) -> bool:
        """Consume one score. Returns True on the step an alarm fires."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state, keeping configuration."""

    @property
    @abstractmethod
    def config(self) -> dict:
        """Current tunables, for display and round-tripping in the UI."""

    @property
    def last_statistic(self) -> float:
        return getattr(self, "_last_statistic", float("nan"))

    def run(self, stream) -> list[DriftEvent]:
        """Replay a whole stream, collecting every alarm."""
        events: list[DriftEvent] = []
        for i, x in enumerate(stream):
            if self.update(float(x)):
                events.append(
                    DriftEvent(index=i, statistic=self.last_statistic,
                               detail=type(self).__name__)
                )
        return events


@dataclass
class RecalibrationState:
    """Wraps a detector with the 'threshold needs recalibrating' flag.

    Phase 5 consumes this: the banner reflects ``needs_recalibration``, and
    acknowledging it calls ``acknowledge`` rather than resetting the detector,
    so the alarm history survives for the audit trail.
    """

    detector: DriftDetector
    needs_recalibration: bool = False
    events: list[DriftEvent] = field(default_factory=list)
    n_seen: int = 0

    def update(self, value: float) -> bool:
        fired = self.detector.update(value)
        if fired:
            self.needs_recalibration = True
            self.events.append(
                DriftEvent(self.n_seen, self.detector.last_statistic,
                           type(self.detector).__name__)
            )
        self.n_seen += 1
        return fired

    def acknowledge(self) -> None:
        """Operator has recalibrated; clear the flag but keep the history."""
        self.needs_recalibration = False

    def summary(self) -> dict:
        return {
            "detector": type(self.detector).__name__,
            "config": self.detector.config,
            "n_seen": self.n_seen,
            "n_alarms": len(self.events),
            "needs_recalibration": self.needs_recalibration,
        }


def config_as_dict(cfg) -> dict:
    return asdict(cfg)
