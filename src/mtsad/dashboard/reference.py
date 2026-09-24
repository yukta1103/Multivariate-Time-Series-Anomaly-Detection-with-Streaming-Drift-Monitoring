"""Measured numbers from Phases 3 and 4, for display in the dashboard.

Everything here is read from the committed ``reports/*.json`` rather than
retyped, so a slider's "expected false positive rate" is the figure actually
measured in Phase 4 and cannot drift out of sync with it.

The sliders snap to the swept values for the same reason: every reachable
position has a real measurement behind it, with no interpolation implying
precision we did not measure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MODEL_KEYS = {"at": "Anomaly Transformer", "ae": "LSTM-AE baseline"}
MODEL_LABELS = {"at": "Anomaly Transformer", "ae": "LSTM-AE baseline"}


@dataclass(frozen=True)
class ModelQuality:
    """Phase 3 test-set performance for one model on one spacecraft."""

    model: str
    spacecraft: str
    roc_auc: float
    pr_auc: float
    raw_f1: float
    adjusted_f1: float
    all_positive_f1: float

    @property
    def beats_chance(self) -> bool:
        return self.roc_auc > 0.5

    @property
    def beats_trivial(self) -> bool:
        return self.raw_f1 > self.all_positive_f1

    @property
    def verdict(self) -> str:
        if self.roc_auc < 0.5:
            return "below chance"
        if self.roc_auc < 0.55:
            return "barely above chance"
        if self.roc_auc < 0.7:
            return "weak but real signal"
        return "usable signal"

    @property
    def severity(self) -> str:
        """Maps to the Streamlit callout type."""
        return "error" if self.roc_auc < 0.5 else (
            "warning" if self.roc_auc < 0.6 else "info"
        )


@dataclass(frozen=True)
class SweepPoint:
    value: float
    fp_per_1000: float
    delay: int | None

    @property
    def label(self) -> str:
        delay = "missed" if self.delay is None else f"~{self.delay} step delay"
        return f"≈{self.fp_per_1000:.2f} FP/1k · {delay}"


@lru_cache(maxsize=8)
def _phase3(reports: str, spacecraft: str) -> dict:
    path = Path(reports) / f"phase3_{spacecraft.lower()}.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def _phase4(reports: str) -> dict:
    return json.loads((Path(reports) / "phase4_drift.json").read_text())


def model_quality(reports: Path, spacecraft: str, model: str) -> ModelQuality:
    data = _phase3(str(reports), spacecraft)
    rows = {r["name"]: r for r in data["results"]}
    row = rows[MODEL_KEYS[model]]
    trivial = rows["all-positive (trivial)"]
    return ModelQuality(
        model=model,
        spacecraft=spacecraft,
        roc_auc=row["roc_auc_pre_adjustment"],
        pr_auc=row["pr_auc_pre_adjustment"],
        raw_f1=row["raw"]["f1"],
        adjusted_f1=row["point_adjusted"]["f1"],
        all_positive_f1=trivial["raw"]["f1"],
    )


def sweep_points(reports: Path, spacecraft: str, detector: str) -> list[SweepPoint]:
    """Measured (setting -> false positives, delay) pairs, ascending by value."""
    entry = _phase4(str(reports))["spacecraft"][spacecraft]
    if "sweep" not in entry:
        raise KeyError(
            "phase4_drift.json has no sweep block; rerun "
            "scripts/06_drift.py --sweep"
        )
    table = entry["sweep"][detector]
    points = [
        SweepPoint(float(k), v["fp_per_1000"], v["delay"])
        for k, v in table.items()
    ]
    return sorted(points, key=lambda p: p.value)


def nearest_point(points: list[SweepPoint], value: float) -> SweepPoint:
    return min(points, key=lambda p: abs(p.value - value))


def raw_stream_alarm_rate(reports: Path, spacecraft: str, detector: str) -> float:
    """Alarms per 1000 steps on the spliced stream, for the advanced-view note."""
    entry = _phase4(str(reports))["spacecraft"][spacecraft]
    return entry["raw_stream_alarms"][detector]["per_1000"]
