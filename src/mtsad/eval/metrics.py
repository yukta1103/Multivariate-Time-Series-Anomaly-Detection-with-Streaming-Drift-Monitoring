"""Detection metrics, with point adjustment kept strictly optional.

Point adjustment (Xu et al. 2018; Su et al. 2019) credits an entire ground
truth anomaly segment as detected if *any single timestep* inside it is
flagged. Essentially every SMAP/MSL result in the literature is reported this
way, so we compute it for comparability -- but Kim et al. (2022, "Towards a
Rigorous Evaluation of Time-series Anomaly Detection") showed it is close to
meaningless on this benchmark: a uniform random scorer reaches ~0.9 adjusted
F1. ``evaluate`` therefore always returns the raw number alongside it, and
``random_baseline`` exists to quantify the inflation on this exact test set.

AUC is deliberately computed *before* any adjustment, on the continuous
scores. Point adjustment rewrites binary predictions at a chosen threshold;
ROC/PR AUC integrate over all thresholds and have no binary predictions to
rewrite. Applying it would not be meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def segments(y: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s in a binary vector, as half-open ``[start, end)``."""
    y = np.asarray(y).astype(np.int8)
    d = np.diff(np.concatenate([[0], y, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Expand each detected ground-truth segment to fully detected.

    If any timestep inside a true anomaly segment is predicted positive, every
    timestep in that segment becomes positive. False positives outside true
    segments are left untouched.
    """
    adjusted = np.asarray(y_pred).astype(np.int8).copy()
    for start, end in segments(y_true):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    n_predicted: int

    def row(self, label: str) -> str:
        return (
            f"  {label:<26} P {self.precision:.4f}  R {self.recall:.4f}  "
            f"F1 {self.f1:.4f}"
        )


@dataclass
class EvalResult:
    """Everything needed to report honestly. ``raw`` is the headline."""

    name: str
    raw: ThresholdResult
    adjusted: ThresholdResult
    roc_auc: float          # pre-adjustment, continuous scores
    pr_auc: float           # pre-adjustment, continuous scores
    n_points: int
    n_segments: int
    positive_rate: float
    extra: dict = field(default_factory=dict)

    @property
    def inflation(self) -> float:
        """How much point adjustment multiplies F1 by."""
        return self.adjusted.f1 / self.raw.f1 if self.raw.f1 else float("inf")


def _candidate_thresholds(scores: np.ndarray, n: int) -> np.ndarray:
    """Quantile-spaced candidates; sweeping every unique value is wasteful."""
    qs = np.linspace(0.0, 1.0, n)
    return np.unique(np.quantile(scores, qs))


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    adjust: bool,
    n_thresholds: int = 1000,
) -> ThresholdResult:
    """Sweep thresholds and keep the best F1.

    This is an *oracle* threshold -- it is chosen using the test labels. That
    is the convention in this literature and it is an upper bound, not a
    deployable number. ``threshold_at_quantile`` gives the deployable one.
    """
    best = ThresholdResult(float("nan"), 0.0, 0.0, -1.0, 0)
    for t in _candidate_thresholds(scores, n_thresholds):
        # ">=" not ">": with ">", the lowest candidate (the minimum score)
        # excludes every point tied at the minimum, so the sweep can never
        # reach the all-positive prediction. On SMAP 70% of the Anomaly
        # Transformer's scores tied at exactly 0.0, which capped recall at
        # 0.33 and let the trivial baseline "beat" the model.
        pred = (scores >= t).astype(np.int8)
        if adjust:
            pred = point_adjust(y_true, pred)
        p, r, f1 = _prf(y_true, pred)
        if f1 > best.f1:
            best = ThresholdResult(float(t), p, r, f1, int(pred.sum()))
    return best


def threshold_at_quantile(
    reference_scores: np.ndarray, quantile: float
) -> float:
    """Deployable threshold from scores on data with no labels.

    Used by Phase 4/5, where a threshold has to exist before any label does.
    """
    return float(np.quantile(reference_scores, quantile))


def evaluate_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[ThresholdResult, ThresholdResult]:
    """Raw and adjusted results at one fixed threshold."""
    pred = (scores >= threshold).astype(np.int8)  # matches best_threshold
    p, r, f1 = _prf(y_true, pred)
    raw = ThresholdResult(threshold, p, r, f1, int(pred.sum()))

    adj_pred = point_adjust(y_true, pred)
    p, r, f1 = _prf(y_true, adj_pred)
    adjusted = ThresholdResult(threshold, p, r, f1, int(adj_pred.sum()))
    return raw, adjusted


def evaluate(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 1000,
) -> EvalResult:
    """Full evaluation: best raw F1, best adjusted F1, and pre-adjustment AUC."""
    y_true = np.asarray(y_true).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(
            f"shape mismatch: labels {y_true.shape} vs scores {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}: scores contain non-finite values")

    # AUC on the raw continuous scores -- no threshold, no adjustment.
    roc = float(roc_auc_score(y_true, scores))
    pr = float(average_precision_score(y_true, scores))

    return EvalResult(
        name=name,
        raw=best_threshold(y_true, scores, adjust=False, n_thresholds=n_thresholds),
        adjusted=best_threshold(y_true, scores, adjust=True, n_thresholds=n_thresholds),
        roc_auc=roc,
        pr_auc=pr,
        n_points=int(y_true.size),
        n_segments=len(segments(y_true)),
        positive_rate=float(y_true.mean()),
    )


def random_baseline(
    y_true: np.ndarray, seed: int = 0, n_thresholds: int = 1000
) -> EvalResult:
    """Uniform random scores. The control that exposes adjustment inflation.

    A scorer with zero information should land at ROC-AUC 0.5 and raw F1 near
    the all-positive floor ``2p / (1 + p)``. Whatever adjusted F1 it reaches is
    pure artifact of the metric.
    """
    rng = np.random.default_rng(seed)
    scores = rng.random(len(y_true))
    result = evaluate("random (uniform)", y_true, scores, n_thresholds)
    p = float(np.mean(y_true))
    result.extra["all_positive_f1"] = 2 * p / (1 + p)
    return result


def all_positive_baseline(y_true: np.ndarray) -> EvalResult:
    """Flag every timestep. Raw F1 is exactly ``2p / (1 + p)``; adjusted is 1.0."""
    y_true = np.asarray(y_true).astype(np.int8)
    scores = np.ones(len(y_true), dtype=np.float64)
    pred = np.ones(len(y_true), dtype=np.int8)
    p, r, f1 = _prf(y_true, pred)
    raw = ThresholdResult(0.0, p, r, f1, int(pred.sum()))
    p2, r2, f2 = _prf(y_true, point_adjust(y_true, pred))
    return EvalResult(
        name="all-positive (trivial)",
        raw=raw,
        adjusted=ThresholdResult(0.0, p2, r2, f2, len(y_true)),
        roc_auc=0.5,
        pr_auc=float(np.mean(y_true)),
        n_points=int(y_true.size),
        n_segments=len(segments(y_true)),
        positive_rate=float(np.mean(y_true)),
    )
