"""Prove our raw-channel concatenation reproduces the published benchmark.

Anomaly Transformer, OmniAnomaly and friends all train on pre-concatenated
``SMAP_train.npy`` / ``MSL_test.npy`` / ... arrays that circulate as opaque
binaries. We rebuild them from NASA's raw per-channel release instead. This
module checks the rebuild is exact, so any later result is attributable to the
model rather than to a preprocessing discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .telemanom import build_concat


@dataclass
class ArrayCheck:
    name: str
    ours: tuple[int, ...]
    reference: tuple[int, ...]
    exact: bool
    max_abs_diff: float | None

    @property
    def ok(self) -> bool:
        return self.exact

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        shape = (
            f"{self.ours}"
            if self.ours == self.reference
            else f"{self.ours} vs ref {self.reference}"
        )
        diff = "" if self.max_abs_diff is None else f"  max|diff|={self.max_abs_diff:g}"
        return f"  [{mark}] {self.name:<18} {shape}{diff}"


def verify_spacecraft(root: Path, spacecraft: str) -> list[ArrayCheck]:
    """Compare our rebuild against the reference arrays for one spacecraft."""
    root = Path(root)
    ref_dir = root / "reference"

    train, test, labels, _, _ = build_concat(root, spacecraft)
    pairs = {
        "train": (train, ref_dir / f"{spacecraft}_train.npy"),
        "test": (test, ref_dir / f"{spacecraft}_test.npy"),
        "test_label": (labels, ref_dir / f"{spacecraft}_test_label.npy"),
    }

    checks: list[ArrayCheck] = []
    for name, (ours, ref_path) in pairs.items():
        if not ref_path.exists():
            raise FileNotFoundError(
                f"missing reference array {ref_path}; run "
                "scripts/01_download_data.py without --skip-reference"
            )
        ref = np.load(ref_path)

        if ours.shape != ref.shape:
            checks.append(
                ArrayCheck(name, ours.shape, ref.shape, exact=False, max_abs_diff=None)
            )
            continue

        diff = float(np.max(np.abs(ours.astype(np.float64) - ref.astype(np.float64))))
        checks.append(
            ArrayCheck(name, ours.shape, ref.shape, exact=diff == 0.0, max_abs_diff=diff)
        )

    return checks


def verify_all(root: Path, spacecraft: tuple[str, ...] = ("SMAP", "MSL")):
    """Run verification for each spacecraft; returns ``(all_ok, report_lines)``."""
    lines: list[str] = []
    all_ok = True
    for sc in spacecraft:
        lines.append(f"{sc}:")
        for check in verify_spacecraft(root, sc):
            lines.append(str(check))
            all_ok &= check.ok
    return all_ok, lines
