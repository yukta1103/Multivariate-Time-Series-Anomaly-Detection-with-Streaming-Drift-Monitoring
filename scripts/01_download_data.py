"""Download the SMAP/MSL benchmark into ./data (~333 MB, gitignored).

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --skip-reference
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.download import download_raw, download_reference  # noqa: E402


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:,.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument(
        "--skip-reference",
        action="store_true",
        help="skip the pre-concatenated verification arrays",
    )
    args = ap.parse_args()
    root: Path = args.data_root

    print(f"data root: {root}")

    print("\n[1/2] raw per-channel telemetry (lxr11111/telemanom) ...")
    raw = download_raw(root)
    n_train = len(list((root / "raw" / "train").glob("*.npy")))
    n_test = len(list((root / "raw" / "test").glob("*.npy")))
    print(
        f"      {len(raw)} files, {_mb(sum(raw.values()))}"
        f"  ({n_train} train channels, {n_test} test channels)"
    )

    if args.skip_reference:
        print("\n[2/2] reference arrays: skipped")
    else:
        print("\n[2/2] reference concatenated arrays (thuml/Time-Series-Library) ...")
        ref = download_reference(root)
        print(f"      {len(ref)} files, {_mb(sum(ref.values()))}")
        for name in sorted(ref):
            print(f"        {name:<24} {_mb(ref[name]):>12}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
