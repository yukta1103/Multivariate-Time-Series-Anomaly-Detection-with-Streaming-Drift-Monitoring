"""Fetch the NASA SMAP/MSL telemetry benchmark.

The dataset originates from NASA JPL's ``telemanom`` release (Hundman et al.,
KDD 2018). The original S3 bucket (``s3-us-west-2.amazonaws.com/telemanom``)
now returns 403 and NASA moved distribution to Kaggle, which requires an API
key. We therefore pull from two public, auth-free HuggingFace mirrors:

RAW (``lxr11111/telemanom``)
    The untouched per-channel arrays: 82 channels, each with a ``train`` and a
    ``test`` ``.npy`` of shape ``(n_timesteps, n_features)``. Column 0 is the
    telemetry value; the remaining columns are one-hot encoded commands sent to
    the spacecraft. This is what we build our loader on.

REFERENCE (``thuml/Time-Series-Library``)
    The pre-concatenated ``SMAP_train.npy`` / ``SMAP_test.npy`` / ``*_label.npy``
    arrays that Anomaly Transformer and OmniAnomaly actually train on. We do not
    model on these -- we use them to *verify* that our own concatenation of the
    raw channels reproduces the published benchmark exactly.
"""

from __future__ import annotations

import concurrent.futures as cf
import urllib.error
import urllib.request
from pathlib import Path

RAW_REPO = "lxr11111/telemanom"
REF_REPO = "thuml/Time-Series-Library"

_HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_API = "https://huggingface.co/api/datasets/{repo}"

# The concatenated arrays used by the published baselines.
REFERENCE_FILES = (
    "SMAP/SMAP_train.npy",
    "SMAP/SMAP_test.npy",
    "SMAP/SMAP_test_label.npy",
    "MSL/MSL_train.npy",
    "MSL/MSL_test.npy",
    "MSL/MSL_test_label.npy",
)


def _get(url: str, dest: Path, retries: int = 3) -> int:
    """Download ``url`` to ``dest``, skipping if already present. Returns bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mtsad/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                break

    raise RuntimeError(f"failed to download {url}: {last}")


def _list_repo_files(repo: str) -> list[str]:
    req = urllib.request.Request(
        _API.format(repo=repo), headers={"User-Agent": "mtsad/0.1"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        import json

        return [s["rfilename"] for s in json.load(r)["siblings"]]


def download_raw(root: Path, workers: int = 8) -> dict[str, int]:
    """Download the 82 raw per-channel train/test arrays + the label CSV."""
    files = _list_repo_files(RAW_REPO)

    # Only the .npy channel arrays; the mirror also carries parquet duplicates
    # and the original authors' trained .h5 models, which we do not need.
    channels = [
        f
        for f in files
        if f.endswith(".npy") and ("/data/train/" in f or "/data/test/" in f)
    ]

    jobs: list[tuple[str, Path]] = []
    for f in channels:
        split = "train" if "/train/" in f else "test"
        jobs.append((f, root / "raw" / split / Path(f).name))
    jobs.append(("labeled_anomalies.csv", root / "raw" / "labeled_anomalies.csv"))

    return _run(RAW_REPO, jobs, workers)


def download_reference(root: Path, workers: int = 6) -> dict[str, int]:
    """Download the pre-concatenated arrays used by the published baselines."""
    jobs = [(f, root / "reference" / Path(f).name) for f in REFERENCE_FILES]
    return _run(REF_REPO, jobs, workers)


def _run(repo: str, jobs: list[tuple[str, Path]], workers: int) -> dict[str, int]:
    # Keyed by "<split>/<file>" rather than bare filename: train/A-1.npy and
    # test/A-1.npy share a name and would otherwise collide, silently halving
    # both the file count and the reported byte total.
    sizes: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_get, _HF.format(repo=repo, path=src), dst): dst
            for src, dst in jobs
        }
        for fut in cf.as_completed(futs):
            dst = futs[fut]
            sizes[f"{dst.parent.name}/{dst.name}"] = fut.result()
    return sizes
