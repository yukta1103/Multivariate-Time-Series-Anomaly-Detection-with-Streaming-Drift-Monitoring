"""Phase 1 sanity check: verify the rebuild, print the stats, plot the channels.

Usage:
    python scripts/02_inspect_and_plot.py
    python scripts/02_inspect_and_plot.py --spacecraft MSL --channels M-6 M-7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.data.telemanom import (  # noqa: E402
    EXPECTED,
    channel_ids,
    load_channel,
    load_labels,
)
from mtsad.data.verify import verify_all  # noqa: E402

FIGDIR = REPO / "reports" / "figures"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def report_verification(root: Path) -> bool:
    section("1. Rebuild verification vs. published arrays")
    ok, lines = verify_all(root)
    print("\n".join(lines))
    print(f"\n  -> rebuild is {'bit-exact' if ok else 'NOT exact'}")
    return ok


def report_overview(root: Path) -> None:
    section("2. Dataset overview")
    df = load_labels(root)
    print(f"  labeled_anomalies.csv: {len(df)} rows, "
          f"{df['chan_id'].duplicated().sum()} duplicate chan_id")

    for sc in ("SMAP", "MSL"):
        ids = channel_ids(root, sc)
        sub = df[df["spacecraft"] == sc].drop_duplicates("chan_id")
        spans = [s for seqs in sub["anomaly_sequences"] for s in seqs]
        lens = [b - a + 1 for a, b in spans]
        classes = [c.strip() for cs in sub["class"] for c in cs]
        exp = EXPECTED[sc]
        print(
            f"\n  {sc}: {len(ids)} channels used, {exp['features']} features "
            f"(1 telemetry value + {exp['features'] - 1} one-hot commands)"
            f"\n      train {exp['train']:>7,} steps   test {exp['test']:>7,} steps"
            f"\n      {len(spans)} labelled anomaly spans, "
            f"median length {int(np.median(lens)):,} steps "
            f"(min {min(lens):,}, max {max(lens):,})"
            f"\n      class mix: "
            + ", ".join(f"{c}={classes.count(c)}" for c in sorted(set(classes)))
        )


def report_splits(root: Path, spacecraft: str) -> None:
    section(f"3. Windowed splits ({spacecraft}, concat mode)")
    cfg = WindowConfig()
    sp = build_splits(root, spacecraft, cfg)

    pos = float(sp.test_labels.mean())
    print(
        f"  win_size={cfg.win_size}  train_stride={cfg.train_stride}  "
        f"test_stride={cfg.stride_for('test')}  normalize={cfg.normalize}"
        f"\n\n  train windows {len(sp.train):>8,}"
        f"\n  val   windows {len(sp.val):>8,}   (chronological last "
        f"{cfg.val_fraction:.0%} of train)"
        f"\n  test  windows {len(sp.test):>8,}   (non-overlapping)"
        f"\n  features      {sp.n_features:>8,}"
        f"\n\n  anomalous test timesteps: {int(sp.test_labels.sum()):,} "
        f"/ {len(sp.test_labels):,}  ({pos:.2%})"
        f"\n  zero-variance features clamped by scaler: {sp.scaler.n_degenerate_}"
        f" / {sp.n_features}"
        f"\n  windows straddling a channel boundary: {sp.n_boundary_windows:,}"
        f" / {len(sp.test):,}  ({sp.n_boundary_windows / len(sp.test):.2%})"
    )

    x, y = sp.test[0]
    print(f"\n  sample batch item: x={tuple(x.shape)} {x.dtype}, "
          f"y={tuple(y.shape)} {y.dtype}")
    print(f"  train window value range: "
          f"[{sp.train.data.min():.3f}, {sp.train.data.max():.3f}]")

    # A leak check worth keeping: val must be normalized by train statistics,
    # so its mean should be near zero but not exactly zero.
    vm = float(np.abs(sp.val.data.mean()))
    print(f"  |mean| of val after train-fit scaling: {vm:.4f} "
          f"({'ok' if vm < 0.5 else 'SUSPICIOUS'})")


def plot_channels(root: Path, spacecraft: str, chans: list[str]) -> Path:
    section(f"4. Sanity plots ({spacecraft}: {', '.join(chans)})")
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(chans), 1, figsize=(13, 2.6 * len(chans)), squeeze=False
    )
    for ax, cid in zip(axes[:, 0], chans):
        ch = load_channel(root, cid, spacecraft)
        value = ch.test[:, 0]  # column 0 is the monitored telemetry value
        ax.plot(value, lw=0.6, color="#2b6cb0", label="telemetry (col 0)")

        for k, (a, b) in enumerate(ch.anomaly_sequences):
            ax.axvspan(
                a, b, color="#e53e3e", alpha=0.28,
                label="labelled anomaly" if k == 0 else None,
            )

        n_cmd = int((ch.test[:, 1:] != 0).any(axis=0).sum())
        ax.set_title(
            f"{cid}  -  test {len(ch.test):,} steps, "
            f"{len(ch.anomaly_sequences)} anomaly span(s), "
            f"{n_cmd}/{ch.n_features - 1} active command channels",
            fontsize=9, loc="left",
        )
        ax.set_xlim(0, len(value))
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.grid(alpha=0.2)

    axes[-1, 0].set_xlabel("test timestep")
    fig.suptitle(
        f"{spacecraft} telemetry with labelled anomaly regions", fontsize=12, y=0.999
    )
    fig.tight_layout()

    out = FIGDIR / f"phase1_{spacecraft.lower()}_channels.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")
    return out


def plot_distribution(root: Path, spacecraft: str) -> Path:
    """Train vs. test distribution of the telemetry value, concatenated.

    Phase 4 hangs off this: if train and test already differ, some of what a
    drift detector fires on is baked into the benchmark rather than injected.
    """
    from mtsad.data.telemanom import build_concat

    train, test, labels, _, _ = build_concat(root, spacecraft)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    axes[0].hist(train[:, 0], bins=120, alpha=0.65, density=True, label="train",
                 color="#2b6cb0")
    axes[0].hist(test[:, 0], bins=120, alpha=0.65, density=True, label="test",
                 color="#dd6b20")
    axes[0].set_title("telemetry value: train vs test", fontsize=10, loc="left")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    normal, anom = test[labels == 0, 0], test[labels == 1, 0]
    axes[1].hist(normal, bins=120, alpha=0.65, density=True, label="normal",
                 color="#2f855a")
    axes[1].hist(anom, bins=120, alpha=0.65, density=True, label="anomalous",
                 color="#e53e3e")
    axes[1].set_title("test: normal vs anomalous", fontsize=10, loc="left")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"{spacecraft} value distributions", fontsize=12)
    fig.tight_layout()

    out = FIGDIR / f"phase1_{spacecraft.lower()}_distributions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument("--spacecraft", default="SMAP", choices=["SMAP", "MSL"])
    ap.add_argument("--channels", nargs="*", default=None)
    args = ap.parse_args()
    root: Path = args.data_root

    ok = report_verification(root)
    report_overview(root)
    report_splits(root, args.spacecraft)

    chans = args.channels
    if not chans:
        # Pick channels that actually have anomalies, for a readable plot.
        df = load_labels(root)
        sub = df[df["spacecraft"] == args.spacecraft].drop_duplicates("chan_id")
        sub = sub[sub["chan_id"].isin(channel_ids(root, args.spacecraft))]
        sub = sub.assign(n=sub["anomaly_sequences"].str.len())
        chans = sub.sort_values("n", ascending=False)["chan_id"].head(3).tolist()

    plot_channels(root, args.spacecraft, chans)
    plot_distribution(root, args.spacecraft)

    print(f"\n{'=' * 72}\nPhase 1 complete. Rebuild exact: {ok}\n{'=' * 72}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
