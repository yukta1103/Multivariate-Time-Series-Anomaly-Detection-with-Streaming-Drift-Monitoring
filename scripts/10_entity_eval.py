"""Per-channel (entity) evaluation, which the concatenated protocol hides.

Two questions, deliberately separated:

A. **Concat-trained, per-channel scored.** Take the existing concatenated
   run and compute ROC-AUC *within each channel* instead of pooling. If
   per-channel AUC is much better than the pooled number, then pooling is
   what destroys the signal -- scores are not comparable across channels, so
   a global threshold ranks channel identity rather than anomalousness. No
   retraining; costs seconds.

B. **Entity-trained, entity-scored.** Train a separate model per channel.
   The methodologically clean setup the repo supports but never reported.

A and B answer different things: A isolates the *evaluation* protocol, B
isolates the *training* protocol.

Usage:
    python scripts/10_entity_eval.py --skip-training      # A only
    python scripts/10_entity_eval.py --n-channels 5       # A and B
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.dashboard.data import DashboardBundle  # noqa: E402
from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.eval.metrics import evaluate  # noqa: E402
from mtsad.eval.scoring import align_labels, score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.training import TrainConfig, train_model  # noqa: E402


def pooled_roc(sc: str) -> float:
    d = json.loads((REPO / f"reports/phase3_{sc.lower()}.json").read_text())
    return next(r["roc_auc_pre_adjustment"] for r in d["results"]
                if r["name"] == "Anomaly Transformer")


def part_a(sc: str) -> dict:
    """Per-channel AUC from the concat-trained scores."""
    bundle = DashboardBundle.load(
        REPO / "data" / "processed" / f"dashboard_{sc.lower()}.npz"
    )
    rows = []
    for cid in bundle.channel_ids:
        s = bundle.slice_for(cid, "at")
        y = np.asarray(s.labels)
        if len(np.unique(y)) < 2:
            continue  # AUC undefined with one class
        rows.append({
            "channel": cid, "n": int(len(y)), "pos_rate": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, s.scores)),
            "pr_auc": float(average_precision_score(y, s.scores)),
        })
    aucs = np.array([r["roc_auc"] for r in rows])
    return {
        "spacecraft": sc, "pooled_roc": pooled_roc(sc),
        "n_channels_scored": len(rows),
        "mean_roc": float(aucs.mean()), "median_roc": float(np.median(aucs)),
        "frac_above_chance": float((aucs > 0.5).mean()),
        "best": sorted(rows, key=lambda r: -r["roc_auc"])[:5],
        "worst": sorted(rows, key=lambda r: r["roc_auc"])[:5],
        "all": rows,
    }


def part_b(sc: str, channels: list[str], args, device) -> list[dict]:
    """Train and evaluate one model per channel."""
    out = []
    for cid in channels:
        cfg = WindowConfig(win_size=args.win_size)
        try:
            splits = build_splits(REPO / "data", sc, cfg, mode="entity",
                                  chan_id=cid)
        except ValueError as exc:
            print(f"    {cid}: skipped ({exc})")
            continue
        if len(splits.train) < 50 or len(splits.test) < 5:
            print(f"    {cid}: skipped (too few windows)")
            continue

        tl = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True,
                        drop_last=True)
        vl = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
        el = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)
        if len(tl) == 0:
            print(f"    {cid}: skipped (no full train batch)")
            continue

        torch.manual_seed(42)
        model = AnomalyTransformer(args.win_size, splits.n_features,
                                   splits.n_features, d_model=512, n_heads=8,
                                   e_layers=3, d_ff=512)
        tcfg = TrainConfig(epochs=args.entity_epochs, patience=3,
                           lr_halve_each_epoch=False, log_every=0)
        res = train_model(model, tl, vl, tcfg, device,
                          checkpoint_dir=REPO / "checkpoints",
                          tag=f"entity_{sc.lower()}_{cid}")
        scores = score_dataset(model, el, device)
        _, labels = align_labels(scores, splits.test_labels)
        if len(np.unique(labels)) < 2:
            print(f"    {cid}: skipped (no positives in scored region)")
            continue
        ev = evaluate(cid, labels, scores, n_thresholds=500)
        row = {"channel": cid, "roc_auc": ev.roc_auc, "pr_auc": ev.pr_auc,
               "raw_f1": ev.raw.f1, "pa_f1": ev.adjusted.f1,
               "best_epoch": res.best_epoch + 1,
               "configured_epochs": args.entity_epochs,
               "n_test": int(len(labels))}
        out.append(row)
        print(f"    {cid:<6} ROC {row['roc_auc']:.4f}  rawF1 "
              f"{row['raw_f1']:.4f}  best_ep {row['best_epoch']}"
              f"/{args.entity_epochs}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacecraft", nargs="*", default=["SMAP", "MSL"])
    ap.add_argument("--n-channels", type=int, default=6)
    ap.add_argument("--entity-epochs", type=int, default=10)
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    report = {}
    for sc in args.spacecraft:
        print("=" * 74)
        print(f"A. {sc}: concat-trained model, scored PER CHANNEL")
        print("=" * 74)
        a = part_a(sc)
        report[sc] = {"part_a": a}
        print(f"  pooled ROC (published protocol) : {a['pooled_roc']:.4f}")
        print(f"  per-channel mean ROC            : {a['mean_roc']:.4f}")
        print(f"  per-channel median ROC          : {a['median_roc']:.4f}")
        print(f"  channels above chance           : "
              f"{a['frac_above_chance']:.0%} of {a['n_channels_scored']}")
        print("  best  : " + ", ".join(f"{r['channel']} {r['roc_auc']:.3f}"
                                       for r in a["best"]))
        print("  worst : " + ", ".join(f"{r['channel']} {r['roc_auc']:.3f}"
                                       for r in a["worst"]))

        if not args.skip_training:
            print(f"\nB. {sc}: one model trained PER CHANNEL "
                  f"(top {args.n_channels} by per-channel ROC)")
            picks = [r["channel"] for r in a["best"][:args.n_channels]]
            if len(picks) < args.n_channels:
                picks = [r["channel"] for r in a["all"][:args.n_channels]]
            device = torch.device("cuda")
            b = part_b(sc, picks, args, device)
            report[sc]["part_b"] = b
            if b:
                m = float(np.mean([r["roc_auc"] for r in b]))
                print(f"  entity-trained mean ROC: {m:.4f}")
        print()

    out = REPO / "reports" / "phase7_entity.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
