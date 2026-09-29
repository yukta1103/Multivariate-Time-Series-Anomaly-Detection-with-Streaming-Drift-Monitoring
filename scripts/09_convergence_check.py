"""Is the 'association hurts on MSL' finding an artifact of a 3-epoch budget?

The Phase 2/3 runs stopped at best_epoch == configured_epochs == 3 on both
datasets, with SMAP validation loss still falling. Early stopping never
fired, so those numbers are budget-constrained rather than converged, and
any finding resting on them is provisional.

Note on the schedule: the reference halves the learning rate every epoch
(``lr * 0.5**epoch``). At epoch 10 that is 1.95e-07, so simply running more
epochs under it trains almost not at all -- it would produce a spurious
"converged" verdict. This therefore compares:

    halve    reference schedule, extended (faithful but barely learns late)
    const    constant lr, the honest convergence test

Both with early stopping actually able to fire.

Usage:
    python scripts/09_convergence_check.py --spacecraft MSL SMAP --epochs 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.eval.metrics import evaluate  # noqa: E402
from mtsad.eval.scoring import align_labels, score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.training import TrainConfig, train_model  # noqa: E402


def baseline(sc: str) -> dict:
    data = json.loads((REPO / f"reports/phase3_{sc.lower()}.json").read_text())
    return next(r for r in data["results"] if r["name"] == "Anomaly Transformer")


def run(sc: str, epochs: int, patience: int, decay: bool, args, device) -> dict:
    tag = f"convcheck_at_{sc.lower()}_e{epochs}_{'halve' if decay else 'const'}"
    splits = build_splits(REPO / "data", sc, WindowConfig(win_size=args.win_size))
    nf = splits.n_features

    train_loader = DataLoader(splits.train, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(splits.val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(42)
    model = AnomalyTransformer(args.win_size, nf, nf, d_model=512, n_heads=8,
                               e_layers=3, d_ff=512)
    cfg = TrainConfig(epochs=epochs, lr=args.lr, patience=patience,
                      lr_halve_each_epoch=decay, log_every=0)

    t0 = time.perf_counter()
    res = train_model(model, train_loader, val_loader, cfg, device,
                      checkpoint_dir=REPO / "checkpoints", tag=tag)
    mins = (time.perf_counter() - t0) / 60

    scores = score_dataset(model, test_loader, device)
    _, labels = align_labels(scores, splits.test_labels)
    ev = evaluate("AT", labels, scores, n_thresholds=1000)

    base = baseline(sc)
    out = {
        "spacecraft": sc, "schedule": "halve" if decay else "const",
        "configured_epochs": epochs, "best_epoch": res.best_epoch + 1,
        "stopped_early": res.stopped_early,
        "converged": res.stopped_early or (res.best_epoch + 1) < epochs,
        "minutes": round(mins, 1),
        "val_rec_by_epoch": [round(h.val_rec, 5) for h in res.history],
        "val_obj_by_epoch": [round(h.val_objective, 4) for h in res.history],
        "roc_auc": ev.roc_auc, "pr_auc": ev.pr_auc,
        "raw_f1": ev.raw.f1, "pa_f1": ev.adjusted.f1,
        "baseline_3ep": {
            "roc_auc": base["roc_auc_pre_adjustment"],
            "pr_auc": base["pr_auc_pre_adjustment"],
            "raw_f1": base["raw"]["f1"], "pa_f1": base["point_adjusted"]["f1"],
        },
    }
    out["delta_roc"] = out["roc_auc"] - out["baseline_3ep"]["roc_auc"]

    print(f"\n  {sc} / {out['schedule']}: best_epoch {out['best_epoch']}"
          f"/{epochs}  early_stop={res.stopped_early}  "
          f"converged={out['converged']}  ({out['minutes']} min)")
    print(f"    val_rec  {out['val_rec_by_epoch']}")
    print(f"    ROC {out['roc_auc']:.4f} (3ep was "
          f"{out['baseline_3ep']['roc_auc']:.4f}, delta "
          f"{out['delta_roc']:+.4f})   raw F1 {out['raw_f1']:.4f}   "
          f"PA F1 {out['pa_f1']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacecraft", nargs="*", default=["MSL", "SMAP"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--schedules", nargs="*", default=["const", "halve"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1
    device = torch.device("cuda")

    results = []
    for sc in args.spacecraft:
        for sched in args.schedules:
            print("=" * 74)
            print(f"{sc}  schedule={sched}  epochs={args.epochs}  "
                  f"patience={args.patience}")
            print("=" * 74)
            results.append(run(sc, args.epochs, args.patience,
                               sched == "halve", args, device))

    out = REPO / "reports" / "phase7_convergence.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")

    print("\n" + "=" * 74)
    print(f"{'craft':<6}{'sched':<7}{'best/max':<10}{'conv':<7}"
          f"{'ROC':>8}{'3ep ROC':>9}{'delta':>8}")
    print("-" * 74)
    for r in results:
        print(f"{r['spacecraft']:<6}{r['schedule']:<7}"
              f"{str(r['best_epoch']) + '/' + str(r['configured_epochs']):<10}"
              f"{str(r['converged']):<7}{r['roc_auc']:>8.4f}"
              f"{r['baseline_3ep']['roc_auc']:>9.4f}{r['delta_roc']:>+8.4f}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
