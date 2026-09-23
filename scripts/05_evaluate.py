"""Phase 3: train the LSTM-AE baseline and evaluate everything side by side.

Reports raw F1 first and point-adjusted second, with the random-scorer control
alongside so the inflation is visible rather than asserted. AUC is computed on
the continuous scores before any adjustment.

Usage:
    python scripts/05_evaluate.py --spacecraft SMAP
    python scripts/05_evaluate.py --spacecraft MSL --skip-baseline-training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.eval.metrics import (  # noqa: E402
    EvalResult,
    all_positive_baseline,
    evaluate,
    random_baseline,
)
from mtsad.eval.scoring import align_labels, score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.models.lstm_ae import LSTMAutoencoder  # noqa: E402
from mtsad.training import TrainConfig, train_model  # noqa: E402

FEATURES = {"SMAP": 25, "MSL": 55}


def load_anomaly_transformer(path: Path, win: int, n_feat: int, device):
    model = AnomalyTransformer(win, n_feat, n_feat, d_model=512, n_heads=8,
                               e_layers=3, d_ff=512)
    ck = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    return model.to(device)


def train_baseline(splits, batch, device, args, tag: str):
    """Train the LSTM-AE through the identical loop the transformer used."""
    model = LSTMAutoencoder(splits.n_features, hidden_size=args.hidden_size,
                            num_layers=args.num_layers)
    print(f"  LSTM-AE: {model.n_parameters():,} parameters "
          f"({model.n_parameters() / 1e6:.3f} M), hidden={args.hidden_size}, "
          f"layers={args.num_layers}")

    train_loader = DataLoader(splits.train, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(splits.val, batch_size=batch, shuffle=False,
                            num_workers=0)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_size=batch,
                      log_every=400)
    result = train_model(model, train_loader, val_loader, cfg, device,
                         checkpoint_dir=REPO / "checkpoints", tag=tag)
    print(f"  best epoch {result.best_epoch + 1}  val {result.best_val:.5f}")
    return model


def print_result(r: EvalResult) -> None:
    print(f"\n  {r.name}")
    print(f"    RAW            P {r.raw.precision:.4f}  R {r.raw.recall:.4f}  "
          f"F1 {r.raw.f1:.4f}")
    print(f"    POINT-ADJUSTED P {r.adjusted.precision:.4f}  "
          f"R {r.adjusted.recall:.4f}  F1 {r.adjusted.f1:.4f}   "
          f"(x{r.inflation:.2f})")
    print(f"    AUC (pre-adj)  ROC {r.roc_auc:.4f}   PR {r.pr_auc:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument("--spacecraft", default="SMAP", choices=["SMAP", "MSL"])
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-thresholds", type=int, default=1000)
    ap.add_argument("--skip-baseline-training", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1
    device = torch.device("cuda")
    sc = args.spacecraft

    print("=" * 74)
    print(f"PHASE 3 EVALUATION -- {sc}")
    print("=" * 74)

    splits = build_splits(args.data_root, sc, WindowConfig(win_size=args.win_size))
    test_loader = DataLoader(splits.test, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f"  test windows {len(splits.test):,}  features {splits.n_features}")

    # ---------- Anomaly Transformer ----------
    at_path = REPO / "checkpoints" / f"anomaly_transformer_{sc.lower()}.pt"
    if not at_path.exists():
        print(f"missing {at_path}; run scripts/04_train.py --spacecraft {sc}")
        return 1
    at = load_anomaly_transformer(at_path, args.win_size, splits.n_features, device)
    print(f"\n  Anomaly Transformer: {at.n_parameters():,} parameters")

    at_scores = score_dataset(at, test_loader, device, use_association=True)
    # Ablation: same model, same weights, association weighting removed.
    at_recon = score_dataset(at, test_loader, device, use_association=False)

    # ---------- LSTM-AE baseline ----------
    tag = f"lstm_ae_{sc.lower()}"
    ae_path = REPO / "checkpoints" / f"{tag}.pt"
    print("\n  Training LSTM-AE baseline (identical loop, split, and eval) ...")
    if args.skip_baseline_training and ae_path.exists():
        ae = LSTMAutoencoder(splits.n_features, args.hidden_size, args.num_layers)
        ae.load_state_dict(
            torch.load(ae_path, map_location=device, weights_only=True)["model"]
        )
        ae = ae.to(device)
        print("  (loaded existing checkpoint)")
    else:
        ae = train_baseline(splits, args.batch_size, device, args, tag)
    ae_scores = score_dataset(ae, test_loader, device, use_association=False)

    # ---------- align and evaluate ----------
    _, labels = align_labels(at_scores, splits.test_labels)
    dropped = len(splits.test_labels) - len(labels)
    print(f"\n  scored {len(labels):,} timesteps "
          f"({dropped} trailing unscored, identical for all models)")
    print(f"  positive rate {labels.mean():.4%}")

    results = [
        evaluate("Anomaly Transformer", labels, at_scores, args.n_thresholds),
        evaluate("  ablation: recon only", labels, at_recon, args.n_thresholds),
        evaluate("LSTM-AE baseline", labels, ae_scores, args.n_thresholds),
        random_baseline(labels, seed=0, n_thresholds=args.n_thresholds),
        all_positive_baseline(labels),
    ]

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    for r in results:
        print_result(r)

    print("\n" + "=" * 74)
    print(f"{'model':<24} {'RAW F1':>8} {'PA F1':>8} {'infl':>6} "
          f"{'ROC':>7} {'PR':>7}")
    print("-" * 74)
    for r in results:
        print(f"{r.name.strip():<24} {r.raw.f1:>8.4f} {r.adjusted.f1:>8.4f} "
              f"{r.inflation:>5.2f}x {r.roc_auc:>7.4f} {r.pr_auc:>7.4f}")
    print("=" * 74)

    rand = next(r for r in results if r.name.startswith("random"))
    print(
        f"\nSanity check: a uniform random scorer gets ROC-AUC {rand.roc_auc:.4f} "
        f"(chance) and raw F1 {rand.raw.f1:.4f},\nbut point adjustment lifts it to "
        f"F1 {rand.adjusted.f1:.4f} -- a {rand.inflation:.1f}x inflation on zero "
        f"information.\nThat is why raw F1 is the headline here, not PA F1."
    )

    out = REPO / "reports" / f"phase3_{sc.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "spacecraft": sc,
            "n_points": int(len(labels)),
            "positive_rate": float(labels.mean()),
            "results": [
                {
                    "name": r.name.strip(),
                    "raw": {"precision": r.raw.precision, "recall": r.raw.recall,
                            "f1": r.raw.f1, "threshold": r.raw.threshold},
                    "point_adjusted": {"precision": r.adjusted.precision,
                                       "recall": r.adjusted.recall,
                                       "f1": r.adjusted.f1},
                    "inflation": r.inflation,
                    "roc_auc_pre_adjustment": r.roc_auc,
                    "pr_auc_pre_adjustment": r.pr_auc,
                }
                for r in results
            ],
        },
        indent=2,
    ))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
