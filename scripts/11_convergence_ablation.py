"""Does 'association helps on SMAP, hurts on MSL' survive longer training?

The Phase 3 finding is a *delta*: ROC of the full association-weighted score
minus ROC of the same weights scored on reconstruction error alone. Measuring
only the full score after retraining would not test it. This re-scores each
extended-training checkpoint both ways.

Usage:
    python scripts/11_convergence_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.eval.metrics import evaluate  # noqa: E402
from mtsad.eval.scoring import align_labels, score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402


def phase3(sc: str, name: str, field="roc_auc_pre_adjustment") -> float:
    d = json.loads((REPO / f"reports/phase3_{sc.lower()}.json").read_text())
    return next(r[field] for r in d["results"] if r["name"] == name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda")
    rows = []

    print(f"{'craft':<6}{'sched':<7}{'assoc ROC':>11}{'recon ROC':>11}"
          f"{'delta':>9}   {'3ep delta':>10}")
    print("-" * 60)

    for sc in ("SMAP", "MSL"):
        splits = build_splits(REPO / "data", sc,
                              WindowConfig(win_size=args.win_size))
        loader = DataLoader(splits.test, batch_size=args.batch_size,
                            shuffle=False)
        nf = splits.n_features
        base_delta = (phase3(sc, "Anomaly Transformer")
                      - phase3(sc, "ablation: recon only"))

        for sched in ("const", "halve"):
            ck = (REPO / "checkpoints" /
                  f"convcheck_at_{sc.lower()}_e{args.epochs}_{sched}.pt")
            if not ck.exists():
                print(f"{sc:<6}{sched:<7}  missing {ck.name}")
                continue
            model = AnomalyTransformer(args.win_size, nf, nf, d_model=512,
                                       n_heads=8, e_layers=3, d_ff=512)
            model.load_state_dict(
                torch.load(ck, map_location=device, weights_only=True)["model"]
            )
            model = model.to(device)

            s_assoc = score_dataset(model, loader, device, use_association=True)
            s_recon = score_dataset(model, loader, device, use_association=False)
            _, labels = align_labels(s_assoc, splits.test_labels)

            a = evaluate("assoc", labels, s_assoc, n_thresholds=1000)
            r = evaluate("recon", labels, s_recon, n_thresholds=1000)
            delta = a.roc_auc - r.roc_auc
            rows.append({
                "spacecraft": sc, "schedule": sched,
                "assoc_roc": a.roc_auc, "recon_roc": r.roc_auc,
                "assoc_minus_recon": delta,
                "assoc_raw_f1": a.raw.f1, "recon_raw_f1": r.raw.f1,
                "assoc_pa_f1": a.adjusted.f1,
                "delta_at_3_epochs": base_delta,
            })
            print(f"{sc:<6}{sched:<7}{a.roc_auc:>11.4f}{r.roc_auc:>11.4f}"
                  f"{delta:>+9.4f}   {base_delta:>+10.4f}")
            del model
            torch.cuda.empty_cache()

    out = REPO / "reports" / "phase7_ablation.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
