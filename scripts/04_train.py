"""Train the Anomaly Transformer on SMAP or MSL (Config A).

Probes VRAM first and drops to Plan B (batch 32 + accumulation x2) on its own
if the measurement exceeds the budget.

Usage:
    python scripts/04_train.py --spacecraft SMAP
    python scripts/04_train.py --spacecraft MSL --epochs 3
    python scripts/04_train.py --smoke        # 50 steps, verifies the loop runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.models.memory import select_batch_size  # noqa: E402
from mtsad.training import TrainConfig, train_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=REPO / "data")
    ap.add_argument("--spacecraft", default="SMAP", choices=["SMAP", "MSL"])
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--e-layers", type=int, default=3)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--budget-gb", type=float, default=6.5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, 1 epoch: proves the loop runs end to end")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1
    device = torch.device("cuda")

    print("=" * 70)
    print(f"Loading {args.spacecraft} (concat mode, win={args.win_size}) ...")
    splits = build_splits(
        args.data_root, args.spacecraft, WindowConfig(win_size=args.win_size)
    )
    n_feat = splits.n_features
    print(f"  train {len(splits.train):,} windows   val {len(splits.val):,}   "
          f"test {len(splits.test):,}   features {n_feat}")

    model_kwargs = dict(
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_ff=args.d_ff,
    )

    # ---- VRAM probe with automatic Plan B fallback ----
    print(f"\nProbing VRAM at batch {args.batch_size} "
          f"(budget {args.budget_gb} GB) ...")
    batch, accum, probes = select_batch_size(
        args.batch_size, args.win_size, n_feat, device,
        args.budget_gb, not args.no_amp, **model_kwargs,
    )
    for p in probes:
        print(p)
    if accum > 1:
        print(f"  -> PLAN B ENGAGED: batch {batch} x {accum} accumulation "
              f"(effective {batch * accum})")
    else:
        print(f"  -> batch {batch}, no accumulation")

    train_ds, val_ds = splits.train, splits.val
    if args.smoke:
        train_ds = Subset(train_ds, range(min(50 * batch, len(train_ds))))
        val_ds = Subset(val_ds, range(min(10 * batch, len(val_ds))))
        args.epochs = 1
        print(f"\n  SMOKE TEST: {len(train_ds)} train / {len(val_ds)} val windows")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=0)

    model = AnomalyTransformer(args.win_size, n_feat, n_feat, **model_kwargs)
    print(f"\nModel: {model.n_parameters():,} parameters "
          f"({model.n_parameters() / 1e6:.3f} M)")

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, k=args.k, batch_size=batch,
        grad_accum=accum, amp=not args.no_amp,
        log_every=10 if args.smoke else 200,
    )
    print(f"AMP: {'off' if args.no_amp else 'fp16 (KL forced fp32)'}   "
          f"lr {cfg.lr}   k {cfg.k}   epochs {cfg.epochs}")
    print(f"NaN guard active for the first {cfg.nan_check_steps} steps")
    print("=" * 70 + "\n")

    tag = f"anomaly_transformer_{args.spacecraft.lower()}"
    if args.smoke:
        tag += "_smoke"
    result = train_model(
        model, train_loader, val_loader, cfg, device,
        checkpoint_dir=REPO / "checkpoints", tag=tag,
    )

    print("\n" + "=" * 70)
    print(f"best epoch {result.best_epoch + 1}  val objective "
          f"{result.best_val:+.5f}")
    if result.checkpoint:
        print(f"checkpoint: {result.checkpoint.relative_to(REPO)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
