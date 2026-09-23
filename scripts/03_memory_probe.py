"""Measure actual VRAM for Config A before committing to a training run.

Usage:
    python scripts/03_memory_probe.py
    python scripts/03_memory_probe.py --spacecraft MSL --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.models.memory import probe_batch, select_batch_size  # noqa: E402

FEATURES = {"SMAP": 25, "MSL": 55}
ESTIMATE = (2.4, 3.0)  # GB, from the Phase 2 sizing analysis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacecraft", default="SMAP", choices=["SMAP", "MSL"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--e-layers", type=int, default=3)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--budget-gb", type=float, default=6.5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="probe a batch ladder")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; nothing to probe.")
        return 1

    device = torch.device("cuda")
    n_feat = FEATURES[args.spacecraft]
    kwargs = dict(
        d_model=args.d_model, n_heads=args.n_heads,
        e_layers=args.e_layers, d_ff=args.d_ff,
    )

    model = AnomalyTransformer(args.win_size, n_feat, n_feat, **kwargs)
    n_params = model.n_parameters()
    del model

    print("=" * 70)
    print(f"Config: d_model={args.d_model} heads={args.n_heads} "
          f"layers={args.e_layers} d_ff={args.d_ff} win={args.win_size}")
    print(f"Dataset: {args.spacecraft} ({n_feat} features)")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"AMP: {'off' if args.no_amp else 'fp16 autocast (KL loss forced fp32)'}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.3f} M)")
    print("=" * 70)

    if args.sweep:
        print("\nBatch ladder:")
        for b in (16, 32, 64, 128, 256):
            print(probe_batch(b, args.win_size, n_feat, device,
                              args.budget_gb, not args.no_amp, **kwargs))
        return 0

    print(f"\nProbing batch {args.batch_size} (budget {args.budget_gb} GB)...")
    batch, accum, probes = select_batch_size(
        args.batch_size, args.win_size, n_feat, device,
        args.budget_gb, not args.no_amp, **kwargs,
    )
    for p in probes:
        print(p)

    measured = probes[-1].train_alloc_gb
    lo, hi = ESTIMATE
    verdict = (
        "within" if lo <= measured <= hi
        else ("below" if measured < lo else "above")
    )
    print(f"\n  estimate was {lo}-{hi} GB; measured {measured:.2f} GB ({verdict})")

    if accum == 1:
        print(f"  -> proceeding with batch {batch}, no gradient accumulation")
    else:
        print(f"  -> PLAN B ENGAGED: batch {batch} x {accum} accumulation steps "
              f"(effective batch {batch * accum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
