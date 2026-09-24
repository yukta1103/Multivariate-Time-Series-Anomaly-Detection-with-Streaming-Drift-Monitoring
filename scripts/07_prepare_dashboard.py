"""Bundle everything the dashboard replays into one .npz per spacecraft.

Runs the models once, offline, so the Streamlit app needs no GPU and starts
instantly. Output lands in data/processed/ (gitignored).

Usage:
    python scripts/07_prepare_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.data.dataset import WindowConfig, build_splits  # noqa: E402
from mtsad.data.telemanom import build_concat, channel_ids  # noqa: E402
from mtsad.eval.scoring import score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.models.lstm_ae import LSTMAutoencoder  # noqa: E402

OUT = REPO / "data" / "processed"


def build_model(kind: str, n_feat: int, win: int, spacecraft: str, device):
    if kind == "at":
        model = AnomalyTransformer(win, n_feat, n_feat, d_model=512, n_heads=8,
                                   e_layers=3, d_ff=512)
        ckpt = f"anomaly_transformer_{spacecraft.lower()}.pt"
    else:
        model = LSTMAutoencoder(n_feat, hidden_size=384, num_layers=2)
        ckpt = f"lstm_ae_{spacecraft.lower()}.pt"
    state = torch.load(REPO / "checkpoints" / ckpt, map_location=device,
                       weights_only=True)["model"]
    model.load_state_dict(state)
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacecraft", nargs="*", default=["SMAP", "MSL"])
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available (needed once, to precompute scores).")
        return 1
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    for sc in args.spacecraft:
        print(f"\n{sc}")
        splits = build_splits(REPO / "data", sc,
                              WindowConfig(win_size=args.win_size))
        loader = DataLoader(splits.test, batch_size=args.batch_size,
                            shuffle=False)
        _, test_raw, labels, _, bounds = build_concat(REPO / "data", sc)

        scores = {}
        for kind in ("at", "ae"):
            model = build_model(kind, splits.n_features, args.win_size, sc, device)
            scores[kind] = score_dataset(
                model, loader, device, use_association=(kind == "at")
            )
            print(f"  {kind}: {len(scores[kind]):,} scores")
            del model
            torch.cuda.empty_cache()

        # Non-overlapping windows leave a short unscored tail; trim everything
        # to the scored length so index i means the same thing in every array.
        n = min(len(v) for v in scores.values())
        path = OUT / f"dashboard_{sc.lower()}.npz"
        np.savez_compressed(
            path,
            spacecraft=sc,
            channel_ids=np.array(channel_ids(REPO / "data", sc)),
            bounds=np.asarray(bounds, dtype=np.int64),
            telemetry=test_raw[:n, 0].astype(np.float32),
            labels=labels[:n].astype(np.int8),
            **{f"scores_{k}": v[:n].astype(np.float32) for k, v in scores.items()},
        )
        print(f"  wrote {path.relative_to(REPO)} "
              f"({path.stat().st_size / 1024**2:.1f} MB, {n:,} steps, "
              f"{len(bounds) - 1} channels)")

    print("\ndone. launch with:  streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
