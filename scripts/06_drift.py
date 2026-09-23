"""Phase 4: measure the drift layer on real SMAP/MSL anomaly scores.

Experiment design notes, because the obvious design is wrong:

* Detection delay is measured on a stream that is **stationary before the
  injection**. Measured on the raw score stream it is meaningless: that
  stream splices 53 unrelated channels end to end, so ADWIN is already
  alarming tens of times per thousand steps and "delay 0" just means it
  never stopped. The raw stream is still reported, as experiment A2, but as
  a property of the data rather than as a delay measurement.

* The false-positive control resamples the clean stream **and** the KS
  reference from the same pool, so the two are i.i.d. by construction and
  every alarm is unambiguously spurious. Drawing them from different slices
  of a non-stationary stream makes the KS test correctly reject, which looks
  like a false-positive explosion but is not one.

* Model-quality blindness compares two *real* score streams from the same
  spacecraft -- Anomaly Transformer vs LSTM-AE -- whose ROC-AUCs differ
  substantially. Comparing real scores against Gaussian noise would confound
  model quality with stationarity.

Usage:
    python scripts/06_drift.py
    python scripts/06_drift.py --spacecraft SMAP --sweep
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
from mtsad.drift.adwin import ADWIN  # noqa: E402
from mtsad.drift.harness import (  # noqa: E402
    detection_delay,
    false_positive_rate,
    inject_shift,
)
from mtsad.drift.kswin import KSWIN  # noqa: E402
from mtsad.eval.scoring import score_dataset  # noqa: E402
from mtsad.models.anomaly_transformer import AnomalyTransformer  # noqa: E402
from mtsad.models.lstm_ae import LSTMAutoencoder  # noqa: E402

CACHE = REPO / "data" / "processed"


def get_scores(spacecraft: str, model_kind: str, args, device) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"scores_{model_kind}_{spacecraft.lower()}.npy"
    if path.exists():
        return np.load(path)

    splits = build_splits(REPO / "data", spacecraft,
                          WindowConfig(win_size=args.win_size))
    loader = DataLoader(splits.test, batch_size=args.batch_size, shuffle=False)
    nf = splits.n_features

    if model_kind == "at":
        model = AnomalyTransformer(args.win_size, nf, nf, d_model=512,
                                   n_heads=8, e_layers=3, d_ff=512)
        ckpt = f"anomaly_transformer_{spacecraft.lower()}.pt"
        use_assoc = True
    else:
        model = LSTMAutoencoder(nf, hidden_size=384, num_layers=2)
        ckpt = f"lstm_ae_{spacecraft.lower()}.pt"
        use_assoc = False

    state = torch.load(REPO / "checkpoints" / ckpt, map_location=device,
                       weights_only=True)["model"]
    model.load_state_dict(state)
    scores = score_dataset(model.to(device), loader, device,
                           use_association=use_assoc)
    np.save(path, scores)
    return scores


def resample(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(pool, size=n, replace=True)


def factories(args, reference: np.ndarray, delta=None, alpha=None):
    return {
        "ADWIN": lambda: ADWIN(delta=delta or args.delta, clock=args.clock),
        "KSWIN": lambda: KSWIN(
            reference=reference, window_size=args.ks_window,
            alpha=alpha or args.ks_alpha, stride=args.ks_stride,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spacecraft", nargs="*", default=["SMAP", "MSL"])
    ap.add_argument("--win-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--clock", type=int, default=1)
    ap.add_argument("--ks-window", type=int, default=300)
    ap.add_argument("--ks-reference", type=int, default=2000)
    ap.add_argument("--ks-alpha", type=float, default=1e-3)
    ap.add_argument("--ks-stride", type=int, default=25)
    ap.add_argument("--fp-steps", type=int, default=100_000)
    ap.add_argument("--slice-len", type=int, default=40_000)
    ap.add_argument("--sweep", action="store_true",
                    help="sensitivity sweep for Phase 5 slider defaults")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1
    device = torch.device("cuda")
    report: dict = {"config": vars(args), "spacecraft": {}}

    for sc in args.spacecraft:
        at_scores = get_scores(sc, "at", args, device)
        ae_scores = get_scores(sc, "ae", args, device)
        entry: dict = {"n_scores": int(len(at_scores))}

        print("=" * 80)
        print(f"{sc}  --  {len(at_scores):,} scores   "
              f"mean {at_scores.mean():.2f}  sd {at_scores.std():.2f}")
        print("=" * 80)

        # Pool that both the reference and the clean stream are drawn from.
        pool = at_scores[: len(at_scores) // 5]
        reference = resample(pool, args.ks_reference, seed=0)
        dets = factories(args, reference)

        # ---------- A1. false positives, properly controlled ----------
        print("\nA1. FALSE POSITIVES -- i.i.d. stream, reference from the SAME pool")
        clean = resample(pool, args.fp_steps, seed=1)
        entry["false_positives_controlled"] = {}
        for name, f in dets.items():
            r = false_positive_rate(f, clean)
            print(f"  {name:<6} {r}")
            entry["false_positives_controlled"][name] = {
                "alarms": r.n_alarms, "per_1000": r.rate_per_1000,
            }

        # ---------- A2. the raw stream, for context not as an FPR ----------
        n_channels = len(build_splits(
            REPO / "data", sc, WindowConfig(win_size=args.win_size)
        ).test_bounds) - 1
        print(f"\nA2. RAW SCORE STREAM ({n_channels} spliced channels -- "
              f"genuinely non-stationary)")
        entry["raw_stream_alarms"] = {}
        for name, f in dets.items():
            r = false_positive_rate(f, at_scores)
            print(f"  {name:<6} {r}")
            entry["raw_stream_alarms"][name] = {
                "alarms": r.n_alarms, "per_1000": r.rate_per_1000,
            }

        # ---------- B. detection delay on a quiet base ----------
        print("\nB. DETECTION DELAY (stationary base, shift at midpoint)")
        base = resample(pool, args.slice_len, seed=2)
        at_idx = args.slice_len // 2
        donor = next((o for o in args.spacecraft if o != sc), None)
        cases = [
            ("offset +0.5 sd", dict(kind="offset", magnitude=0.5)),
            ("offset +1.0 sd", dict(kind="offset", magnitude=1.0)),
            ("offset +2.0 sd", dict(kind="offset", magnitude=2.0)),
            ("offset -1.0 sd", dict(kind="offset", magnitude=-1.0)),
            ("variance x3",    dict(kind="variance", magnitude=3.0)),
            ("variance x0.33", dict(kind="variance", magnitude=1 / 3)),
        ]
        if donor:
            cases.append((f"splice from {donor}",
                          dict(kind="splice",
                               donor=get_scores(donor, "at", args, device))))

        print(f"  {'injected shift':<22}"
              f"{'ADWIN delay':>13}{'(pre-FP)':>10}"
              f"{'KSWIN delay':>13}{'(pre-FP)':>10}")
        print("  " + "-" * 66)
        entry["delays"] = {}
        for label, kw in cases:
            stream = inject_shift(base, at_idx, **kw)
            row = f"  {label:<22}"
            entry["delays"][label] = {}
            for name, f in dets.items():
                res = detection_delay(f, stream, shift_at=at_idx)
                cell = f"{res.delay:,}" if res.detected else "MISSED"
                row += f"{cell:>13}{res.n_alarms_before_shift:>10}"
                entry["delays"][label][name] = {
                    "detected": res.detected, "delay": res.delay,
                    "alarms_before_shift": res.n_alarms_before_shift,
                }
            print(row)

        # ---------- C. blindness: two real models, very different ROC ----------
        print("\nC. BLINDNESS TO MODEL QUALITY (both streams real, same spacecraft)")
        phase3 = json.loads(
            (REPO / f"reports/phase3_{sc.lower()}.json").read_text()
        )
        roc = {r["name"]: r["roc_auc_pre_adjustment"] for r in phase3["results"]}
        entry["blindness"] = {}
        for label, scores, auc in (
            ("Anomaly Transformer", at_scores, roc["Anomaly Transformer"]),
            ("LSTM-AE baseline", ae_scores, roc["LSTM-AE baseline"]),
        ):
            p = scores[: len(scores) // 5]
            ref = resample(p, args.ks_reference, seed=0)
            b = resample(p, args.slice_len, seed=2)
            stream = inject_shift(b, at_idx, kind="offset", magnitude=2.0)
            entry["blindness"][label] = {"roc": auc}
            cells = []
            for name, f in factories(args, ref).items():
                res = detection_delay(f, stream, shift_at=at_idx)
                fp = false_positive_rate(f, resample(p, args.fp_steps, seed=1))
                cells.append(f"{name} delay {res.delay}, FP/1k "
                             f"{fp.rate_per_1000:.2f}")
                entry["blindness"][label][name] = {
                    "delay": res.delay, "fp_per_1000": fp.rate_per_1000,
                }
            print(f"  {label:<22} ROC {auc:.4f}   " + " | ".join(cells))

        # ---------- D. sensitivity sweep for Phase 5 defaults ----------
        if args.sweep:
            print("\nD. SENSITIVITY SWEEP (for the dashboard's sliders)")
            stream = inject_shift(base, at_idx, kind="offset", magnitude=1.0)
            entry["sweep"] = {"ADWIN": {}, "KSWIN": {}}
            print(f"  {'ADWIN delta':<16}{'FP/1000':>10}{'delay':>10}")
            for d in (0.05, 0.01, 0.002, 1e-4, 1e-6):
                f = factories(args, reference, delta=d)["ADWIN"]
                fp = false_positive_rate(f, clean)
                res = detection_delay(f, stream, at_idx)
                dly = res.delay if res.detected else "MISSED"
                print(f"  {d:<16.0e}{fp.rate_per_1000:>10.3f}{str(dly):>10}")
                entry["sweep"]["ADWIN"][str(d)] = {
                    "fp_per_1000": fp.rate_per_1000, "delay": res.delay,
                }
            print(f"  {'KSWIN alpha':<16}{'FP/1000':>10}{'delay':>10}")
            for a in (0.05, 0.01, 1e-3, 1e-5, 1e-8):
                f = factories(args, reference, alpha=a)["KSWIN"]
                fp = false_positive_rate(f, clean)
                res = detection_delay(f, stream, at_idx)
                dly = res.delay if res.detected else "MISSED"
                print(f"  {a:<16.0e}{fp.rate_per_1000:>10.3f}{str(dly):>10}")
                entry["sweep"]["KSWIN"][str(a)] = {
                    "fp_per_1000": fp.rate_per_1000, "delay": res.delay,
                }

        report["spacecraft"][sc] = entry
        print()

    out = REPO / "reports" / "phase4_drift.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
