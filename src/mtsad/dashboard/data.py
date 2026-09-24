"""Stream bundles for the dashboard.

The dashboard replays *precomputed* scores rather than running the model live:
it starts instantly, needs no GPU, and cannot stall partway through a demo.
The stream is simulated and the UI says so.

Per-channel slices are cut from the concatenated run rather than re-scored in
entity mode. Caveat, surfaced in the UI: roughly 1.2% of SMAP windows (3.5% of
MSL) straddle a channel boundary, so a slice's first window carries a little
signal from the preceding channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FULL_STREAM = "__full__"


@dataclass(frozen=True)
class StreamSlice:
    """One replayable stream: telemetry, labels, and one score per timestep."""

    name: str
    telemetry: np.ndarray       # (N,) raw channel-0 value, pre-normalization
    labels: np.ndarray          # (N,) 0/1 ground truth
    scores: np.ndarray          # (N,) anomaly score for the active model
    offset: int                 # start index within the full test stream
    is_full_stream: bool = False

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def n_anomaly_segments(self) -> int:
        d = np.diff(np.concatenate([[0], self.labels.astype(np.int8), [0]]))
        return int((d == 1).sum())


@dataclass(frozen=True)
class DashboardBundle:
    spacecraft: str
    channel_ids: list[str]
    bounds: np.ndarray                  # channel boundaries into the test stream
    telemetry: np.ndarray
    labels: np.ndarray
    scores: dict[str, np.ndarray]       # "at" | "ae" -> per-timestep scores

    @classmethod
    def load(cls, path: Path) -> "DashboardBundle":
        z = np.load(path, allow_pickle=False)
        keys = [k for k in z.files if k.startswith("scores_")]
        return cls(
            spacecraft=str(z["spacecraft"]),
            channel_ids=[str(c) for c in z["channel_ids"]],
            bounds=z["bounds"],
            telemetry=z["telemetry"],
            labels=z["labels"],
            scores={k.removeprefix("scores_"): z[k] for k in keys},
        )

    def slice_for(self, channel: str, model: str) -> StreamSlice:
        """Slice by channel id, or :data:`FULL_STREAM` for the spliced stream."""
        if model not in self.scores:
            raise KeyError(f"no scores for model {model!r}")
        scores = self.scores[model]

        if channel == FULL_STREAM:
            return StreamSlice(
                name=f"{self.spacecraft} full spliced stream "
                     f"({len(self.channel_ids)} channels)",
                telemetry=self.telemetry,
                labels=self.labels,
                scores=scores,
                offset=0,
                is_full_stream=True,
            )

        if channel not in self.channel_ids:
            raise KeyError(f"unknown channel {channel!r}")
        i = self.channel_ids.index(channel)
        start, end = int(self.bounds[i]), int(self.bounds[i + 1])
        end = min(end, len(scores))
        start = min(start, end)
        return StreamSlice(
            name=f"{self.spacecraft} channel {channel}",
            telemetry=self.telemetry[start:end],
            labels=self.labels[start:end],
            scores=scores[start:end],
            offset=start,
        )

    def channels_with_anomalies(self, min_length: int = 500) -> list[str]:
        """Channels worth demoing, best first.

        Ranked by number of labelled anomaly segments, then by how much the
        telemetry actually moves. Several SMAP channels are near-constant in
        column 0 (A-1 among them) and make a flat, unreadable demo chart even
        though they do contain anomalies.
        """
        scored: list[tuple[int, float, str]] = []
        for i, cid in enumerate(self.channel_ids):
            a = int(self.bounds[i])
            b = min(int(self.bounds[i + 1]), len(self.labels))
            if b - a < min_length or not self.labels[a:b].any():
                continue
            lab = self.labels[a:b].astype(np.int8)
            n_seg = int((np.diff(np.concatenate([[0], lab, [0]])) == 1).sum())
            scored.append((n_seg, float(self.telemetry[a:b].std()), cid))

        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [cid for _, _, cid in scored]
