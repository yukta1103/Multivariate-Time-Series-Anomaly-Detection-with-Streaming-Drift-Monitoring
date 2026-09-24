"""Streaming anomaly detection dashboard with drift monitoring.

Run:
    streamlit run app/streamlit_app.py

Design constraints carried over from Phase 4's findings:

* Model quality is its own always-visible strip, never folded into the drift
  status. Phase 4 showed drift behaviour was near-identical across a 0.16 AUC
  gap spanning below-chance to functional, so a quiet drift banner says
  nothing about whether the model works.
* The demo defaults to a single continuous channel. On the full spliced
  stream both detectors fire ~40-56 times per 1000 steps, because the
  benchmark joins 53 unrelated channels end to end -- real drift, but an
  artifact of the dataset's construction rather than model instability. That
  view stays available, with an explanation attached.
* Both detectors run at once: ADWIN is structurally blind to variance-only
  decreases, KSWIN catches them.
* Slider positions snap to the values actually swept in Phase 4, so the
  false-positive and delay figures shown are measured, not interpolated.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mtsad.dashboard.data import FULL_STREAM, DashboardBundle  # noqa: E402
from mtsad.dashboard.reference import (  # noqa: E402
    MODEL_LABELS,
    model_quality,
    nearest_point,
    raw_stream_alarm_rate,
    sweep_points,
)
from mtsad.dashboard.session import DetectorSettings, StreamSession  # noqa: E402

DATA = REPO / "data" / "processed"
REPORTS = REPO / "reports"

ADWIN_COLOR = "#7F77DD"
KSWIN_COLOR = "#D4537E"
SCORE_COLOR = "#1D9E75"
THRESH_COLOR = "#BA7517"

st.set_page_config(page_title="Telemetry anomaly + drift monitor",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def load_bundle(spacecraft: str) -> DashboardBundle:
    path = DATA / f"dashboard_{spacecraft.lower()}.npz"
    if not path.exists():
        st.error(
            f"Missing {path.relative_to(REPO)}. Run "
            "`python scripts/07_prepare_dashboard.py` first."
        )
        st.stop()
    return DashboardBundle.load(path)


# ----------------------------------------------------------------- sidebar --

st.sidebar.title("Controls")
st.sidebar.caption("Replaying precomputed scores as a simulated live stream.")

spacecraft = st.sidebar.selectbox("Spacecraft", ["SMAP", "MSL"])
bundle = load_bundle(spacecraft)

model = st.sidebar.selectbox(
    "Model", ["at", "ae"], format_func=lambda k: MODEL_LABELS[k]
)

interesting = bundle.channels_with_anomalies()
advanced = st.sidebar.toggle(
    "Full spliced stream (advanced)", value=False,
    help="The concatenated benchmark stream. Both detectors alarm constantly "
         "here because it joins unrelated channels end to end.",
)
if advanced:
    channel = FULL_STREAM
else:
    channel = st.sidebar.selectbox(
        "Channel", interesting,
        help="A single continuous telemetry channel, where drift is legible.",
    )

st.sidebar.divider()
st.sidebar.subheader("Detector sensitivity")

adwin_pts = sweep_points(REPORTS, spacecraft, "ADWIN")
ks_pts = sweep_points(REPORTS, spacecraft, "KSWIN")

adwin_delta = st.sidebar.select_slider(
    "ADWIN delta", options=[p.value for p in adwin_pts], value=0.002,
    format_func=lambda v: f"{v:.0e}",
    help="Confidence bound. Lower is more conservative.",
)
st.sidebar.caption(
    f"Phase 4 on {spacecraft}: {nearest_point(adwin_pts, adwin_delta).label}"
)

ks_alpha = st.sidebar.select_slider(
    "KSWIN alpha", options=[p.value for p in ks_pts], value=1e-3,
    format_func=lambda v: f"{v:.0e}",
    help="KS p-value threshold. Lower is more conservative.",
)
st.sidebar.caption(
    f"Phase 4 on {spacecraft}: {nearest_point(ks_pts, ks_alpha).label}"
)

with st.sidebar.expander("Advanced detector settings"):
    adwin_cooldown = st.number_input(
        "ADWIN cooldown", 0, 5000, 200, step=50,
        help="Refractory period. ADWIN cuts repeatedly after one change, so "
             "0 produces a burst of alarms per event.",
    )
    ks_window = st.number_input("KSWIN window", 50, 2000, 300, step=50)
    ks_stride = st.number_input("KSWIN stride", 1, 200, 25, step=5)
    ks_cooldown = st.number_input("KSWIN cooldown", 0, 5000, 200, step=50)
    threshold_q = st.slider("Threshold quantile", 0.90, 0.999, 0.99, 0.001)

st.sidebar.divider()
st.sidebar.subheader("Playback")
speed = st.sidebar.slider("Steps per tick", 1, 500, 50)
col_a, col_b, col_c = st.sidebar.columns(3)
play = col_a.button("Play", use_container_width=True)
pause = col_b.button("Pause", use_container_width=True)
reset = col_c.button("Reset", use_container_width=True)

# ------------------------------------------------------------------ state --

settings = DetectorSettings(
    adwin_delta=adwin_delta, adwin_cooldown=int(adwin_cooldown),
    ks_alpha=ks_alpha, ks_window=int(ks_window),
    ks_stride=int(ks_stride), ks_cooldown=int(ks_cooldown),
)
stream_key = (spacecraft, model, channel, threshold_q)

if st.session_state.get("stream_key") != stream_key:
    st.session_state.stream_key = stream_key
    st.session_state.session = StreamSession(
        bundle.slice_for(channel, model), settings,
        threshold_quantile=threshold_q,
    )
    st.session_state.playing = False

session: StreamSession = st.session_state.session
session.apply_settings(settings)

if play:
    st.session_state.playing = True
if pause:
    st.session_state.playing = False
if reset:
    session.reset()
    st.session_state.playing = False

if st.session_state.get("playing") and not session.finished:
    session.advance(speed)

# ------------------------------------------------------------------- head --

st.title("Telemetry anomaly detection with drift monitoring")
st.caption(f"{session.stream.name} · {len(session.stream):,} timesteps · "
           f"{session.stream.n_anomaly_segments} labelled anomaly segments")

quality = model_quality(REPORTS, spacecraft, model)
getattr(st, quality.severity)(
    f"**Model quality — reported independently of drift status.** "
    f"{MODEL_LABELS[model]} on {spacecraft}: "
    f"**ROC-AUC {quality.roc_auc:.4f}** ({quality.verdict}). "
    f"Raw F1 {quality.raw_f1:.4f} against an all-positive floor of "
    f"{quality.all_positive_f1:.4f}. "
    f"Point-adjusted F1 is {quality.adjusted_f1:.4f}, but a random scorer "
    f"reaches a comparable figure on this benchmark — treat it as "
    f"non-informative."
)

if session.stream.is_full_stream:
    st.warning(
        f"**Full spliced stream.** Frequent alerts here reflect the "
        f"benchmark's channel-splicing construction, not model instability: "
        f"this stream joins {len(bundle.channel_ids)} unrelated channels end "
        f"to end. Phase 4 measured "
        f"{raw_stream_alarm_rate(REPORTS, spacecraft, 'ADWIN'):.0f} "
        f"ADWIN and "
        f"{raw_stream_alarm_rate(REPORTS, spacecraft, 'KSWIN'):.0f} "
        f"KSWIN alarms per 1000 steps here, against "
        f"{nearest_point(adwin_pts, adwin_delta).fp_per_1000:.2f} and "
        f"{nearest_point(ks_pts, ks_alpha).fp_per_1000:.2f} on stationary "
        f"data. Switch off 'Full spliced stream' for a legible drift signal."
    )

# ----------------------------------------------------------- drift banner --

counts = session.counts()
if session.needs_recalibration:
    recent = [a for a in session.alarms][-4:]
    detail = ", ".join(f"{a.detector} at t={a.index + session.stream.offset:,}"
                       for a in recent)
    left, right = st.columns([5, 1])
    left.error(
        f"**Score distribution shifted** — {detail}. The operating threshold "
        f"may need recalibrating. This tracks the *score distribution*, not "
        f"whether the model is correct."
    )
    if right.button("Recalibrate", use_container_width=True, type="primary"):
        new = session.recalibrate()
        st.toast(f"Threshold recalibrated to {new:.2f}")
        st.rerun()
else:
    st.success(
        "**No distribution shift detected.** Note this is a statement about "
        "the score distribution only — see the model-quality panel above for "
        "whether those scores mean anything."
    )

# ----------------------------------------------------------------- charts --

view = session.visible(span=2000)
if len(view["index"]) == 0:
    st.info("Press **Play** to begin the replay.")
    st.stop()

lo, hi = int(view["index"][0]), int(view["index"][-1]) + 1
alarms = session.alarms_in(lo, hi)


def anomaly_bands(fig: go.Figure, idx: np.ndarray, labels: np.ndarray) -> None:
    d = np.diff(np.concatenate([[0], labels.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        fig.add_vrect(x0=idx[s], x1=idx[min(e, len(idx) - 1)],
                      fillcolor="#E24B4A", opacity=0.18, line_width=0)


def alarm_lines(fig: go.Figure) -> None:
    for a in alarms:
        fig.add_vline(
            x=a.index, line_width=1.4,
            line_dash="solid" if a.detector == "ADWIN" else "dash",
            line_color=ADWIN_COLOR if a.detector == "ADWIN" else KSWIN_COLOR,
        )


c1, c2 = st.columns([3, 2])

with c1:
    fig = go.Figure()
    fig.add_scatter(x=view["index"], y=view["telemetry"], mode="lines",
                    line=dict(width=1, color="#378ADD"), name="telemetry")
    anomaly_bands(fig, view["index"], view["labels"])
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=28, b=0),
                      title="Sensor channel · shaded = labelled anomaly",
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_scatter(x=view["index"], y=view["scores"], mode="lines",
                    line=dict(width=1, color=SCORE_COLOR), name="score")
    fig.add_hline(y=session.threshold, line_dash="dot",
                  line_color=THRESH_COLOR, annotation_text="threshold")
    flagged = view["flagged"]
    if len(flagged):
        fig.add_scatter(
            x=flagged, y=session.stream.scores[flagged], mode="markers",
            marker=dict(size=5, color="#E24B4A"), name="flagged",
        )
    anomaly_bands(fig, view["index"], view["labels"])
    alarm_lines(fig)
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=28, b=0),
                      title="Anomaly score · threshold · flagged points · "
                            "drift alarms", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"<span style='color:{ADWIN_COLOR}'>━ ADWIN alarm</span> &nbsp;&nbsp; "
        f"<span style='color:{KSWIN_COLOR}'>╌ KSWIN alarm</span> &nbsp;&nbsp; "
        f"<span style='color:{THRESH_COLOR}'>╌ threshold</span>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("**Detector internals — why both run**")
    m1, m2 = st.columns(2)
    m1.metric("ADWIN window", f"{session.adwin_width:,}",
              help="Collapses when ADWIN cuts on a mean shift.")
    p = session.kswin_p_value
    m2.metric("KSWIN p-value", "—" if np.isnan(p) else f"{p:.2e}",
              help="Compares the recent window's full CDF to the reference.")
    st.caption(
        "ADWIN cuts when two sub-window **means** diverge, so it is "
        "structurally blind to a variance-only decrease — Phase 4 confirmed "
        "it missed a `variance x0.33` injection on both spacecraft, while "
        "KSWIN caught it in 99 steps. That is why both ship."
    )

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Step", f"{session.position:,}")
    k2.metric("ADWIN alarms", counts["ADWIN"])
    k3.metric("KSWIN alarms", counts["KSWIN"])
    k4, k5, k6 = st.columns(3)
    k4.metric("Flagged", f"{session.n_flagged:,}")
    k5.metric("Threshold", f"{session.threshold:.3g}")
    k6.metric("Recalibrations", session.n_recalibrations)
    st.caption(
        "Alarm counts group bursts: ADWIN cuts repeatedly after one change, "
        f"so a {settings.adwin_cooldown}-step refractory period collapses a "
        "burst into a single event. The per-crossing rates quoted by the "
        "sliders were measured without it."
    )

    st.progress(min(session.position / max(len(session.stream), 1), 1.0))

# ---------------------------------------------------------------- ticking --

if st.session_state.get("playing") and not session.finished:
    time.sleep(0.15)
    st.rerun()
elif session.finished and st.session_state.get("playing"):
    st.session_state.playing = False
    st.toast("Replay complete.")
