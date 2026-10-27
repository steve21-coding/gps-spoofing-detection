"""Shared helpers for the Streamlit dashboard: data sources, cached loaders, plots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make the `ml` package importable when run via `streamlit run`
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from ml.src.config import ALL_SCENARIOS, ARTIFACTS_DIR, MAX_UPLOAD_BYTES, SAMPLE_DATA_DIR  # noqa: E402
from ml.src.schema import ValidationError, read_csv_safely, validate_navigation_frame  # noqa: E402
from ml.src.synthetic_data import GeneratorConfig, generate_dataset  # noqa: E402

DISCLAIMER = ("**Research and educational demonstration.** All models were trained and evaluated on **simulated** "
              "navigation data. Results show that the models recognise the simulated patterns; they do not show that "
              "real-world GPS spoofing can be detected, and no prediction here is proof that a receiver was spoofed.")

STATUS_COLORS = {"normal": "#5b7db1", "suspicious": "#f2a900", "potential_spoofing": "#e5484d"}
STATUS_SYMBOLS = {"normal": "circle", "suspicious": "diamond", "potential_spoofing": "x"}

SCENARIO_INFO = {
    "normal": "Ordinary flight with receiver noise.",
    "glitch": "Benign: a few isolated multipath-like position outliers (not spoofing).",
    "dropout": "Benign: a gap with missing measurements (not spoofing).",
    "low_sat_noise": "Benign: poor satellite visibility, noisier fixes (not spoofing).",
    "jump": "Attack: position suddenly offset by 150-3000 m and stays offset.",
    "drift": "Attack: slowly growing position offset; reported speed untouched.",
    "consistent_drift": "Attack: growing offset with the reported speed also made consistent.",
    "freeze": "Attack: reported position freezes while reported speed continues.",
}


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"{title} · GPS Spoofing Detection", page_icon="🛰️", layout="wide")
    st.title(title)
    st.warning(DISCLAIMER, icon="⚠️")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    p = ARTIFACTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_persistence() -> dict | None:
    p = ARTIFACTS_DIR / "persistence.json"
    return json.loads(p.read_text()) if p.exists() else None


def require_metrics() -> dict:
    m = load_metrics()
    if m is None:
        st.error("No trained artifacts found. Run `python -m ml.src.train` from the repository root first.")
        st.stop()
    return m


@st.cache_data(show_spinner=False)
def demo_flights() -> pd.DataFrame:
    """Deterministic built-in demo: one SIMULATED flight per scenario. Seed 2025 differs from the training
    seed, so these exact flights were not used for training or evaluation."""
    cfg = GeneratorConfig(n_trajectories=len(ALL_SCENARIOS), duration_s=300, seed=2025,
                          scenario_weights={k: 1.0 for k in ALL_SCENARIOS})
    return generate_dataset(cfg)


@st.cache_data(show_spinner=False)
def example_csv_bytes() -> bytes:
    return (SAMPLE_DATA_DIR / "example_navigation.csv").read_bytes()


@st.cache_data(show_spinner="Validating data...")
def clean_data(df: pd.DataFrame):
    return validate_navigation_frame(df)


def data_source(prefix: str) -> tuple[pd.DataFrame | None, str]:
    """Render the data-source picker. Returns (raw DataFrame or None, display name)."""
    choice = st.radio("Data source", ["Built-in demo flight (simulated)", "Example CSV (simulated)", "Upload my own CSV"],
                      key=f"{prefix}_src", horizontal=True)
    if choice.startswith("Built-in"):
        scen = st.selectbox("Demo scenario", list(ALL_SCENARIOS), key=f"{prefix}_scen",
                            format_func=lambda s: f"{s} - {SCENARIO_INFO[s]}")
        d = demo_flights()
        return d[d["scenario"] == scen].reset_index(drop=True), f"demo:{scen}"
    if choice.startswith("Example"):
        return read_csv_safely(example_csv_bytes()), "example_navigation.csv"
    st.download_button("Download example CSV", example_csv_bytes(), "example_navigation.csv", "text/csv", key=f"{prefix}_dl")
    with st.expander("Required columns"):
        st.markdown("Required: `timestamp`, `lat`, `lon`. Optional: `alt`, `speed_mps`, `sat_count`, `cn0_dbhz`, `trajectory_id`. "
                    f"Maximum {MAX_UPLOAD_BYTES // 2**20} MB / 250,000 rows. Full details in `sample_data/SCHEMA.md`.")
    up = st.file_uploader("CSV file", type=["csv"], key=f"{prefix}_up")
    if up is None:
        st.info("Upload a CSV, or switch to a built-in source.")
        return None, ""
    try:
        return read_csv_safely(up.getvalue()), up.name
    except ValidationError as exc:
        st.error(str(exc))
        return None, ""


def get_clean(raw: pd.DataFrame | None):
    """Validate with friendly errors; stops the page if the data cannot be used."""
    if raw is None:
        st.stop()
    try:
        return clean_data(raw)
    except ValidationError as exc:
        st.error(f"Cannot use this file: {exc}")
        st.stop()


def trajectory_figure(d: pd.DataFrame, color_col: str | None = None, color_map: dict | None = None, title: str = "") -> go.Figure:
    """Reported track as lon/lat with correct aspect ratio (degrees of longitude shrink with latitude)."""
    d = d.dropna(subset=["lat", "lon"])
    fig = go.Figure()
    if d.empty:
        return fig
    if color_col and color_map:
        for key, g in d.groupby(color_col):
            fig.add_trace(go.Scattergl(x=g["lon"], y=g["lat"], mode="markers", name=str(key),
                                       marker=dict(size=6, color=color_map.get(key, "#888"), symbol=STATUS_SYMBOLS.get(key, "circle"))))
    else:
        t = d["timestamp"] - d["timestamp"].min()
        fig.add_trace(go.Scattergl(x=d["lon"], y=d["lat"], mode="markers", name="track",
                                   marker=dict(size=5, color=t, colorscale="Viridis", showscale=True, colorbar=dict(title="s"))))
    fig.add_trace(go.Scatter(x=[d["lon"].iloc[0]], y=[d["lat"].iloc[0]], mode="markers", name="start",
                             marker=dict(size=12, color="#22d3ee", symbol="star")))
    fig.update_layout(title=title, xaxis_title="longitude", yaxis_title="latitude", height=420,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / max(np.cos(np.radians(d["lat"].mean())), 0.05))
    return fig


def timeline_figure(res: pd.DataFrame, threshold: float, onset: float | None = None) -> go.Figure:
    """Model score over time (with threshold) above rule-alert / status strips."""
    t = res["timestamp"] - res["timestamp"].iloc[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06,
                        subplot_titles=("Detector score", "Alerts (model vs. independent rule checks)"))
    fig.add_trace(go.Scattergl(x=t, y=res["score"], mode="lines", name="score", line=dict(color="#5b7db1", width=1.5)), row=1, col=1)
    fig.add_hline(y=threshold, line=dict(color="#e6edf7", dash="dash", width=1), row=1, col=1,
                  annotation_text=f"threshold {threshold:.3g}")
    if onset is not None and not np.isnan(onset):
        x0 = onset - res["timestamp"].iloc[0]
        fig.add_vrect(x0=x0, x1=float(t.iloc[-1]), fillcolor="#e5484d", opacity=0.10, line_width=0, row="all", col=1,
                      annotation_text="simulated attack window (ground truth)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=t, y=res["model_alert"].astype(int) * 1.0, mode="lines", line_shape="hv", name="model alert",
                             line=dict(color="#e5484d")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=res["rule_alert"].astype(int) * 0.5, mode="lines", line_shape="hv", name="rule alert (not ML)",
                             line=dict(color="#f2a900")), row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="seconds since start", row=2, col=1)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig
