"""Inspect a navigation file: data quality, trajectory, derived speed and acceleration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from common import data_source, get_clean, page_setup, trajectory_figure  # noqa: E402
from ml.src.feature_engineering import compute_features  # noqa: E402
from ml.src.rules import apply_rules  # noqa: E402
from ml.src.schema import available_feature_sets  # noqa: E402

page_setup("Navigation Data Explorer")
raw, name = data_source("explore")
clean, rep = get_clean(raw)

st.subheader("Preview")
st.dataframe(raw.head(200), width="stretch")
st.caption(f"Source: {name}. Showing the first 200 raw rows.")

st.subheader("Data quality")
k = st.columns(5)
k[0].metric("Rows after cleaning", f"{rep['n_rows_clean']:,}", f"{rep['n_rows_clean'] - rep['n_rows_input']} vs input")
k[1].metric("Trajectories", rep["n_trajectories"])
k[2].metric("Invalid coordinates", rep["n_invalid_coordinates"])
k[3].metric("Duplicate timestamps dropped", rep["n_duplicate_timestamps_dropped"])
k[4].metric("Bad timestamps dropped", rep["n_bad_timestamps_dropped"])
for w in rep["warnings"]:
    st.warning(w)
miss = pd.Series(rep["missing_by_column"], name="missing values")
if len(miss):
    st.dataframe(miss.to_frame(), width="stretch")
fsets = available_feature_sets(clean)
st.success(f"Feature sets this data supports: **{', '.join(fsets) or 'none'}** "
           "(optional columns unlock richer models; see `sample_data/SCHEMA.md`).")

traj = st.selectbox("Trajectory", sorted(clean["trajectory_id"].unique()))
d = clean[clean["trajectory_id"] == traj].reset_index(drop=True)
feats = compute_features(d)
rules = apply_rules(feats)

left, right = st.columns([1, 1])
left.plotly_chart(trajectory_figure(d, title="Reported track (colour = elapsed seconds)"), width="stretch")

t = d["timestamp"] - d["timestamp"].iloc[0]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Speed implied by positions (m/s)", "Acceleration of implied speed (m/s²)", "Altitude (m)"))
fig.add_trace(go.Scattergl(x=t, y=feats["derived_speed"], name="implied speed", mode="lines"), row=1, col=1)
if d["speed_mps"].notna().any():
    fig.add_trace(go.Scattergl(x=t, y=d["speed_mps"], name="reported speed", mode="lines", line=dict(dash="dot")), row=1, col=1)
fig.add_trace(go.Scattergl(x=t, y=feats["accel"], name="acceleration", mode="lines"), row=2, col=1)
if d["alt"].notna().any():
    fig.add_trace(go.Scattergl(x=t, y=d["alt"], name="altitude", mode="lines"), row=3, col=1)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.12))
right.plotly_chart(fig, width="stretch")

st.subheader("Physically implausible segments (rule checks, not ML)")
hit = d.assign(**{c: rules[c] for c in ("rule_alert", "rule_reasons")}).query("rule_alert")
if hit.empty:
    st.success("No rule fired on this trajectory.")
else:
    st.dataframe(hit[["timestamp", "lat", "lon", "rule_reasons"]], width="stretch", hide_index=True)
    st.caption("Rules flag implausible motion only. Glitches and dropouts trigger them too, so a rule alert alone is not evidence of spoofing.")
