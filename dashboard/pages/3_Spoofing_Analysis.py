"""Run a trained model on a flight and review alerts, reasons and limitations."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    STATUS_COLORS, data_source, get_clean, load_metrics, page_setup, timeline_figure, trajectory_figure,
)
from ml.src.predict import (  # noqa: E402
    FEATURE_DESCRIPTIONS, available_models, decide, default_model_key, find_intervals, score_frame, summarize,
)
from ml.src.schema import ValidationError  # noqa: E402

page_setup("Interactive Spoofing Analysis")


@st.cache_data(show_spinner="Scoring...")
def cached_score(df: pd.DataFrame, model_key: str):
    return score_frame(df, model_key)


raw, name = data_source("analysis")
clean, rep = get_clean(raw)
models = available_models(clean)
if not models:
    st.error("This file has no supported feature set. Provide at least `timestamp`, `lat`, `lon`.")
    st.stop()

keys = [m["key"] for m in models]
dflt = default_model_key()
c1, c2, c3 = st.columns([2, 2, 1])
model_key = c1.selectbox("Model", keys, index=keys.index(dflt) if dflt in keys else 0,
                         format_func=lambda k: k.replace("__", " · ").replace("_", " ") + ("  (selected)" if k == dflt else ""))
try:
    sf = cached_score(clean, model_key)
except ValidationError as exc:
    st.error(str(exc))
    st.stop()

finite = sf.score[np.isfinite(sf.score)]
lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
if sf.kind == "supervised":
    lo, hi = 0.0, 1.0
step = (hi - lo) / 500 or 0.01
thr = c2.slider("Decision threshold", lo, hi, float(np.clip(sf.default_threshold, lo, hi)), step,
                help="Default is the threshold chosen on the validation split during training.")
k_alerts = c3.number_input("Consecutive alerts needed", 1, 30, 3, help="Require this many flagged samples in a row. "
                           "Higher values cut false alarms but delay detection (see Model Comparison).")

kind_txt = ("**Supervised classifier**: trained on labelled simulated attacks. Flags show *potential spoofing*." if sf.kind == "supervised"
            else "**Unsupervised anomaly detector**: trained on normal flights only, no labels. Flags show only *suspicious* behaviour.")
st.info(kind_txt + " Rule-based checks run independently and are shown separately (they are not ML).")

res = decide(sf, thr, int(k_alerts))
summary = summarize(res, sf, int(k_alerts))
sc = summary["status_counts"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Normal rows", f"{sc['normal']:,}")
m2.metric("Suspicious rows", f"{sc['suspicious']:,}")
m3.metric("Potential spoofing rows", f"{sc['potential_spoofing']:,}")
m4.metric("Alert intervals", summary["intervals"])

traj = st.selectbox("Trajectory", sorted(res["trajectory_id"].unique()))
mask = (res["trajectory_id"] == traj).to_numpy()
r = res[mask].reset_index(drop=True)
onset = None
if "attack_onset_ts" in clean.columns:
    o = clean.loc[mask, "attack_onset_ts"].iloc[0]
    onset = None if pd.isna(o) else float(o)

left, right = st.columns([1, 1.3])
left.plotly_chart(trajectory_figure(r, "status", STATUS_COLORS, "Reported track by status"), width="stretch")
right.plotly_chart(timeline_figure(r, thr, onset), width="stretch")

if "scenario" in clean.columns:
    st.caption(f"Ground truth (simulation): scenario **{clean.loc[mask, 'scenario'].iloc[0]}**. "
               "Labels come from the simulator and are shown for evaluation only; the model never sees them.")

st.subheader("Suspicious intervals")
iv = find_intervals(res)
iv = iv[iv["trajectory_id"] == traj].reset_index(drop=True)
if iv.empty:
    st.success("No alerts on this trajectory at the current settings.")
else:
    shown = iv.assign(start=iv["start"] - r["timestamp"].iloc[0], end=iv["end"] - r["timestamp"].iloc[0])
    shown = shown.rename(columns={"start": "start (s)", "end": "end (s)"})
    st.dataframe(shown.drop(columns=["trajectory_id"]).style.format({"max_score": "{:.3f}", "start (s)": "{:.0f}", "end (s)": "{:.0f}", "duration_s": "{:.0f}"}),
                 width="stretch", hide_index=True)

    pick = st.selectbox("Inspect an interval", range(len(iv)),
                        format_func=lambda i: f"#{i}: {iv.loc[i, 'status']} at {iv.loc[i, 'start'] - r['timestamp'].iloc[0]:.0f}s ({iv.loc[i, 'evidence']})")
    row = res[(res["trajectory_id"] == traj) & (res["timestamp"] == iv.loc[pick, "start"])].index[0]
    st.markdown(f"**Why flagged:** {res.loc[row, 'top_deviating_features'] or 'no single feature stands out'}")
    if res.loc[row, "rule_reasons"]:
        st.markdown(f"**Rule checks:** {res.loc[row, 'rule_reasons']}")
    base = sf.baseline
    fv = pd.DataFrame({"feature": sf.bundle_features,
                       "meaning": [FEATURE_DESCRIPTIONS.get(f, "") for f in sf.bundle_features],
                       "value": sf.features.loc[row, sf.bundle_features].to_numpy(float),
                       "typical (median of normal flights)": [base["median"][f] for f in sf.bundle_features]})
    fv["deviation (robust z)"] = (fv["value"] - fv["typical (median of normal flights)"]).abs() / (
        np.array([base["mad"][f] for f in sf.bundle_features]) * 1.4826)
    st.dataframe(fv.sort_values("deviation (robust z)", ascending=False).style.format(
        {"value": "{:.3f}", "typical (median of normal flights)": "{:.3f}", "deviation (robust z)": "{:.1f}"}, na_rep="n/a"),
        width="stretch", hide_index=True)

with st.expander("Limitations that apply to this run", expanded=True):
    for lim in summary["limitations"]:
        st.markdown(f"- {lim}")
    for w in summary["warnings"]:
        st.markdown(f"- Input: {w}")
    if res["rule_alert"].any() and not res["model_alert"].any():
        st.markdown("- Only rule checks fired. Glitches, dropouts and multipath cause this as well as spoofing.")

st.subheader("Export")
safe = name.replace(":", "_").replace(" ", "_")
e1, e2, e3 = st.columns(3)
e1.download_button("Predictions (CSV)", res.to_csv(index=False), f"predictions_{safe}.csv", "text/csv")
e2.download_button("Alert intervals (CSV)", find_intervals(res).to_csv(index=False), f"intervals_{safe}.csv", "text/csv")
e3.download_button("Analysis summary (JSON)", json.dumps(summary, indent=2, default=float), f"summary_{safe}.json", "application/json")
