"""Model comparison: every number is read from artifacts/metrics.json (written by training)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from common import ARTIFACTS_DIR, load_persistence, page_setup, require_metrics  # noqa: E402

page_setup("Model Comparison")
m = require_metrics()

SPLIT_HELP = {
    "train": "Training split: the models were fitted on it, so these numbers are optimistic.",
    "val": "Validation split: used to choose hyper-parameters and decision thresholds, so also slightly optimistic.",
    "test": "Test split: whole flights never used for fitting or tuning. This is the number to quote.",
}
c1, c2 = st.columns(2)
fs = c1.selectbox("Feature set", list(m["models"]), index=list(m["models"]).index("full"),
                  format_func=lambda k: f"{k}: {m['feature_sets'][k]['requires']}")
split = c2.radio("Split", ["test", "val", "train"], horizontal=True)
st.info(SPLIT_HELP[split])

rows = []
for name, r in m["models"][fs].items():
    x = r[split]
    rows.append({"model": name, "type": r["kind"], "precision": x["precision"], "recall": x["recall"], "F1": x["f1"],
                 "accuracy": x["accuracy"], "PR-AUC": x["pr_auc"], "ROC-AUC": x["roc_auc"], "false-positive rate": x["false_positive_rate"],
                 "fit time (s)": r["timing"].get("fit_s")})
tab = pd.DataFrame(rows)
st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns if c not in ("model", "type")}, na_rep="n/a"),
             width="stretch", hide_index=True)
st.caption("Accuracy is shown for completeness but is misleading here: about 78% of rows are not spoofed. Compare precision, "
           "recall and PR-AUC. Thresholds: supervised = highest recall with at most 2% false positives on validation; "
           "unsupervised = 99th percentile of scores on normal training flights.")

# ---- flat export of everything
flat = [{"feature_set": f, "model": n, "split": s, **{k: v for k, v in r[s].items() if k != "confusion_matrix"},
         **{f"cm_{k}": v for k, v in r[s]["confusion_matrix"].items()}}
        for f, ms in m["models"].items() for n, r in ms.items() for s in ("train", "val", "test")]
st.download_button("Download full comparison (CSV)", pd.DataFrame(flat).to_csv(index=False), "model_comparison.csv", "text/csv")

# ---- single-model detail
st.subheader("Model detail")
model = st.selectbox("Model", list(m["models"][fs]))
r = m["models"][fs][model]
left, right = st.columns(2)
cm = r[split]["confusion_matrix"]
z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
fig = px.imshow(z, text_auto=True, x=["predicted normal", "predicted spoofed"], y=["actually normal", "actually spoofed"],
                color_continuous_scale="Blues", title=f"Confusion matrix ({split}, row level)")
fig.update_layout(height=360, coloraxis_showscale=False, margin=dict(l=10, r=10, t=40, b=10))
left.plotly_chart(fig, width="stretch")
st.caption(f"Missing a spoofed row (false negative) means an undetected attack sample; a false positive is an alert on non-spoofed data. "
           f"Threshold: **{r['threshold']:.4f}**.")

ps = pd.DataFrame([{"scenario": s, **v} for s, v in r["per_scenario_test"].items()])
ps["metric"] = ps.apply(lambda x: "recall (attack rows caught)" if x["kind"] == "attack" else "false-positive rate", axis=1)
ps["value"] = ps.apply(lambda x: x["recall"] if x["kind"] == "attack" else x["false_positive_rate"], axis=1)
fig = px.bar(ps, x="scenario", y="value", color="metric", title="Per-scenario behaviour (test split)",
             color_discrete_map={"recall (attack rows caught)": "#22d3ee", "false-positive rate": "#f2a900"})
fig.update_layout(height=360, yaxis_range=[0, 1.02], margin=dict(l=10, r=10, t=40, b=10))
right.plotly_chart(fig, width="stretch")

ev = r["events_test"]
st.markdown("**Flight-level view (test split):** was each attack noticed at all, and how quickly?")
a = pd.DataFrame([{"attack": k, "flights": v["n"], "detection rate": v["detection_rate"], "median delay after onset (s)": v["median_delay_s"]}
                  for k, v in ev["attacks"].items()])
st.dataframe(a.style.format({"detection rate": "{:.2f}", "median delay after onset (s)": "{:.1f}"}, na_rep="n/a"),
             width="stretch", hide_index=True)
na = ev["non_attack_flights"]
st.caption(f"Non-attack flights: **{na['share_with_any_alert']:.0%}** raised at least one alert ({na['alerts_per_hour']:.0f} alerts per flight-hour) at this threshold.")

if "calibration_test" in r:
    cal = r["calibration_test"]
    b = pd.DataFrame(cal["bins"])
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfectly calibrated", line=dict(dash="dash", color="#888")),
                     go.Scatter(x=b["mean_predicted"], y=b["fraction_positive"], mode="lines+markers", name=model)])
    fig.update_layout(title=f"Reliability (Brier {cal['brier']:.3f}, ECE {cal['ece']:.3f})", height=320,
                      xaxis_title="mean predicted probability", yaxis_title="fraction actually spoofed", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Points far from the diagonal mean `predict_proba` should not be read as a literal probability.")

# ---- persistence trade-off
p = load_persistence()
if p:
    st.subheader("Alert persistence trade-off")
    st.caption(f"Selected model `{p['model']}`, test split, simulated data. Require k consecutive flagged samples before alerting.")
    pr = pd.DataFrame([{"k": x["min_consecutive"], "precision": x["precision"], "recall": x["recall"],
                        "clean flights with any alert": x["events"]["non_attack_flights"]["share_with_any_alert"],
                        "alerts per clean flight-hour": x["events"]["non_attack_flights"]["alerts_per_hour"],
                        "median delay, drift (s)": x["events"]["attacks"]["drift"]["median_delay_s"],
                        "median delay, jump (s)": x["events"]["attacks"]["jump"]["median_delay_s"]} for x in p["results"]])
    st.dataframe(pr.style.format({c: "{:.3f}" for c in pr.columns if c != "k"}, na_rep="n/a"), width="stretch", hide_index=True)

# ---- generalisation
st.subheader("Does it generalise to an attack type it never saw?")
g = m["generalization"]
gfs = st.radio("Feature set", list(g), horizontal=True, key="gen_fs")
gd = pd.DataFrame([{"attack": a, "attack type in training": v["recall_seen_in_training"], "attack type held out": v["recall_held_out"]}
                   for a, v in g[gfs].items()]).melt("attack", var_name="training", value_name="recall")
fig = px.bar(gd, x="attack", y="recall", color="training", barmode="group", title=f"Random forest, {gfs} features (test recall)")
fig.update_layout(height=340, yaxis_range=[0, 1.02], margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Held-out recall is lower for jump and drift: a model is only as general as the attack types it was trained on.")

# ---- rule baseline, kept separate
st.subheader("Rule-based baseline (not machine learning)")
rt = m["rules"]["test"]
st.markdown(f"Thresholds are fixed from physical plausibility, not tuned on any data. Test split: precision **{rt['precision']:.3f}**, "
            f"recall **{rt['recall']:.3f}**, F1 **{rt['f1']:.3f}**, false-positive rate **{rt['false_positive_rate']:.3f}**.")
with st.expander("Rule configuration"):
    st.json(m["rules"]["config"])

# ---- static plots
st.subheader("Plots generated by the training run")
plots = ARTIFACTS_DIR / "plots"
for title, f in [("Precision-recall curves (test)", f"pr_curves_{fs}.png"), ("Random forest feature importance", f"importance_rf_{fs}.png")]:
    if (plots / f).exists():
        st.markdown(f"**{title}**")
        st.image(str(plots / f), width=640)
