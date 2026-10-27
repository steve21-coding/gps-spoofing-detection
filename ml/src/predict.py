"""Inference: raw navigation data -> structured detections.

    python -m ml.src.predict sample_data/example_navigation.csv --model full__random_forest --out preds.csv

Pipeline: validate -> causal features (same code as training) -> model score + independent
rule checks -> threshold -> optional alert persistence -> hybrid status with reasons.

Status meanings (kept deliberately conservative):
  normal              no model alert, no rule alert
  suspicious          a rule fired and/or an UNSUPERVISED detector flagged it. Unusual, but ordinary GPS
                      errors (multipath, dropouts, poor visibility) look like this too.
  potential_spoofing  a SUPERVISED model, trained on labelled (simulated) spoofing, flagged it. Still not proof.

`evidence` says whether the model and the physical-plausibility rules agree.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import load_manifest, load_model
from .config import ARTIFACTS_DIR, FEATURE_SETS
from .feature_engineering import compute_features
from .rules import RuleConfig, apply_rules
from .schema import ValidationError, available_feature_sets, read_csv_safely, validate_navigation_frame

FEATURE_DESCRIPTIONS = {
    "dt": "time since previous valid fix", "step_dist": "distance moved since previous fix",
    "derived_speed": "speed implied by consecutive positions", "accel": "change in implied speed",
    "heading_change": "change in direction of travel", "pred_err": "gap between position and constant-velocity prediction",
    "roll_speed_mean": "recent mean implied speed", "roll_speed_std": "recent variability of implied speed",
    "roll_step_max": "largest recent step", "roll_hdg_std": "recent heading variability",
    "roll_pred_err_max": "largest recent prediction gap", "invalid_obs": "missing or invalid coordinates",
    "vertical_speed": "climb/descent rate", "alt_jump": "size of altitude change",
    "speed_mps": "receiver-reported speed", "speed_residual": "implied minus reported speed",
    "abs_speed_residual": "size of implied-vs-reported speed gap", "roll_resid_mean": "recent mean speed gap",
    "roll_absresid_mean": "recent mean absolute speed gap",
    "disp_excess_60": "displacement over 60 s beyond what reported speed allows",
    "disp_ratio_60": "displacement over 60 s relative to reported travel",
    "sat_count": "satellites used", "d_sat": "change in satellite count", "cn0_dbhz": "signal strength (C/N0)",
    "d_cn0": "change in signal strength", "cn0_minus_baseline": "signal strength vs the flight's early baseline",
    "roll_cn0_std": "recent signal-strength variability",
}

BASE_CAVEATS = [
    "Models were trained on SIMULATED data; scores show resemblance to simulated attack patterns, not proof of spoofing.",
    "A single alert cannot distinguish spoofing from multipath, receiver faults or poor satellite visibility.",
]


@dataclass
class ScoredFrame:
    """Everything needed to (re)apply thresholds cheaply, e.g. from a dashboard slider."""
    data: pd.DataFrame
    features: pd.DataFrame
    rules: pd.DataFrame
    score: np.ndarray
    model_key: str
    feature_set: str
    kind: str
    default_threshold: float
    bundle_features: list[str]
    baseline: dict[str, dict[str, float]]
    report: dict[str, Any] = field(default_factory=dict)


def list_models(artifacts_dir: Path | None = None) -> list[dict[str, Any]]:
    """Models in the manifest with their feature set and kind (for UI pickers)."""
    manifest = load_manifest(artifacts_dir)
    out = []
    for key in sorted(manifest["models"]):
        fs, name = key.split("__", 1)
        out.append({"key": key, "feature_set": fs, "model": name,
                    "kind": "unsupervised_anomaly" if name == "isolation_forest" else "supervised"})
    return out


def available_models(df_clean: pd.DataFrame, artifacts_dir: Path | None = None) -> list[dict[str, Any]]:
    ok = set(available_feature_sets(df_clean))
    return [m for m in list_models(artifacts_dir) if m["feature_set"] in ok]


def default_model_key(artifacts_dir: Path | None = None) -> str:
    return load_manifest(artifacts_dir)["selected"]["artifact"]


def score_frame(df: pd.DataFrame, model_key: str | None = None, artifacts_dir: Path | None = None,
                rule_cfg: RuleConfig | None = None) -> ScoredFrame:
    """Validate, featurise and score. Raises ValidationError with an actionable message."""
    artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    model_key = model_key or default_model_key(artifacts_dir)
    clean, report = validate_navigation_frame(df)
    bundle = load_model(model_key, artifacts_dir)  # checksum-verified, trusted artifacts only
    fs = bundle["feature_set"]
    have = available_feature_sets(clean)
    if fs not in have:
        need = FEATURE_SETS[fs]["requires"]
        missing = [c for c in need if c not in clean.columns or clean[c].notna().mean() <= 0.5]
        raise ValidationError(
            f"Model '{model_key}' needs the '{fs}' feature set, which requires column(s): {', '.join(need)}. "
            f"Missing or mostly empty: {', '.join(missing)}. Choose a model whose feature set is available "
            f"({', '.join(have) or 'none'}).")
    feats = compute_features(clean)
    X = feats[bundle["features"]].to_numpy(float)
    pipe = bundle["pipeline"]
    score = pipe.predict_proba(X)[:, 1] if bundle["kind"] == "supervised" else -pipe.decision_function(X)
    report["feature_set_used"] = fs
    return ScoredFrame(clean, feats, apply_rules(feats, rule_cfg), score, model_key, fs, bundle["kind"],
                       float(bundle["threshold"]), list(bundle["features"]), bundle["baseline"], report)


def run_lengths(flag: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Length of the current run of consecutive True values, reset at each trajectory."""
    s = pd.Series(flag.astype(int))
    g = pd.Series(groups)
    seg = (s == 0).groupby(g).cumsum()
    return s.groupby([g, seg]).cumsum().to_numpy()


def _top_features(sf: ScoredFrame, rows: np.ndarray, k: int = 3) -> list[str]:
    """For flagged rows: the features deviating most from the normal-flight median (robust z)."""
    cols = sf.bundle_features
    med = np.array([sf.baseline["median"][c] for c in cols])
    mad = np.array([sf.baseline["mad"][c] for c in cols]) * 1.4826
    X = sf.features.loc[rows, cols].to_numpy(float)
    z = np.abs(X - med) / mad
    z = np.nan_to_num(z, nan=0.0)
    out = []
    for r in z:
        idx = np.argsort(-r)[:k]
        out.append("; ".join(f"{cols[i]} ({FEATURE_DESCRIPTIONS.get(cols[i], cols[i])}, {r[i]:.0f}x typical spread)"
                             for i in idx if r[i] >= 3.0))
    return out


def decide(sf: ScoredFrame, threshold: float | None = None, min_consecutive: int = 1) -> pd.DataFrame:
    """Apply a threshold and optional alert persistence; build the hybrid result table."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")
    thr = sf.default_threshold if threshold is None else float(threshold)
    raw = np.nan_to_num(sf.score, nan=-np.inf) >= thr
    model_alert = run_lengths(raw, sf.data["trajectory_id"].to_numpy()) >= min_consecutive
    rule_alert = sf.rules["rule_alert"].to_numpy()
    supervised = sf.kind == "supervised"

    status = np.where(model_alert & supervised, "potential_spoofing",
                      np.where(model_alert | rule_alert, "suspicious", "normal"))
    evidence = np.select(
        [model_alert & rule_alert, model_alert & ~rule_alert, ~model_alert & rule_alert],
        ["model and physical rules agree", "model only (no rule fired)", "rules only (model did not flag)"],
        default="")
    d = sf.data
    out = pd.DataFrame({
        "trajectory_id": d["trajectory_id"], "timestamp": d["timestamp"], "lat": d["lat"], "lon": d["lon"],
        "score": sf.score, "threshold": thr,
        "model_alert": model_alert, "detector_type": "supervised classifier" if supervised else "unsupervised anomaly detector",
        "rule_alert": rule_alert, "rule_reasons": sf.rules["rule_reasons"].to_numpy(),
        "status": status, "evidence": evidence, "top_deviating_features": "",
    })
    flagged = np.flatnonzero(status != "normal")
    if len(flagged):
        out.iloc[flagged, out.columns.get_loc("top_deviating_features")] = _top_features(sf, sf.features.index[flagged])
    for c in ("label", "scenario"):
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def predict_frame(df: pd.DataFrame, model_key: str | None = None, threshold: float | None = None,
                  min_consecutive: int = 1, artifacts_dir: Path | None = None) -> pd.DataFrame:
    return decide(score_frame(df, model_key, artifacts_dir), threshold, min_consecutive)


def find_intervals(res: pd.DataFrame, gap_s: float = 5.0) -> pd.DataFrame:
    """Merge consecutive non-normal rows into intervals; a status change or a gap > gap_s starts a new one."""
    r = res[res["status"] != "normal"]
    if r.empty:
        return pd.DataFrame(columns=["trajectory_id", "status", "start", "end", "duration_s", "rows", "max_score", "evidence"])
    new = (r["trajectory_id"].ne(r["trajectory_id"].shift()) | r["status"].ne(r["status"].shift())
           | (r["timestamp"].diff() > gap_s))
    gid = new.cumsum()
    g = r.groupby(gid)
    return pd.DataFrame({
        "trajectory_id": g["trajectory_id"].first(), "status": g["status"].first(),
        "start": g["timestamp"].min(), "end": g["timestamp"].max(),
        "duration_s": g["timestamp"].max() - g["timestamp"].min(), "rows": g.size(),
        "max_score": g["score"].max(),
        "evidence": g["evidence"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""),
    }).reset_index(drop=True)


def summarize(res: pd.DataFrame, sf: ScoredFrame, min_consecutive: int = 1) -> dict[str, Any]:
    """Plain-language summary plus limitations that apply to this particular run."""
    counts = res["status"].value_counts().to_dict()
    caveats = list(BASE_CAVEATS)
    if sf.feature_set == "position":
        caveats.append("Position-only model: a smooth, persistent offset is hard to tell from real motion without reported speed or signal features.")
    if sf.kind != "supervised":
        caveats.append("Unsupervised detector: it flags unusual behaviour, not specifically spoofing, so results are shown as 'suspicious' only.")
    if min_consecutive > 1:
        caveats.append(f"Alerts need {min_consecutive} consecutive flagged samples, which reduces false alarms but delays detection.")
    return {
        "model": sf.model_key, "detector_type": res["detector_type"].iat[0] if len(res) else None,
        "rows": int(len(res)), "trajectories": int(res["trajectory_id"].nunique()),
        "status_counts": {k: int(counts.get(k, 0)) for k in ("normal", "suspicious", "potential_spoofing")},
        "intervals": int(len(find_intervals(res))), "threshold": float(res["threshold"].iat[0]) if len(res) else None,
        "min_consecutive": min_consecutive, "input_report": {k: v for k, v in sf.report.items() if k != "warnings"},
        "warnings": sf.report.get("warnings", []), "limitations": caveats,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a trained model on a navigation CSV")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--model", default=None, help="e.g. full__random_forest (default: selected model)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-consecutive", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    try:
        frame = read_csv_safely(a.csv.read_bytes())
        sf = score_frame(frame, a.model)
        res = decide(sf, a.threshold, a.min_consecutive)
    except (ValidationError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
    if a.out:
        res.to_csv(a.out, index=False)
    print(json.dumps(summarize(res, sf, a.min_consecutive), indent=2))
    print(find_intervals(res).head(20).to_string(index=False))
