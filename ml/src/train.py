"""Training and evaluation pipeline.

    python -m ml.src.train                       # full run, writes ./artifacts
    python -m ml.src.train --n-trajectories 100  # quicker run

Leakage controls: the data are split BY TRAJECTORY (60/20/20, stratified by scenario), so
no flight contributes rows to more than one split; hyper-parameters and decision
thresholds are chosen on the validation split only; the test split is used once for the
reported numbers. Feature computation is causal (see feature_engineering.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import evaluate as ev
from .artifacts import save_model, write_manifest
from .config import (
    ARTIFACTS_DIR, ATTACK_SCENARIOS, DEFAULT_SEED, FEATURE_SETS, MODEL_NAMES, REPORTS_DIR,
)
from .feature_engineering import compute_features, select_features
from .rules import RuleConfig, apply_rules, config_dict
from .schema import validate_navigation_frame
from .synthetic_data import GeneratorConfig, generate_dataset

log = logging.getLogger("train")

MAX_VAL_FPR = 0.02  # operating point: at most 2% of non-spoofed validation rows may alert

GRIDS: dict[str, list[dict]] = {
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "decision_tree": [{"max_depth": d} for d in (4, 6, 8, 12)],
    "random_forest": [{"min_samples_leaf": m} for m in (5, 20, 50)],
    "gaussian_nb": [{}],
}


@dataclass
class TrainConfig:
    n_trajectories: int = 400
    duration_s: int = 300
    seed: int = DEFAULT_SEED
    out_dir: Path = ARTIFACTS_DIR
    reports_dir: Path = REPORTS_DIR
    n_jobs: int = -1


def split_trajectories(df: pd.DataFrame, seed: int) -> dict[str, set[str]]:
    """Group-aware split: whole trajectories go to exactly one of train/val/test."""
    traj = df.groupby("trajectory_id", sort=False)["scenario"].first()
    ids, sc = traj.index.to_numpy(), traj.to_numpy()
    tr, rest, _, rest_sc = train_test_split(ids, sc, test_size=0.4, stratify=sc, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest_sc, random_state=seed)
    return {"train": set(tr), "val": set(va), "test": set(te)}


def make_model(name: str, params: dict, seed: int, n_jobs: int = -1) -> Pipeline:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "logistic_regression":
        est = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, **params)
        return Pipeline([("impute", imp), ("scale", StandardScaler()), ("model", est)])
    if name == "decision_tree":
        est = DecisionTreeClassifier(min_samples_leaf=20, class_weight="balanced", random_state=seed, **params)
        return Pipeline([("impute", imp), ("model", est)])
    if name == "random_forest":
        est = RandomForestClassifier(n_estimators=100, max_depth=14, class_weight="balanced_subsample",
                                     n_jobs=n_jobs, random_state=seed, **params)
        return Pipeline([("impute", imp), ("model", est)])
    if name == "gaussian_nb":
        # NB assumes conditionally independent, roughly Gaussian features. Ours are neither
        # (rolling features overlap, speed features are heavy-tailed); it is included as a
        # baseline and its weaker results are expected.
        return Pipeline([("impute", imp), ("model", GaussianNB(**params))])
    raise ValueError(name)


def _score(pipe: Pipeline, X: np.ndarray, kind: str) -> np.ndarray:
    return pipe.predict_proba(X)[:, 1] if kind == "supervised" else -pipe.decision_function(X)


def _baseline_stats(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Median/MAD of each feature on clean training rows, used for 'what deviates' explanations."""
    med = X.median()
    mad = (X - med).abs().median().replace(0, np.nan)
    return {"median": med.fillna(0).to_dict(), "mad": mad.fillna(1.0).to_dict()}


def _evaluate_scores(scores: dict[str, np.ndarray], frames: dict[str, pd.DataFrame], thr: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        out[split] = ev.classification_metrics(frames[split]["label"].to_numpy(), scores[split], thr)
    pred_test = (scores["test"] >= thr).astype(int)
    out["per_scenario_test"] = ev.per_scenario(frames["test"], pred_test)
    out["events_test"] = ev.event_metrics(frames["test"], pred_test)
    return out


def run(cfg: TrainConfig) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(cfg.out_dir)
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("generating synthetic data (seed=%d)", cfg.seed)
    raw = generate_dataset(GeneratorConfig(n_trajectories=cfg.n_trajectories, duration_s=cfg.duration_s, seed=cfg.seed))
    clean, vrep = validate_navigation_frame(raw)
    feats = compute_features(clean)
    rules = apply_rules(feats)

    assign = split_trajectories(clean, cfg.seed)
    masks = {s: clean["trajectory_id"].isin(ids).to_numpy() for s, ids in assign.items()}
    assert not (assign["train"] & assign["val"] or assign["train"] & assign["test"] or assign["val"] & assign["test"])
    frames = {s: clean.loc[m].reset_index(drop=True) for s, m in masks.items()}
    y = {s: frames[s]["label"].to_numpy() for s in frames}
    normal_train = masks["train"] & (clean["scenario"] == "normal").to_numpy()

    results: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}
    test_scores: dict[str, dict[str, np.ndarray]] = {}
    best_params: dict[str, dict[str, dict]] = {}
    importances: dict[str, dict[str, np.ndarray]] = {}

    for fs, spec in FEATURE_SETS.items():
        Xall = select_features(feats, fs)
        X = {s: Xall.loc[m].to_numpy(float) for s, m in masks.items()}
        results[fs], test_scores[fs], best_params[fs], importances[fs] = {}, {}, {}, {}
        baseline = _baseline_stats(Xall.loc[normal_train])

        for name in MODEL_NAMES:
            # --- tune on validation PR-AUC only
            best, best_ap = None, -1.0
            for params in GRIDS[name]:
                pipe = make_model(name, params, cfg.seed, cfg.n_jobs)
                t = time.time()
                pipe.fit(X["train"], y["train"])
                fit_s = time.time() - t
                ap = ev.classification_metrics(y["val"], _score(pipe, X["val"], "supervised"), 0.5)["pr_auc"] or 0.0
                if ap > best_ap:
                    best, best_ap, best_fit = (pipe, params), ap, fit_s
            pipe, params = best
            scores = {s: _score(pipe, X[s], "supervised") for s in X}
            thr = ev.choose_threshold_fpr(y["val"], scores["val"], MAX_VAL_FPR)
            entry = _evaluate_scores(scores, frames, thr)
            t = time.time()
            _score(pipe, X["test"], "supervised")
            entry.update({
                "kind": "supervised", "params": params, "threshold": thr,
                "threshold_source": f"highest recall with false-positive rate <= {MAX_VAL_FPR:.0%} on the validation split",
                "calibration_test": ev.calibration(y["test"], scores["test"]),
                "timing": {"fit_s": best_fit, "predict_ms_per_1000_rows": (time.time() - t) / len(y["test"]) * 1e6},
            })
            bundle = {
                "pipeline": pipe, "feature_set": fs, "features": spec["features"], "model_name": name,
                "kind": "supervised", "threshold": thr, "baseline": baseline, "seed": cfg.seed,
                "sklearn_version": sklearn.__version__, "trained_on": "synthetic",
            }
            entry["artifact"] = save_model(bundle, f"{fs}__{name}", out_dir / "models")
            manifest[f"{fs}__{name}"] = entry["artifact"]
            results[fs][name] = entry
            test_scores[fs][name] = scores["test"]
            best_params[fs][name] = params
            est = pipe.named_steps["model"]
            if hasattr(est, "feature_importances_"):
                importances[fs][name] = est.feature_importances_
            log.info("%-9s %-20s val PR-AUC %.3f | test F1 %.3f recall %.3f precision %.3f",
                     fs, name, best_ap, entry["test"]["f1"], entry["test"]["recall"], entry["test"]["precision"])

        # --- unsupervised anomaly detector: trained on normal flights only, no labels used
        iso = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("model", IsolationForest(n_estimators=200, random_state=cfg.seed, n_jobs=cfg.n_jobs))])
        t = time.time()
        iso.fit(Xall.loc[normal_train].to_numpy(float))
        fit_s = time.time() - t
        sc = {s: _score(iso, X[s], "unsupervised") for s in X}
        ref = _score(iso, Xall.loc[normal_train].to_numpy(float), "unsupervised")
        thr = float(np.quantile(ref, 0.99))
        entry = _evaluate_scores(sc, frames, thr)
        entry.update({"kind": "unsupervised_anomaly", "params": {"n_estimators": 200}, "threshold": thr,
                      "threshold_source": "99th percentile of anomaly scores on normal training flights (no labels)",
                      "timing": {"fit_s": fit_s}})
        bundle = {"pipeline": iso, "feature_set": fs, "features": spec["features"], "model_name": "isolation_forest",
                  "kind": "unsupervised_anomaly", "threshold": thr, "baseline": baseline, "seed": cfg.seed,
                  "sklearn_version": sklearn.__version__, "trained_on": "synthetic"}
        entry["artifact"] = save_model(bundle, f"{fs}__isolation_forest", out_dir / "models")
        manifest[f"{fs}__isolation_forest"] = entry["artifact"]
        results[fs]["isolation_forest"] = entry
        test_scores[fs]["isolation_forest"] = sc["test"]
        log.info("%-9s %-20s test F1 %.3f recall %.3f precision %.3f", fs, "isolation_forest",
                 entry["test"]["f1"], entry["test"]["recall"], entry["test"]["precision"])

    # --- rule-based baseline (not ML)
    rule_score = {s: rules.loc[m, "rule_alert"].to_numpy(float) for s, m in masks.items()}
    rules_res = _evaluate_scores(rule_score, frames, 0.5)
    rules_res.update({"kind": "rules", "config": config_dict(RuleConfig())})
    log.info("rules                      test F1 %.3f recall %.3f precision %.3f",
             rules_res["test"]["f1"], rules_res["test"]["recall"], rules_res["test"]["precision"])

    # --- leave-one-attack-type-out generalisation (random forest)
    generalization: dict[str, Any] = {}
    for fs in ("kinematic", "full"):
        Xall = select_features(feats, fs)
        generalization[fs] = {}
        for att in ATTACK_SCENARIOS:
            keep = (clean["scenario"] != att).to_numpy()
            tr_m, va_m = masks["train"] & keep, masks["val"] & keep
            pipe = make_model("random_forest", best_params[fs]["random_forest"], cfg.seed, cfg.n_jobs)
            pipe.fit(Xall.loc[tr_m].to_numpy(float), clean.loc[tr_m, "label"].to_numpy())
            thr = ev.choose_threshold_fpr(clean.loc[va_m, "label"].to_numpy(),
                                          _score(pipe, Xall.loc[va_m].to_numpy(float), "supervised"), MAX_VAL_FPR)
            te = frames["test"]
            pt = (_score(pipe, select_features(feats, fs).loc[masks["test"]].to_numpy(float), "supervised") >= thr)
            on_att = (te["scenario"] == att) & (te["label"] == 1)
            non_atk = ~te["scenario"].isin(ATTACK_SCENARIOS)
            generalization[fs][att] = {
                "recall_held_out": float(pt[on_att.to_numpy()].mean()),
                "recall_seen_in_training": results[fs]["random_forest"]["per_scenario_test"][att]["recall"],
                "false_positive_rate_non_attack": float(pt[non_atk.to_numpy()].mean()),
            }
        log.info("generalisation (%s): %s", fs, {a: round(v["recall_held_out"], 2) for a, v in generalization[fs].items()})

    # --- selection: best validation PR-AUC among supervised models on the richest feature set
    sup = {n: r for n, r in results["full"].items() if r["kind"] == "supervised"}
    sel_name = max(sup, key=lambda n: sup[n]["val"]["pr_auc"] or 0.0)
    selected = {"feature_set": "full", "model": sel_name, "selected_by": "validation PR-AUC",
                "artifact": f"full__{sel_name}"}

    # --- plots
    ev.plot_model_comparison(results, plots / "model_comparison.png")
    for fs in results:
        sup_fs = {n: r for n, r in results[fs].items() if r["kind"] == "supervised"}
        ev.plot_confusion_matrices(sup_fs, fs, plots / f"confusion_{fs}.png")
        ev.plot_pr_curves({n: (y["test"], test_scores[fs][n]) for n in results[fs]}, fs, plots / f"pr_curves_{fs}.png")
        if "random_forest" in importances[fs]:
            ev.plot_importance(FEATURE_SETS[fs]["features"], importances[fs]["random_forest"],
                               f"Random forest feature importance ({fs})", plots / f"importance_rf_{fs}.png")
    ev.plot_generalization(generalization, plots / "generalization.png")
    te = frames["test"]
    for scenario in ("drift", "jump"):
        ids = te.loc[te["scenario"] == scenario, "trajectory_id"].unique()
        if len(ids):
            m = (te["trajectory_id"] == ids[0]).to_numpy()
            ev.plot_example_timeline(
                te.loc[m], test_scores["full"][sel_name][m], results["full"][sel_name]["threshold"],
                rules.loc[masks["test"], "rule_alert"].to_numpy()[m],
                f"Test trajectory, scenario: {scenario} (SIMULATED data), model: {sel_name}", plots / f"timeline_{scenario}.png")

    meta = {
        "data_is_synthetic": True,
        "disclaimer": "Models were trained and evaluated on simulated navigation data. Results show the ability "
                      "to recognise the simulated patterns, not real-world spoofing detection performance.",
        "seed": cfg.seed, "n_trajectories": cfg.n_trajectories, "duration_s": cfg.duration_s,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "validation_report": {k: v for k, v in vrep.items() if k != "warnings"},
        "runtime_s": time.time() - t0,
    }
    split_info = {s: {"trajectories": len(assign[s]), "rows": int(masks[s].sum()),
                      "positive_rate": float(y[s].mean())} for s in assign}
    scenario_counts = {s: clean.loc[masks[s]].groupby("scenario")["trajectory_id"].nunique().to_dict() for s in masks}
    metrics = {"meta": meta, "split": split_info, "trajectories_by_scenario": scenario_counts,
               "feature_sets": {k: {"features": v["features"], "requires": v["requires"]} for k, v in FEATURE_SETS.items()},
               "models": results, "rules": rules_res, "generalization": generalization, "selected": selected}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_manifest(manifest, {"selected": selected, "seed": cfg.seed, "trained_on": "synthetic"}, out_dir)
    write_results_summary(metrics, Path(cfg.reports_dir) / "results_summary.md")
    log.info("done in %.1fs; artifacts in %s", time.time() - t0, out_dir)
    return metrics


def write_results_summary(m: dict[str, Any], path: Path) -> None:
    """Markdown tables generated from metrics.json (so README numbers can be pasted, not typed)."""
    lines = ["# Results summary (generated by `python -m ml.src.train`)", "",
             f"> {m['meta']['disclaimer']}", "",
             f"Seed {m['meta']['seed']}; {m['meta']['n_trajectories']} simulated trajectories; "
             f"split by trajectory: " + ", ".join(f"{k} {v['trajectories']}" for k, v in m["split"].items()), ""]
    for fs, models in m["models"].items():
        lines += [f"## Feature set: {fs}", "", "| model | kind | precision | recall | F1 | accuracy | PR-AUC | FPR |", "|---|---|---|---|---|---|---|---|"]
        for name, r in models.items():
            t = r["test"]
            pa = f"{t['pr_auc']:.3f}" if t["pr_auc"] is not None else "n/a"
            lines.append(f"| {name} | {r['kind']} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | {t['accuracy']:.3f} | {pa} | {t['false_positive_rate']:.3f} |")
        lines.append("")
    t = m["rules"]["test"]
    lines += ["## Rule-based baseline (not ML), test split", "",
              f"precision {t['precision']:.3f}, recall {t['recall']:.3f}, F1 {t['f1']:.3f}, FPR {t['false_positive_rate']:.3f}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate spoofing-detection models on synthetic data")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-trajectories", type=int, default=400)
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(TrainConfig(a.n_trajectories, a.duration, a.seed, a.out_dir, REPORTS_DIR, a.n_jobs))
