"""Effect of requiring k consecutive flagged samples before alerting.

    python -m ml.src.analyze_persistence

Regenerates the training run's test split from the recorded seed (generation and splitting are
deterministic), scores it with the selected model, and writes artifacts/persistence.json.
k=1 must reproduce the test metrics already stored in metrics.json (checked below).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import evaluate as ev
from .artifacts import load_manifest
from .config import ARTIFACTS_DIR
from .predict import decide, score_frame
from .schema import validate_navigation_frame
from .synthetic_data import GeneratorConfig, generate_dataset
from .train import split_trajectories

KS = (1, 2, 3, 5, 8, 12)


def analyze(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    manifest = load_manifest(artifacts_dir)
    metrics = json.loads((artifacts_dir / "metrics.json").read_text())
    meta, key = metrics["meta"], manifest["selected"]["artifact"]
    raw = generate_dataset(GeneratorConfig(n_trajectories=meta["n_trajectories"], duration_s=meta["duration_s"], seed=meta["seed"]))
    clean, _ = validate_navigation_frame(raw)
    ids = split_trajectories(clean, meta["seed"])["test"]
    test = clean[clean["trajectory_id"].isin(ids)].reset_index(drop=True)
    sf = score_frame(test, key, artifacts_dir)
    y = test["label"].to_numpy()

    rows = []
    for k in KS:
        res = decide(sf, min_consecutive=k)
        pred = res["model_alert"].to_numpy().astype(int)
        m = ev.classification_metrics(y, pred.astype(float), 0.5)
        e = ev.event_metrics(test, pred)
        rows.append({"min_consecutive": k, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "false_positive_rate": m["false_positive_rate"], "events": e})

    stored = metrics["models"][manifest["selected"]["feature_set"]][manifest["selected"]["model"]]["test"]
    k1 = rows[0]
    assert abs(k1["precision"] - stored["precision"]) < 1e-9 and abs(k1["recall"] - stored["recall"]) < 1e-9, \
        "k=1 does not reproduce stored test metrics"
    out = {"model": key, "data_is_synthetic": True, "split": "test", "results": rows,
           "note": "Persistence trades detection delay for fewer false alarms; k=1 equals metrics.json."}
    (artifacts_dir / "persistence.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = analyze()
    print(f"model {r['model']} (synthetic test split)")
    print("k  precision recall  f1    FPR    flights-alarmed  alerts/h(non-attack)  median delay s (jump/drift/cdrift/freeze)")
    for x in r["results"]:
        ev_ = x["events"]
        d = [ev_["attacks"][a]["median_delay_s"] for a in ("jump", "drift", "consistent_drift", "freeze")]
        print(f"{x['min_consecutive']:<2} {x['precision']:.3f}     {x['recall']:.3f}  {x['f1']:.3f} {x['false_positive_rate']:.3f}  "
              f"{ev_['non_attack_flights']['share_with_any_alert']:.2f}             {ev_['non_attack_flights']['alerts_per_hour']:.1f}                 "
              f"{[None if v is None else round(v, 1) for v in d]}")
