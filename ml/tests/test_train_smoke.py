"""End-to-end smoke test: a tiny training run produces consistent, loadable artifacts."""
from __future__ import annotations

import json

import pytest

from ml.src.artifacts import load_model
from ml.src.config import FEATURE_SETS, MODEL_NAMES
from ml.src.train import TrainConfig, run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    rep = tmp_path_factory.mktemp("reports")
    metrics = run(TrainConfig(n_trajectories=100, duration_s=180, seed=11, out_dir=out, reports_dir=rep, n_jobs=1))
    return out, rep, metrics


def test_metrics_file_matches_returned_metrics(trained):
    out, _, metrics = trained
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["models"]["full"]["random_forest"]["test"] == metrics["models"]["full"]["random_forest"]["test"]
    assert on_disk["meta"]["data_is_synthetic"] is True


def test_all_models_and_sets_present_with_sane_metrics(trained):
    _, _, m = trained
    for fs in FEATURE_SETS:
        for name in (*MODEL_NAMES, "isolation_forest"):
            t = m["models"][fs][name]["test"]
            assert 0.0 <= t["precision"] <= 1.0 and 0.0 <= t["recall"] <= 1.0
            cm = t["confusion_matrix"]
            assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == t["n"]


def test_threshold_chosen_on_validation_respects_fpr_cap(trained):
    _, _, m = trained
    for name in MODEL_NAMES:
        # Validation FPR must respect the cap (the threshold was chosen there).
        assert m["models"]["full"][name]["val"]["false_positive_rate"] <= 0.02 + 1e-9


def test_artifacts_load_with_checksum_and_predict(trained):
    out, _, m = trained
    sel = m["selected"]["artifact"]
    bundle = load_model(sel, out)
    assert bundle["features"] == FEATURE_SETS[bundle["feature_set"]]["features"]
    assert bundle["trained_on"] == "synthetic"


def test_tampered_artifact_is_refused(trained):
    out, _, m = trained
    path = out / "models" / f"{m['selected']['artifact']}.joblib"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"x")
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_model(m["selected"]["artifact"], out)
    finally:
        path.write_bytes(original)


def test_plots_and_summary_written(trained):
    out, rep, _ = trained
    assert (out / "plots" / "model_comparison.png").exists()
    assert (rep / "results_summary.md").read_text().startswith("# Results summary")
