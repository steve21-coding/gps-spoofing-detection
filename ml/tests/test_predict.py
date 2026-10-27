"""Tests for the inference layer, using the committed trained artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.config import ARTIFACTS_DIR
from ml.src.predict import (
    available_models, decide, default_model_key, find_intervals, list_models, run_lengths,
    score_frame, summarize,
)
from ml.src.schema import ValidationError, read_csv_safely
from ml.src.synthetic_data import GeneratorConfig, generate_dataset

pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "manifest.json").exists(),
                                reason="trained artifacts not present; run `python -m ml.src.train`")


@pytest.fixture(scope="module")
def demo():
    # seed differs from the training seed, so these flights were never seen in training
    return generate_dataset(GeneratorConfig(n_trajectories=8, duration_s=200, seed=2025,
                                            scenario_weights={"normal": 1, "jump": 1, "drift": 1, "freeze": 1}))


@pytest.fixture(scope="module")
def scored(demo):
    return score_frame(demo)


def test_run_lengths_reset_on_false_and_on_new_trajectory():
    flag = np.array([1, 1, 0, 1, 1, 1, 1, 1], bool)
    grp = np.array(["a", "a", "a", "a", "a", "b", "b", "b"])
    assert run_lengths(flag, grp).tolist() == [1, 2, 0, 1, 2, 1, 2, 3]


def test_output_shape_and_status_vocabulary(demo, scored):
    res = decide(scored)
    assert len(res) == len(demo)
    assert set(res["status"]) <= {"normal", "suspicious", "potential_spoofing"}
    assert res["score"].between(0, 1).all()


def test_label_columns_do_not_influence_scores(demo, scored):
    stripped = demo.drop(columns=["label", "scenario", "attack_onset_ts"])
    np.testing.assert_allclose(score_frame(stripped).score, scored.score)


def test_threshold_extremes(scored):
    assert not decide(scored, threshold=1.5)["model_alert"].any()
    assert decide(scored, threshold=-1.0)["model_alert"].all()


def test_persistence_only_removes_alerts(scored):
    a1 = decide(scored, min_consecutive=1)["model_alert"].to_numpy()
    a3 = decide(scored, min_consecutive=3)["model_alert"].to_numpy()
    assert a3.sum() <= a1.sum() and not (a3 & ~a1).any()
    with pytest.raises(ValueError):
        decide(scored, min_consecutive=0)


def test_attacks_detected_and_normal_flights_mostly_quiet(demo, scored):
    res = decide(scored, min_consecutive=3)
    res["scenario"] = demo["scenario"].to_numpy()
    res["label"] = demo["label"].to_numpy()
    spoof = res["status"] == "potential_spoofing"
    assert spoof[(res["scenario"] == "freeze") & (res["label"] == 1)].mean() > 0.8
    assert spoof[res["scenario"] == "normal"].mean() < 0.05


def test_unsupervised_model_never_claims_potential_spoofing(demo):
    res = decide(score_frame(demo, "full__isolation_forest"), min_consecutive=1)
    assert "potential_spoofing" not in set(res["status"])
    assert (res["detector_type"] == "unsupervised anomaly detector").all()


def test_rule_only_alert_is_suspicious_not_spoofing(scored):
    res = decide(scored, threshold=1.5)  # model silenced
    assert "potential_spoofing" not in set(res["status"])
    assert (res.loc[res["status"] == "suspicious", "rule_alert"]).all()


def test_missing_columns_for_model_give_actionable_error(demo):
    pos_only = demo[["trajectory_id", "timestamp", "lat", "lon"]]
    with pytest.raises(ValidationError, match="requires column"):
        score_frame(pos_only, "full__random_forest")
    ok = decide(score_frame(pos_only, "position__random_forest"))
    assert len(ok) == len(pos_only)


def test_available_models_match_columns(demo):
    from ml.src.schema import validate_navigation_frame

    clean, _ = validate_navigation_frame(demo[["trajectory_id", "timestamp", "lat", "lon"]])
    assert {m["feature_set"] for m in available_models(clean)} == {"position"}
    assert default_model_key() in {m["key"] for m in list_models()}


def test_intervals_and_summary(demo, scored):
    res = decide(scored, min_consecutive=3)
    iv = find_intervals(res)
    assert iv["rows"].sum() == (res["status"] != "normal").sum()
    s = summarize(res, scored, 3)
    assert sum(s["status_counts"].values()) == len(res) and s["limitations"]


def test_reasons_are_given_for_flagged_rows_only(scored):
    res = decide(scored)
    assert (res.loc[res["status"] == "normal", "top_deviating_features"] == "").all()
    flagged = res[res["status"] == "potential_spoofing"]
    assert (flagged["top_deviating_features"] != "").mean() > 0.5


def test_oversized_upload_rejected():
    with pytest.raises(ValidationError, match="limit"):
        read_csv_safely(b"a,b\n" + b"1,2\n" * 10, max_bytes=8)
    with pytest.raises(ValidationError, match="parse"):
        read_csv_safely(b"\x00\x01\x02not,a\ncsv\xff" * 0 + b"")
