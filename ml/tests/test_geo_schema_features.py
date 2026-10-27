"""Unit tests for geodesy, validation, feature engineering, rules and the generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.config import ALL_SCENARIOS, ATTACK_SCENARIOS, FEATURE_SETS, LABEL_COLUMNS
from ml.src.feature_engineering import ALL_FEATURES, compute_features, select_features
from ml.src.geo import bearing_deg, destination, enu_to_latlon, haversine_m, wrap_angle_deg
from ml.src.rules import apply_rules
from ml.src.schema import ValidationError, validate_navigation_frame
from ml.src.synthetic_data import GeneratorConfig, generate_dataset
from ml.src.train import split_trajectories


# ----------------------------------------------------------------------- geo
def test_haversine_one_degree_of_longitude_at_equator():
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_194.9, rel=1e-4)


def test_haversine_symmetric_and_zero():
    assert haversine_m(10.0, 20.0, 10.0, 20.0) == 0.0
    assert haversine_m(10, 20, 11, 22) == pytest.approx(haversine_m(11, 22, 10, 20))


def test_bearing_cardinal_directions():
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90.0, abs=1e-6)
    assert bearing_deg(0, 0, -1, 0) == pytest.approx(180.0, abs=1e-6)


def test_destination_round_trip():
    lat, lon = destination(45.0, 10.0, 60.0, 5_000.0)
    assert haversine_m(45.0, 10.0, lat, lon) == pytest.approx(5_000.0, rel=1e-6)
    assert bearing_deg(45.0, 10.0, lat, lon) == pytest.approx(60.0, abs=0.05)


def test_wrap_angle():
    assert wrap_angle_deg(190.0) == pytest.approx(-170.0)
    assert wrap_angle_deg(-190.0) == pytest.approx(170.0)


def test_longitude_scale_shrinks_with_latitude():
    # a degree of longitude is much shorter at 60N: coordinates are NOT cartesian
    assert haversine_m(60, 0, 60, 1) == pytest.approx(haversine_m(0, 0, 0, 1) * np.cos(np.radians(60)), rel=1e-3)


# -------------------------------------------------------------------- schema
def _frame(n=20, **extra):
    d = pd.DataFrame({"timestamp": np.arange(n, dtype=float), "lat": 10.0 + np.arange(n) * 1e-5, "lon": 20.0})
    for k, v in extra.items():
        d[k] = v
    return d


def test_missing_required_column_is_actionable():
    with pytest.raises(ValidationError, match="Missing required column"):
        validate_navigation_frame(_frame().drop(columns=["lon"]))


def test_empty_and_oversized_inputs_rejected():
    with pytest.raises(ValidationError):
        validate_navigation_frame(pd.DataFrame())
    with pytest.raises(ValidationError, match="exceeds the limit"):
        validate_navigation_frame(_frame(50), max_rows=10)


def test_invalid_coordinates_flagged_not_used():
    d = _frame()
    d["lon"] = d["lon"].astype(object)  # a messy CSV column mixing numbers and text
    d.loc[3, "lat"] = 123.0
    d.loc[4, "lon"] = "abc"
    clean, rep = validate_navigation_frame(d)
    assert rep["n_invalid_coordinates"] == 2
    assert clean["lat"].isna().sum() == 2
    assert compute_features(clean)["invalid_obs"].sum() == 2


def test_duplicates_dropped_unsorted_sorted_iso_timestamps_parsed():
    d = _frame(5)
    d = pd.concat([d, d.iloc[[2]]]).sample(frac=1, random_state=1)
    clean, rep = validate_navigation_frame(d)
    assert rep["n_duplicate_timestamps_dropped"] == 1
    assert clean["timestamp"].is_monotonic_increasing
    iso = _frame(3)
    iso["timestamp"] = ["2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z", "2025-01-01T00:00:02Z"]
    clean, _ = validate_navigation_frame(iso)
    assert np.diff(clean["timestamp"]).tolist() == [1.0, 1.0]


# ------------------------------------------------------------------ features
def _straight(n=40, speed=10.0):
    """Constant 10 m/s eastward flight at the equator, reported speed matches."""
    lat, lon = enu_to_latlon(0.0, 0.0, np.arange(n) * speed, np.zeros(n))
    return pd.DataFrame({"trajectory_id": "t", "timestamp": np.arange(n, dtype=float), "lat": lat, "lon": lon,
                         "alt": 50.0, "speed_mps": speed, "sat_count": 10.0, "cn0_dbhz": 42.0})


def test_constant_velocity_features():
    f = compute_features(_straight())
    assert np.isnan(f["derived_speed"].iloc[0])  # no previous observation yet
    assert f["derived_speed"].iloc[1:].to_numpy() == pytest.approx(10.0, rel=1e-3)
    assert f["accel"].iloc[3:].abs().max() < 1e-3
    assert f["heading_change"].iloc[3:].abs().max() < 1e-3
    assert f["pred_err"].iloc[3:].max() < 0.05
    assert f["speed_residual"].iloc[1:].abs().max() < 0.05
    assert f["disp_excess_60"].iloc[1:].abs().max() < 1.0


def test_position_jump_shows_up():
    d = _straight()
    d.loc[20:, "lon"] += 0.01  # ~1.1 km eastward step
    f = compute_features(d)
    assert f["step_dist"].iloc[20] > 1000
    assert f["pred_err"].iloc[20] > 1000
    assert f["disp_excess_60"].iloc[25] > 500  # persists: displacement exceeds what speed allows


def test_features_are_causal():
    d = _straight()
    base = compute_features(d)
    changed = d.copy()
    changed.loc[30:, "lat"] += 0.02  # alter only the future
    alt = compute_features(changed)
    pd.testing.assert_frame_equal(base.iloc[:30], alt.iloc[:30])


def test_missing_optional_columns_give_nan_not_crash():
    f = compute_features(_straight().drop(columns=["speed_mps", "cn0_dbhz", "sat_count", "alt"]))
    assert f["speed_residual"].isna().all() and f["cn0_dbhz"].isna().all()
    assert select_features(f, "position").iloc[5:].notna().mean().min() > 0.5


def test_no_label_columns_in_any_feature_set():
    for spec in FEATURE_SETS.values():
        assert not set(spec["features"]) & set(LABEL_COLUMNS)
    assert not set(ALL_FEATURES) & set(LABEL_COLUMNS)


def test_dropout_rows_flagged_and_recovered():
    d = _straight()
    d.loc[10:14, ["lat", "lon"]] = np.nan
    f = compute_features(d)
    assert f["invalid_obs"].iloc[10:15].eq(1).all()
    assert f["dt"].iloc[15] == pytest.approx(6.0)  # measured from the last valid observation
    assert f["derived_speed"].iloc[15] == pytest.approx(10.0, rel=1e-3)


# --------------------------------------------------------------------- rules
def test_rules_quiet_on_clean_flight_and_fire_on_jump():
    d = _straight()
    assert not apply_rules(compute_features(d))["rule_alert"].any()
    d.loc[20:, "lon"] += 0.01
    r = apply_rules(compute_features(d))
    assert r["rule_position_jump"].iloc[20]
    assert "jump" in r["rule_reasons"].iloc[20].lower()


# ----------------------------------------------------------------- generator
def test_generator_is_deterministic_and_seed_sensitive():
    cfg = GeneratorConfig(n_trajectories=16, duration_s=160, seed=7)
    a, b = generate_dataset(cfg), generate_dataset(cfg)
    pd.testing.assert_frame_equal(a, b)
    c = generate_dataset(GeneratorConfig(n_trajectories=16, duration_s=160, seed=8))
    assert not a["lat"].equals(c["lat"])


def test_generator_labels_follow_scenarios():
    d = generate_dataset(GeneratorConfig(n_trajectories=len(ALL_SCENARIOS) * 2, duration_s=160, seed=3,
                                         scenario_weights={k: 1.0 for k in ALL_SCENARIOS}))
    assert set(d["scenario"]) == set(ALL_SCENARIOS)
    benign = d[~d["scenario"].isin(ATTACK_SCENARIOS)]
    assert benign["label"].eq(0).all()  # glitches/dropouts/low-sat are NOT spoofing
    for _, g in d[d["scenario"].isin(ATTACK_SCENARIOS)].groupby("trajectory_id"):
        onset = g["attack_onset_ts"].iloc[0]
        assert (g.loc[g["timestamp"] < onset, "label"] == 0).all()
        assert (g.loc[g["timestamp"] >= onset, "label"] == 1).all()


def test_generated_data_passes_validation():
    d = generate_dataset(GeneratorConfig(n_trajectories=8, duration_s=160, seed=1))
    clean, rep = validate_navigation_frame(d)
    assert rep["n_rows_clean"] == len(d)


# --------------------------------------------------------------------- split
def test_split_is_by_trajectory_with_no_overlap():
    d = generate_dataset(GeneratorConfig(n_trajectories=100, duration_s=160, seed=5))
    s = split_trajectories(d, seed=5)
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"])
    assert len(s["train"] | s["val"] | s["test"]) == d["trajectory_id"].nunique()
