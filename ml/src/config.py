"""Central configuration: paths, seeds, schema constants and feature-set definitions.

Everything that training, inference, the dashboard and the website export must agree on
lives here so there is a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.environ.get("GPS_ARTIFACTS_DIR", ROOT / "artifacts"))
SAMPLE_DATA_DIR = ROOT / "sample_data"
REPORTS_DIR = ROOT / "reports"

# Random seed used for data generation, splits and models. Override with GPS_SEED.
DEFAULT_SEED = int(os.environ.get("GPS_SEED", "42"))

# ---------------------------------------------------------------- input schema
REQUIRED_COLUMNS = ("timestamp", "lat", "lon")
OPTIONAL_COLUMNS = ("alt", "speed_mps", "sat_count", "cn0_dbhz", "trajectory_id")
# Columns that exist only in labelled/synthetic data. They must never be used as features.
LABEL_COLUMNS = ("label", "scenario", "attack_onset_ts")

# Upload limits (CSV files are treated as untrusted input).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 250_000

# ------------------------------------------------------------------ scenarios
ATTACK_SCENARIOS = ("jump", "drift", "consistent_drift", "freeze")
# Benign anomalies: ordinary GPS problems. Labelled 0 (not spoofing) on purpose.
BENIGN_ANOMALIES = ("glitch", "dropout", "low_sat_noise")
ALL_SCENARIOS = ("normal",) + BENIGN_ANOMALIES + ATTACK_SCENARIOS

# ------------------------------------------------------------------- features
ROLLING_WINDOW = 10  # samples; windows are causal (current and past samples only)
DEAD_RECKONING_WINDOW = 60  # samples for the displacement-vs-reported-speed consistency check

POSITION_FEATURES = [
    "dt", "step_dist", "derived_speed", "accel", "heading_change", "pred_err",
    "roll_speed_mean", "roll_speed_std", "roll_step_max", "roll_hdg_std",
    "roll_pred_err_max", "invalid_obs",
]
KINEMATIC_FEATURES = POSITION_FEATURES + [
    "vertical_speed", "alt_jump", "speed_mps", "speed_residual",
    "abs_speed_residual", "roll_resid_mean", "roll_absresid_mean",
    "disp_excess_60", "disp_ratio_60",
]
FULL_FEATURES = KINEMATIC_FEATURES + [
    "sat_count", "d_sat", "cn0_dbhz", "d_cn0", "cn0_minus_baseline", "roll_cn0_std",
]

# name -> (feature list, columns the uploaded data must provide)
FEATURE_SETS: dict[str, dict] = {
    "position": {
        "features": POSITION_FEATURES,
        "requires": ["timestamp", "lat", "lon"],
        "description": "Timestamp and coordinates only.",
    },
    "kinematic": {
        "features": KINEMATIC_FEATURES,
        "requires": ["timestamp", "lat", "lon", "alt", "speed_mps"],
        "description": "Adds altitude and the receiver-reported ground speed.",
    },
    "full": {
        "features": FULL_FEATURES,
        "requires": ["timestamp", "lat", "lon", "alt", "speed_mps", "sat_count", "cn0_dbhz"],
        "description": "Adds satellite count and carrier-to-noise density.",
    },
}

MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest", "gaussian_nb")
