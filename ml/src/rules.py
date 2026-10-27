"""Rule-based physical-plausibility checks.

This is NOT machine learning. It is the interpretable baseline: thresholds are set from
physical plausibility for a small multirotor, not tuned on any evaluation data, and its
results are always reported separately from the learned models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    max_speed_mps: float = 45.0           # position-derived speed above this is implausible
    max_accel_mps2: float = 12.0          # between consecutive derived speeds
    max_step_jump_m: float = 120.0        # single-step displacement at ~1 Hz
    max_vertical_speed_mps: float = 15.0
    speed_mismatch_mps: float = 3.0       # rolling mean |derived - reported speed|
    max_gap_s: float = 30.0               # a long data gap makes other checks unreliable


RULE_DESCRIPTIONS = {
    "rule_invalid_obs": "Missing or invalid coordinates/timestamp",
    "rule_excess_speed": "Position-derived speed exceeds physical limit",
    "rule_excess_accel": "Acceleration exceeds physical limit",
    "rule_position_jump": "Single-step position jump exceeds limit",
    "rule_vertical_speed": "Vertical speed exceeds physical limit",
    "rule_speed_mismatch": "Position-derived speed disagrees with reported speed",
    "rule_long_gap": "Long gap between observations",
}


def apply_rules(feats: pd.DataFrame, cfg: RuleConfig | None = None) -> pd.DataFrame:
    """Return one boolean column per rule plus `rule_alert` and a readable `rule_reasons`."""
    cfg = cfg or RuleConfig()
    f = feats
    out = pd.DataFrame(index=f.index)
    out["rule_invalid_obs"] = f["invalid_obs"] > 0
    out["rule_excess_speed"] = f["derived_speed"] > cfg.max_speed_mps
    out["rule_excess_accel"] = f["accel"].abs() > cfg.max_accel_mps2
    out["rule_position_jump"] = f["step_dist"] > cfg.max_step_jump_m
    out["rule_vertical_speed"] = f["vertical_speed"].abs() > cfg.max_vertical_speed_mps
    out["rule_speed_mismatch"] = f["roll_absresid_mean"] > cfg.speed_mismatch_mps
    out["rule_long_gap"] = f["dt"] > cfg.max_gap_s
    flags = out.fillna(False).astype(bool)  # NaN comparisons are "not triggered"
    flags["rule_alert"] = flags[list(RULE_DESCRIPTIONS)].any(axis=1)
    names = list(RULE_DESCRIPTIONS)
    flags["rule_reasons"] = [
        "; ".join(RULE_DESCRIPTIONS[n] for n, hit in zip(names, row) if hit)
        for row in flags[names].to_numpy()
    ]
    return flags


def config_dict(cfg: RuleConfig | None = None) -> dict:
    return asdict(cfg or RuleConfig())
