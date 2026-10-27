"""Causal feature engineering shared by training and inference.

Every feature for row i uses only rows <= i of the same trajectory, so nothing from the
future (or from the label columns) can leak into a prediction. Great-circle formulas are
used throughout; coordinates are never treated as Cartesian.

Measured columns: timestamp, lat, lon, alt, speed_mps, sat_count, cn0_dbhz.
Everything else below is derived.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEAD_RECKONING_WINDOW, FEATURE_SETS, ROLLING_WINDOW
from .geo import bearing_deg, destination, haversine_m, wrap_angle_deg

ALL_FEATURES = FEATURE_SETS["full"]["features"]


def _prev_index(mask: np.ndarray) -> np.ndarray:
    """For each i, the index of the last True in mask strictly before i (or -1)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    prev = np.empty(n, dtype=int)
    prev[0] = -1
    prev[1:] = last[:-1]
    return prev


def _col(g: pd.DataFrame, name: str) -> np.ndarray:
    return g[name].to_numpy(float) if name in g.columns else np.full(len(g), np.nan)


def _roll(x: np.ndarray, fn: str, window: int) -> np.ndarray:
    r = pd.Series(x).rolling(window, min_periods=1)
    return getattr(r, fn)().to_numpy()


def _trajectory_features(g: pd.DataFrame, window: int) -> pd.DataFrame:
    n = len(g)
    t, lat, lon = _col(g, "timestamp"), _col(g, "lat"), _col(g, "lon")
    alt, spd = _col(g, "alt"), _col(g, "speed_mps")
    sat, cn0 = _col(g, "sat_count"), _col(g, "cn0_dbhz")

    valid = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
    prev = _prev_index(valid)
    p = np.where(prev >= 0, prev, 0)
    has_prev = valid & (prev >= 0)
    dt = np.where(has_prev, t - t[p], np.nan)
    ok = has_prev & (dt > 0)

    dist = np.where(ok, haversine_m(lat[p], lon[p], lat, lon), np.nan)
    speed = np.where(ok, dist / np.where(ok, dt, 1.0), np.nan)
    # Bearing is undefined for (near-)stationary steps, so it is left NaN there.
    brg = np.where(ok & (dist > 0.5), bearing_deg(lat[p], lon[p], lat, lon), np.nan)

    pb = _prev_index(np.isfinite(brg))
    pbi = np.where(pb >= 0, pb, 0)
    heading_change = np.where(np.isfinite(brg) & (pb >= 0), wrap_angle_deg(brg - brg[pbi]), np.nan)

    ps = _prev_index(np.isfinite(speed))
    psi = np.where(ps >= 0, ps, 0)
    dts = t - t[psi]
    accel = np.where(np.isfinite(speed) & (ps >= 0) & (dts > 0), (speed - speed[psi]) / np.where(dts > 0, dts, 1.0), np.nan)

    # Constant-velocity prediction from the previous valid observation and its own velocity.
    v_p, b_p = speed[p], brg[p]
    can_pred = ok & np.isfinite(v_p) & np.isfinite(b_p)
    with np.errstate(invalid="ignore"):
        plat, plon = destination(lat[p], lon[p], np.where(can_pred, b_p, 0.0), np.where(can_pred, v_p * dt, 0.0))
    pred_err = np.where(can_pred, haversine_m(plat, plon, lat, lon), np.nan)

    vertical = np.where(ok & np.isfinite(alt) & np.isfinite(alt[p]), (alt - alt[p]) / np.where(ok, dt, 1.0), np.nan)
    resid = speed - spd  # position-derived speed minus the receiver-reported speed

    # Dead-reckoning consistency over the last DR_WINDOW rows: the straight-line displacement
    # between two positions cannot exceed the distance the reported speed allows. A persistent
    # position offset (jump/drift) breaks this; a transient glitch that returns does not.
    latf = pd.Series(lat).ffill().to_numpy()
    lonf = pd.Series(lon).ffill().to_numpy()
    spd_ff = pd.Series(spd).ffill().fillna(0.0).to_numpy()
    dt_row = np.diff(t, prepend=t[0]) if n else t
    cum_path = np.cumsum(spd_ff * np.clip(np.nan_to_num(dt_row), 0, None))
    start = np.maximum(np.arange(n) - DEAD_RECKONING_WINDOW, 0)
    disp = haversine_m(latf[start], lonf[start], latf, lonf)
    path_rep = cum_path - cum_path[start]
    have_spd = np.isfinite(spd).any()
    disp_excess = np.where(have_spd, disp - path_rep, np.nan)
    disp_ratio = np.where(have_spd, disp / np.maximum(path_rep, 1.0), np.nan)

    d_sat = np.where(np.isfinite(sat) & np.isfinite(sat[psi]) & (prev >= 0), sat - sat[p], np.nan)
    d_cn0 = np.where(np.isfinite(cn0) & np.isfinite(cn0[p]) & (prev >= 0), cn0 - cn0[p], np.nan)
    # Causal baseline: running mean of the first 30 C/N0 samples, frozen afterwards.
    run_mean = pd.Series(cn0).expanding(min_periods=1).mean().to_numpy()
    base = np.where(np.arange(n) < 30, run_mean, run_mean[min(29, n - 1)])

    feats = {
        "dt": dt, "step_dist": dist, "derived_speed": speed, "accel": accel,
        "heading_change": heading_change, "pred_err": pred_err,
        "roll_speed_mean": _roll(speed, "mean", window),
        "roll_speed_std": _roll(speed, "std", window),
        "roll_step_max": _roll(dist, "max", window),
        "roll_hdg_std": _roll(heading_change, "std", window),
        "roll_pred_err_max": _roll(pred_err, "max", window),
        "invalid_obs": (~valid).astype(float),
        "vertical_speed": vertical, "alt_jump": np.abs(vertical),
        "speed_mps": spd, "speed_residual": resid, "abs_speed_residual": np.abs(resid),
        "roll_resid_mean": _roll(resid, "mean", window),
        "roll_absresid_mean": _roll(np.abs(resid), "mean", window),
        "disp_excess_60": disp_excess, "disp_ratio_60": disp_ratio,
        "sat_count": sat, "d_sat": d_sat, "cn0_dbhz": cn0, "d_cn0": d_cn0,
        "cn0_minus_baseline": cn0 - base, "roll_cn0_std": _roll(cn0, "std", window),
    }
    return pd.DataFrame(feats, index=g.index)[ALL_FEATURES]


def compute_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Compute all candidate features. `df` must be validated and sorted (see schema.py).

    Returns a frame with the same index as `df` and one column per feature in ALL_FEATURES.
    Columns that need data the input does not provide (e.g. cn0_dbhz) are NaN; select a
    feature set with `select_features` to use only what the data supports.
    """
    if "trajectory_id" not in df.columns:
        df = df.assign(trajectory_id="traj_0")
    parts = [_trajectory_features(g, window) for _, g in df.groupby("trajectory_id", sort=False)]
    out = pd.concat(parts).reindex(df.index)
    return out[ALL_FEATURES]


def select_features(feats: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return feats[FEATURE_SETS[feature_set]["features"]]
