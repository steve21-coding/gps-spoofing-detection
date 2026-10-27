"""Input validation for navigation data. Uploaded CSVs are untrusted: everything is
coerced to numbers, range-checked and size-limited before any model sees it."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    LABEL_COLUMNS, MAX_ROWS, MAX_UPLOAD_BYTES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS,
)

NUMERIC_OPTIONAL = ("alt", "speed_mps", "sat_count", "cn0_dbhz")


class ValidationError(ValueError):
    """Raised with an actionable message when input data cannot be used."""


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Accept numeric seconds or ISO-8601 strings; return float seconds."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:  # mostly numeric -> treat as seconds
        return numeric.astype(float)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (parsed - epoch).dt.total_seconds()


def read_csv_safely(data: bytes | str | io.IOBase, max_bytes: int = MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Parse CSV bytes with a size cap. Never evaluates anything from the file."""
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = data.read()
        raw = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(raw) > max_bytes:
        raise ValidationError(f"File is {len(raw) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.")
    try:
        return pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1)
    except Exception as exc:  # noqa: BLE001 - surface any parser error as a user message
        raise ValidationError(f"Could not parse the file as CSV: {exc}") from exc


def validate_navigation_frame(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a navigation DataFrame. Returns (clean_frame, report).

    Raises ValidationError for unusable input. Recoverable problems (invalid coordinates,
    duplicates) are fixed or flagged and listed in the report instead.
    """
    if df is None or len(df) == 0:
        raise ValidationError("The file contains no rows.")
    if len(df) > max_rows:
        raise ValidationError(f"{len(df):,} rows exceeds the limit of {max_rows:,}. Please upload a smaller file.")

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Required: {', '.join(REQUIRED_COLUMNS)}. Optional: {', '.join(OPTIONAL_COLUMNS)}."
        )

    report: dict[str, Any] = {"n_rows_input": int(len(out)), "warnings": []}

    out["timestamp"] = _parse_timestamp(out["timestamp"])
    for col in ("lat", "lon") + NUMERIC_OPTIONAL:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    bad_ts = out["timestamp"].isna()
    report["n_bad_timestamps_dropped"] = int(bad_ts.sum())
    out = out.loc[~bad_ts].copy()
    if out.empty:
        raise ValidationError("No row has a parseable timestamp (numeric seconds or ISO-8601).")

    if "trajectory_id" not in out.columns:
        out["trajectory_id"] = "traj_0"
    out["trajectory_id"] = out["trajectory_id"].astype(str)

    # Invalid coordinates become NaN so they are flagged, not silently used.
    bad_coord = (
        ~np.isfinite(out["lat"]) | ~np.isfinite(out["lon"])
        | (out["lat"].abs() > 90) | (out["lon"].abs() > 180)
    )
    report["n_invalid_coordinates"] = int(bad_coord.sum())
    out.loc[bad_coord, ["lat", "lon"]] = np.nan

    dup = out.duplicated(subset=["trajectory_id", "timestamp"], keep="first")
    report["n_duplicate_timestamps_dropped"] = int(dup.sum())
    out = out.loc[~dup]

    # Sort within each trajectory; report how many rows were out of order.
    was_sorted = bool(out.groupby("trajectory_id", sort=False)["timestamp"].apply(lambda s: s.is_monotonic_increasing).all())
    report["timestamps_were_sorted"] = was_sorted
    if not was_sorted:
        report["warnings"].append("Timestamps were not in order; rows were sorted within each trajectory.")
    out = out.sort_values(["trajectory_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    report["n_rows_clean"] = int(len(out))
    report["n_trajectories"] = int(out["trajectory_id"].nunique())
    report["missing_by_column"] = {c: int(out[c].isna().sum()) for c in out.columns if c not in LABEL_COLUMNS and out[c].isna().any()}
    report["optional_columns_present"] = [c for c in NUMERIC_OPTIONAL if c in out.columns]
    if report["n_invalid_coordinates"]:
        report["warnings"].append(f"{report['n_invalid_coordinates']} row(s) had invalid coordinates and are flagged as invalid observations.")
    return out, report


def available_feature_sets(df: pd.DataFrame) -> list[str]:
    """Feature sets whose required columns are all present (and mostly non-empty)."""
    from .config import FEATURE_SETS

    ok = []
    for name, spec in FEATURE_SETS.items():
        if all(c in df.columns and df[c].notna().mean() > 0.5 for c in spec["requires"]):
            ok.append(name)
    return ok
