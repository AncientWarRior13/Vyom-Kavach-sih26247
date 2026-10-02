"""Schema contract validator — the guard that protects the Pair 2 handoff.

Run against any dataframe/CSV/parquet that claims to follow the RadarSense
schema. Exits non-zero on any violation so it can gate CI or a demo launch.

Usage:
    python -m radarsense.validate_schema artifacts/data/radar_full.parquet
"""

import sys

import numpy as np
import pandas as pd

from radarsense import config as C

RANGES = {
    "doppler_velocity": (-50, 50),
    "rcs_estimate": (-20, 20),
    "micro_doppler_bandwidth": (0, 500),
    "micro_doppler_jitter": (0, 1),
    "range_m": (50, 5000),
    "azimuth_deg": (-180, 180),
    "altitude_m": (0, 3000),
    "acceleration": (-10, 10),
    "snr_db": (-20, -10),
}


def validate(df: pd.DataFrame, require_label: bool = True) -> list[str]:
    errors = []
    expected = C.FEATURE_COLUMNS + ([C.LABEL_COLUMN] if require_label else [])
    if list(df.columns) != expected:
        errors.append(f"column order/names mismatch:\n  got      {list(df.columns)}\n  expected {expected}")
    for col in C.FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            errors.append(f"{col}: dtype {df[col].dtype}, expected float")
        if df[col].isna().any():
            errors.append(f"{col}: contains NaN")
        if col in RANGES:
            lo, hi = RANGES[col]
            bad = (~df[col].between(lo, hi)).sum()
            if bad:
                errors.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if "velocity_variance" in df.columns and (df["velocity_variance"] < 0).any():
        errors.append("velocity_variance: negative values")
    if require_label and C.LABEL_COLUMN in df.columns:
        bad_labels = set(df[C.LABEL_COLUMN].unique()) - set(C.CLASSES)
        if bad_labels:
            errors.append(f"unknown labels: {bad_labels}")
    return errors


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(C.DATA_DIR / "radar_full.parquet")
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    errors = validate(df)
    if errors:
        print(f"SCHEMA VALIDATION FAILED for {path}:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print(f"schema OK: {path} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
