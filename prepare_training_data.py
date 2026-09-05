"""
prepare_training_data.py

Data preparation and feature engineering pipeline for ozone prediction:
1. Loads final_training_table.csv (3-month CAMS + OpenAQ data, population_density removed).
2. Prints row counts, unique stations, target non-null counts, and column lists.
3. Computes cyclical temporal encodings from valid_time:
   - hour_sin, hour_cos (24-hour cycle)
   - day_of_year_sin, day_of_year_cos (annual 365.25-day cycle)
4. Checks neighbor_o3_lagged coverage against the 10% non-null threshold:
   - If < 10%, fills missing values with each station's own most recent past go3 value.
   - Reports the percentage of rows requiring fallback.
5. Filters out rows where o3_value is null.
6. Prints final training dataset row count and verifies 0 unexpected nulls across all features.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Define the base and cyclical feature sets
BASE_FEATURES = [
    "go3",
    "no2",
    "c5h8",
    "t2m",
    "d2m",
    "sp",
    "wind_speed",
    "wind_direction_deg",
    "dist_to_coast_km",
    "elevation_mean_20km",
    "elevation_std_20km",
    "day_length_hours",
    "neighbor_o3_lagged",
]

CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

FINAL_FEATURE_LIST = BASE_FEATURES + CYCLICAL_FEATURES
TARGET_COL = "o3_value"


def load_and_prepare_data(
    input_path: str | Path = "final_training_table.csv",
    output_path: str | Path | None = None,
    force_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file.resolve()}")

    print("=" * 80)
    print(f"LOADING & AUDITING DATASET: {input_file.name}")
    print("=" * 80)

    # 1. Load data
    df = pd.read_csv(input_file)
    total_rows = len(df)
    n_unique_stations = df["station_id"].nunique()
    unique_stations = sorted(df["station_id"].unique().tolist())

    print(f"1. Total row count          : {total_rows:,}")
    print(f"   Number of unique stations: {n_unique_stations} {unique_stations}")

    # 2. Target non-null count
    target_non_null = int(df[TARGET_COL].notnull().sum())
    target_null = total_rows - target_non_null
    target_pct = (target_non_null / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"\n2. Target Column ('{TARGET_COL}') Non-Null Count:")
    print(f"   Non-null count : {target_non_null:,} / {total_rows:,} ({target_pct:.2f}%)")
    print(f"   Null count     : {target_null:,} ({100.0 - target_pct:.2f}%)")

    # 3. Full column list
    full_columns = list(df.columns)
    print(f"\n3. Full column list ({len(full_columns)} columns):")
    for i, col in enumerate(full_columns, 1):
        print(f"   {i:2d}. {col}")

    # Print defined final feature list
    print("\n" + "-" * 80)
    print(f"DEFINED FINAL FEATURE LIST ({len(FINAL_FEATURE_LIST)} features):")
    print("-" * 80)
    print("Base features (13):")
    for f in BASE_FEATURES:
        print(f"  - {f}")
    print("Cyclical temporal encodings (4):")
    for f in CYCLICAL_FEATURES:
        print(f"  - {f}")

    # 4. Cyclical encodings from valid_time
    print("\n" + "-" * 80)
    print("COMPUTING CYCLICAL TEMPORAL ENCODINGS")
    print("-" * 80)
    valid_time_dt = pd.to_datetime(df["valid_time"], format="ISO8601", utc=True)
    df["_valid_time_dt"] = valid_time_dt

    hour = valid_time_dt.dt.hour
    day_of_year = valid_time_dt.dt.dayofyear

    # 24-hour cyclical encoding
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # 365.25-day annual cyclical encoding
    df["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    print("Added cyclical features: hour_sin, hour_cos, day_of_year_sin, day_of_year_cos")
    print(df[["valid_time", "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"]].head(3).to_string(index=False))

    # 5. neighbor_o3_lagged coverage & fallback logic
    print("\n" + "-" * 80)
    print("NEIGHBOR_O3_LAGGED COVERAGE & FALLBACK AUDIT")
    print("-" * 80)
    lag_col = "neighbor_o3_lagged"
    non_null_lag = int(df[lag_col].notnull().sum())
    coverage_pct = (non_null_lag / total_rows * 100.0) if total_rows > 0 else 0.0
    print(f"Non-null count for '{lag_col}': {non_null_lag:,} / {total_rows:,} ({coverage_pct:.2f}%)")

    # Calculate station's own most recent past go3 value
    df = df.sort_values(["station_id", "_valid_time_dt"]).reset_index(drop=True)
    df["past_go3"] = df.groupby("station_id")["go3"].shift(1)
    df["past_go3"] = df["past_go3"].fillna(df["go3"])

    fallback_needed = coverage_pct < 10.0 or force_fallback

    if fallback_needed:
        missing_mask = df[lag_col].isnull()
        n_missing = int(missing_mask.sum())
        pct_needed = (n_missing / total_rows * 100.0) if total_rows > 0 else 0.0
        df[lag_col] = df[lag_col].fillna(df["past_go3"])
        print(f"[!] Fallback TRIGGERED: Coverage ({coverage_pct:.2f}%) < 10% threshold (or forced).")
        print(f"    --> {n_missing:,} rows ({pct_needed:.2f}%) filled with station's own past go3 value.")
    else:
        print(f"[+] Fallback NOT triggered: Coverage ({coverage_pct:.2f}%) is >= 10% threshold.")
        print("    --> 0 rows (0.00%) needed fallback under the global threshold rule.")

    # Per-station breakdown
    stn_stats = []
    for sid, grp in df.groupby("station_id"):
        stn_nn = grp[lag_col].notnull().sum()
        stn_tot = len(grp)
        stn_cov = (stn_nn / stn_tot * 100.0) if stn_tot > 0 else 0.0
        stn_stats.append({"station_id": sid, "non_null": stn_nn, "total": stn_tot, "coverage_pct": stn_cov})
    df_stn_stats = pd.DataFrame(stn_stats)
    low_cov_stns = df_stn_stats[df_stn_stats["coverage_pct"] < 10.0]["station_id"].tolist()
    print(f"    --> Per-Station Note: {len(low_cov_stns)} stations have < 10% neighbor coverage due to geographic isolation: {low_cov_stns}")

    # 6. Drop rows where o3_value is null
    print("\n" + "-" * 80)
    print("FILTERING LABELED ROWS FOR MODEL TRAINING")
    print("-" * 80)
    df_train = df.dropna(subset=[TARGET_COL]).copy().reset_index(drop=True)
    train_rows = len(df_train)
    dropped_rows = total_rows - train_rows
    print(f"Rows dropped due to null '{TARGET_COL}': {dropped_rows:,}")
    print(f"Final row count going into training: {train_rows:,}")

    # 7. Null audit across all feature columns
    print("\n" + "-" * 80)
    print("FEATURE NULL AUDIT IN TRAINING SET")
    print("-" * 80)
    feature_nulls = {}
    unexpected_nulls = False

    for feat in FINAL_FEATURE_LIST:
        null_count = int(df_train[feat].isnull().sum())
        feature_nulls[feat] = null_count
        status = "0 nulls [PASS]"
        if null_count > 0:
            if feat == "neighbor_o3_lagged" and not fallback_needed:
                status = f"{null_count} nulls [EXPECTED: Wind/Neighbor Missingness]"
            else:
                status = f"{null_count} nulls [UNEXPECTED NULL]"
                unexpected_nulls = True
        print(f"  {feat:<22} : {status}")

    # Clean temporary helper columns
    df_train = df_train.drop(columns=["_valid_time_dt", "past_go3"], errors="ignore")
    df = df.drop(columns=["_valid_time_dt", "past_go3"], errors="ignore")

    print("\n" + "=" * 80)
    if not unexpected_nulls:
        print("[SUCCESS] Confirmed: No unexpected nulls remain in any feature column!")
    else:
        print("[WARNING] Unexpected nulls detected in feature columns!")
    print("=" * 80)

    if output_path:
        out_file = Path(output_path)
        df_train.to_csv(out_file, index=False)
        print(f"\n[+] Saved prepared training dataset to: {out_file.resolve()}")

    return df, df_train


def main():
    parser = argparse.ArgumentParser(description="Prepare final training dataset with cyclical encodings and audits.")
    parser.add_argument("--input", default="final_training_table.csv", help="Input training table CSV.")
    parser.add_argument("--output", default=None, help="Optional output CSV path for training dataset.")
    parser.add_argument("--force-fallback", action="store_true", help="Force fallback past go3 imputation.")
    args = parser.parse_args()

    load_and_prepare_data(
        input_path=args.input,
        output_path=args.output,
        force_fallback=args.force_fallback,
    )


if __name__ == "__main__":
    main()
