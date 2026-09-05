"""
train_model.py

Trains an XGBoost regression model for surface ozone (o3_value) prediction:
1. Loads final_training_table.csv and creates cyclical temporal features.
2. Filters out missing ground-truth target rows (o3_value).
3. Evaluates splitting strategy:
   - If 3+ stations have non-null o3_value: spatial holdout (1-2 stations).
   - If < 3 stations: time-based split (last 20% timestamps per station).
4. Trains XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42).
5. Computes naive baseline using raw CAMS ozone (go3) directly as prediction.
6. Evaluates test set metrics (RMSE, MAE, R2).
7. Prints comparison table with % improvements.
8. Saves the trained model as model.pkl using joblib.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Feature configuration
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

FEATURE_COLS = BASE_FEATURES + CYCLICAL_FEATURES
TARGET_COL = "o3_value"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical encodings and sorts data."""
    df = df.copy()
    valid_time_dt = pd.to_datetime(df["valid_time"], format="ISO8601", utc=True)
    df["_valid_time_dt"] = valid_time_dt

    hour = valid_time_dt.dt.hour
    day_of_year = valid_time_dt.dt.dayofyear

    # 24-hour cycle
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # 365.25-day annual cycle
    df["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, list[int]]:
    """
    Implements splitting strategy:
    - If 3+ stations have non-null o3_value: spatial holdout (hold out 1-2 full stations).
    - If fewer than 3 stations: time-based split (last 20% of timestamps per station).
    """
    stns_with_target = sorted(df[df[TARGET_COL].notnull()]["station_id"].unique().tolist())
    n_stns = len(stns_with_target)

    if n_stns >= 3:
        # Spatial holdout strategy: hold out 2 representative stations (North and South)
        # Station 17 (North India - Delhi NCR) and Station 5624 (South India - Telangana)
        held_out_stations = [17, 5624]
        strategy_desc = (
            f"Spatial Holdout: Held out 2 full stations {held_out_stations} "
            f"(Station 17 [North India / Delhi-NCR] and Station 5624 [South India / Hyderabad]) "
            f"because {n_stns} stations have non-null target data (>= 3 station threshold)."
        )
        train_df = df[~df["station_id"].isin(held_out_stations)].copy()
        test_df = df[df["station_id"].isin(held_out_stations)].copy()
    else:
        # Time-based split strategy: last 20% per station
        strategy_desc = (
            f"Time-Based Split: Only {n_stns} station(s) have non-null target data (< 3 station threshold). "
            f"Held out the most recent 20% timestamps per station as test set."
        )
        held_out_stations = []
        train_list = []
        test_list = []
        for sid, grp in df.groupby("station_id"):
            grp_sorted = grp.sort_values("_valid_time_dt")
            n_rows = len(grp_sorted)
            split_idx = int(n_rows * 0.8)
            train_list.append(grp_sorted.iloc[:split_idx])
            test_list.append(grp_sorted.iloc[split_idx:])
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

    return train_df, test_df, strategy_desc, held_out_stations


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes RMSE, MAE, and R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def main():
    csv_path = Path("final_training_table.csv")
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("=" * 85)
    print("           SURFACE OZONE MODEL TRAINING & BASELINE BENCHMARK")
    print("=" * 85)

    # 1. Load and prepare features
    df_raw = pd.read_csv(csv_path)
    print(f"[*] Loaded {csv_path.name}: {len(df_raw):,} total rows across {df_raw['station_id'].nunique()} stations.")

    df_prepared = prepare_features(df_raw)

    # Drop missing target rows
    df_clean = df_prepared.dropna(subset=[TARGET_COL]).copy().reset_index(drop=True)
    n_dropped = len(df_raw) - len(df_clean)
    print(f"[*] Dropped {n_dropped:,} rows where '{TARGET_COL}' is null. Labeled rows: {len(df_clean):,}.")

    # 2. Split dataset
    train_df, test_df, strategy_desc, held_out_stations = split_data(df_clean)

    print("\n" + "-" * 85)
    print("SPLIT STRATEGY & VOLUMES")
    print("-" * 85)
    print(f"Strategy        : {strategy_desc}")
    print(f"Held-out Stations: {held_out_stations}")
    print(f"Training Set    : {len(train_df):,} rows across {train_df['station_id'].nunique()} stations")
    print(f"Test Set        : {len(test_df):,} rows across {test_df['station_id'].nunique()} stations")

    # Volume check
    if len(test_df) < 30:
        print(f"\n[WARNING] Test set has only {len(test_df)} rows (< 30 rows).")
        print("          Results should be treated as preliminary given the limited data volume.")
    else:
        print(f"[PASS] Test set size ({len(test_df)} rows) satisfies >= 30 row threshold.")

    # 3. Train XGBoost model
    print("\n" + "-" * 85)
    print("TRAINING XGBOOST REGRESSOR")
    print("-" * 85)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    model_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
    }
    print(f"Hyperparameters: {model_params}")
    xgb = XGBRegressor(**model_params)
    xgb.fit(X_train, y_train)
    print("[+] Model training complete.")

    # Predict with XGBoost
    y_pred_xgb = xgb.predict(X_test)

    # 4. Compute Naive Baselines
    # Primary: Raw CAMS go3 directly as prediction, no correction applied
    y_pred_naive_raw = test_df["go3"].values

    # Secondary: Calibrated CAMS go3 in ppb (M_air/M_O3 * 1e9)
    ppb_factor = (28.9644 / 47.9982) * 1e9
    y_pred_naive_ppb = test_df["go3"].values * ppb_factor

    # 5. Evaluate on Test Set
    metrics_xgb = calculate_metrics(y_test.values, y_pred_xgb)
    metrics_naive_raw = calculate_metrics(y_test.values, y_pred_naive_raw)
    metrics_naive_ppb = calculate_metrics(y_test.values, y_pred_naive_ppb)

    # 6. Comparison Table
    print("\n" + "=" * 85)
    print("MODEL PERFORMANCE COMPARISON (TEST SET: SPATIAL HOLDOUT)")
    print("=" * 85)
    print(f"{'metric':<10} | {'naive_CAMS_baseline':<20} | {'xgboost_model':<15} | {'% improvement':<15}")
    print("-" * 70)

    for metric in ["RMSE", "MAE", "R2"]:
        val_naive = metrics_naive_raw[metric]
        val_xgb = metrics_xgb[metric]
        if metric in ["RMSE", "MAE"]:
            # Lower is better: percentage reduction in error
            imp = ((val_naive - val_xgb) / val_naive) * 100.0
            imp_str = f"+{imp:.2f}% (reduction)"
        else:
            # Higher is better: point gain
            imp_diff = val_xgb - val_naive
            imp_str = f"+{imp_diff:.4f} pts"
        print(f"{metric:<10} | {val_naive:<20.4f} | {val_xgb:<15.4f} | {imp_str:<15}")
    print("-" * 70)

    print("\n[Supplementary Reference: Naive CAMS after standard mass-to-ppb conversion factor (0.6035e9)]:")
    print(f"{'metric':<10} | {'naive_CAMS_(ppb)':<20} | {'xgboost_model':<15} | {'% improvement':<15}")
    print("-" * 70)
    for metric in ["RMSE", "MAE", "R2"]:
        val_naive = metrics_naive_ppb[metric]
        val_xgb = metrics_xgb[metric]
        if metric in ["RMSE", "MAE"]:
            imp = ((val_naive - val_xgb) / val_naive) * 100.0
            imp_str = f"+{imp:.2f}% (reduction)"
        else:
            imp_diff = val_xgb - val_naive
            imp_str = f"+{imp_diff:.4f} pts"
        print(f"{metric:<10} | {val_naive:<20.4f} | {val_xgb:<15.4f} | {imp_str:<15}")
    print("=" * 85)

    # 7. Save model
    model_path = Path("model.pkl")
    joblib.dump(xgb, model_path)
    print(f"\n[+] Successfully saved trained XGBoost model to: {model_path.resolve()}")

    # Verify model loading
    loaded_model = joblib.load(model_path)
    sample_preds = loaded_model.predict(X_test.iloc[:5])
    assert np.allclose(sample_preds, y_pred_xgb[:5]), "Loaded model predictions mismatch!"
    print(f"[+] Verified model integrity: 'model.pkl' reloaded and confirmed identical inference.")


if __name__ == "__main__":
    main()
