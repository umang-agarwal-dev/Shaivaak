"""
cross_validate.py

Executes 5-Fold Spatial Cross-Validation (GroupKFold grouped by station_id):
1. For each of 5 folds, holds out 4 unseen stations as the test set.
2. Trains XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42).
3. Evaluates RMSE, MAE, R2 on the held-out fold for both XGBoost and naive CAMS baseline.
4. Computes mean and standard deviation across all 5 folds.
5. Saves results_summary.csv with columns: fold, metric, naive_CAMS, xgboost, improvement_pct.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from train_model import FEATURE_COLS, TARGET_COL, prepare_features


def main():
    print("=" * 85)
    print("      5-FOLD SPATIAL GROUP CROSS-VALIDATION (GroupKFold by station_id)")
    print("=" * 85)

    # 1. Load and prepare dataset
    df_raw = pd.read_csv("final_training_table.csv")
    df = prepare_features(df_raw).dropna(subset=[TARGET_COL]).reset_index(drop=True)

    unique_stations = sorted(df["station_id"].unique().tolist())
    n_stations = len(unique_stations)
    n_rows = len(df)
    print(f"[*] Loaded clean dataset: {n_rows:,} labeled rows across {n_stations} unique stations.")
    print(f"[*] Strategy: GroupKFold(n_splits=5) with {n_stations // 5} full stations held out per fold.")

    gkf = GroupKFold(n_splits=5)
    groups = df["station_id"]

    results = []
    fold_details = []

    # 2. Iterate across 5 folds
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups), 1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        held_out_stns = sorted(test_df["station_id"].unique().tolist())
        fold_details.append(held_out_stns)

        X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
        X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

        # Train XGBoost
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )
        xgb.fit(X_train, y_train)

        # Predictions
        y_pred_xgb = xgb.predict(X_test)
        y_pred_naive = test_df["go3"].values

        # Compute Metrics
        rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
        mae_xgb = float(mean_absolute_error(y_test, y_pred_xgb))
        r2_xgb = float(r2_score(y_test, y_pred_xgb))

        rmse_naive = float(np.sqrt(mean_squared_error(y_test, y_pred_naive)))
        mae_naive = float(mean_absolute_error(y_test, y_pred_naive))
        r2_naive = float(r2_score(y_test, y_pred_naive))

        # Improvement percentages
        imp_rmse = ((rmse_naive - rmse_xgb) / rmse_naive) * 100.0
        imp_mae = ((mae_naive - mae_xgb) / mae_naive) * 100.0
        # For R2, point gain is standard, also percentage relative gain
        imp_r2 = r2_xgb - r2_naive

        # Store rows
        results.append({
            "fold": str(fold),
            "metric": "RMSE",
            "naive_CAMS": round(rmse_naive, 4),
            "xgboost": round(rmse_xgb, 4),
            "improvement_pct": round(imp_rmse, 2),
        })
        results.append({
            "fold": str(fold),
            "metric": "MAE",
            "naive_CAMS": round(mae_naive, 4),
            "xgboost": round(mae_xgb, 4),
            "improvement_pct": round(imp_mae, 2),
        })
        results.append({
            "fold": str(fold),
            "metric": "R2",
            "naive_CAMS": round(r2_naive, 4),
            "xgboost": round(r2_xgb, 4),
            "improvement_pct": round(imp_r2, 4),
        })

    # 3. Create DataFrame for results
    df_results = pd.DataFrame(results)

    # 4. Compute Summary Statistics (mean and std per metric)
    summary_rows = []
    print("\n" + "-" * 85)
    print("FOLD-BY-FOLD BREAKDOWN:")
    print("-" * 85)
    for fold in range(1, 6):
        stns = fold_details[fold - 1]
        sub = df_results[df_results["fold"] == str(fold)]
        print(f"Fold {fold} (Held-out Stations: {stns}):")
        for _, r in sub.iterrows():
            imp_sym = "%" if r["metric"] != "R2" else " pts"
            print(f"  {r['metric']:<6} | Naive CAMS: {r['naive_CAMS']:8.4f} | XGBoost: {r['xgboost']:8.4f} | Improvement: {r['improvement_pct']:+7.2f}{imp_sym}")

    print("\n" + "=" * 85)
    print("HEADLINE CROSS-VALIDATION SUMMARY (Mean +/- Std across 5 Folds)")
    print("=" * 85)
    print(f"{'metric':<10} | {'naive_CAMS_mean_std':<25} | {'xgboost_mean_std':<25} | {'% improvement (mean +/- std)':<25}")
    print("-" * 85)

    for metric in ["RMSE", "MAE", "R2"]:
        sub = df_results[df_results["metric"] == metric]
        naive_mean = float(sub["naive_CAMS"].mean())
        naive_std = float(sub["naive_CAMS"].std())
        xgb_mean = float(sub["xgboost"].mean())
        xgb_std = float(sub["xgboost"].std())
        imp_mean = float(sub["improvement_pct"].mean())
        imp_std = float(sub["improvement_pct"].std())

        imp_sym = "%" if metric != "R2" else " pts"
        print(
            f"{metric:<10} | {naive_mean:7.4f} +/- {naive_std:6.4f}       | "
            f"{xgb_mean:7.4f} +/- {xgb_std:6.4f}       | "
            f"{imp_mean:+6.2f}{imp_sym} +/- {imp_std:5.2f}"
        )

        summary_rows.append({
            "fold": "mean",
            "metric": metric,
            "naive_CAMS": round(naive_mean, 4),
            "xgboost": round(xgb_mean, 4),
            "improvement_pct": round(imp_mean, 2),
        })
        summary_rows.append({
            "fold": "std",
            "metric": metric,
            "naive_CAMS": round(naive_std, 4),
            "xgboost": round(xgb_std, 4),
            "improvement_pct": round(imp_std, 2),
        })

    print("=" * 85)

    # 5. Append summary rows to df_results and save results_summary.csv
    df_all = pd.concat([df_results, pd.DataFrame(summary_rows)], ignore_index=True)
    out_csv = Path("results_summary.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n[+] Saved cross-validation results to: {out_csv.resolve()}")
    print("\nFile contents preview (results_summary.csv):")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
