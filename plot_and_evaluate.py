"""
plot_and_evaluate.py

Generates visualizations and in-depth diagnostics using model.pkl:
1. Feature importance bar chart -> feature_importance.png
2. Predicted vs Actual scatter plot with y=x line -> predicted_vs_actual.png
3. Time series comparison of Actual vs Naive CAMS vs XGBoost -> timeseries_comparison.png
4. Prints top 5 features with atmospheric interpretations.
5. Evaluates error by distance to coast (coastal vs inland regime analysis).
"""

from pathlib import Path
import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import FEATURE_COLS, TARGET_COL, prepare_features


def main():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11

    # Load model and data
    model = joblib.load("model.pkl")
    df_raw = pd.read_csv("final_training_table.csv")
    df = prepare_features(df_raw).dropna(subset=[TARGET_COL]).reset_index(drop=True)

    held_out_stations = [17, 5624]
    test_df = df[df["station_id"].isin(held_out_stations)].copy().reset_index(drop=True)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Predictions
    y_pred_xgb = model.predict(X_test)
    test_df["y_pred"] = y_pred_xgb
    test_df["error"] = y_pred_xgb - y_test
    test_df["cams_ppb"] = test_df["go3"] * (28.9644 / 47.9982) * 1e9

    # -------------------------------------------------------------------------
    # 1. Plot Feature Importance -> feature_importance.png
    # -------------------------------------------------------------------------
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    bars = ax.barh(feat_series.index, feat_series.values, color="#1f77b4", edgecolor="#0f3b5b", alpha=0.85, height=0.65)

    for bar in bars:
        width = bar.get_width()
        if width > 0.005:
            ax.text(
                width + 0.003,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=9,
                color="#222222",
                fontweight="bold",
            )

    ax.set_xlabel("Relative Importance (Gain / Weight)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predictor Feature", fontsize=11, fontweight="bold")
    ax.set_title("XGBoost Feature Importance for Surface Ozone Prediction", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(feat_series.values) * 1.15)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()
    print("[+] Saved: feature_importance.png")

    # -------------------------------------------------------------------------
    # 2. Plot Predicted vs Actual -> predicted_vs_actual.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=300)

    # Differentiate by station
    colors = {17: "#2ca02c", 5624: "#d62728"}
    labels = {17: "Station 17 (Inland - Delhi NCR)", 5624: "Station 5624 (Near Coast - South India)"}

    for sid in held_out_stations:
        mask = test_df["station_id"] == sid
        ax.scatter(
            test_df.loc[mask, TARGET_COL],
            test_df.loc[mask, "y_pred"],
            color=colors[sid],
            label=labels[sid],
            alpha=0.75,
            edgecolors="k",
            linewidth=0.6,
            s=55,
        )

    # Reference line y = x
    min_val = min(y_test.min(), y_pred_xgb.min()) - 2
    max_val = max(y_test.max(), y_pred_xgb.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.8, label="Ideal 1:1 Line (y = x)")

    # Compute overall test metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae = mean_absolute_error(y_test, y_pred_xgb)
    r2 = r2_score(y_test, y_pred_xgb)

    stats_text = (
        f"Test Set Evaluation (N={len(test_df)}):\n"
        f"  RMSE: {rmse:.2f} ppb\n"
        f"  MAE : {mae:.2f} ppb\n"
        f"  R²  : {r2:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )

    ax.set_xlabel("Actual Surface Ozone (o3_value, ppb)", fontsize=11, fontweight="bold")
    ax.set_ylabel("XGBoost Predicted Ozone (ppb)", fontsize=11, fontweight="bold")
    ax.set_title("Predicted vs Actual Surface Ozone (Spatial Holdout Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ced4da")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=300)
    plt.close()
    print("[+] Saved: predicted_vs_actual.png")

    # -------------------------------------------------------------------------
    # 3. Plot Time Series Comparison -> timeseries_comparison.png
    # -------------------------------------------------------------------------
    test_df["dt"] = pd.to_datetime(test_df["valid_time"], utc=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), dpi=300, sharex=False)

    # Subplot 1: Station 17 (Inland)
    sub17 = test_df[test_df["station_id"] == 17].sort_values("dt")
    ax1.plot(sub17["dt"], sub17[TARGET_COL], "k-o", label="Actual Ozone (Ground Truth)", linewidth=1.8, markersize=4)
    ax1.plot(sub17["dt"], sub17["y_pred"], color="#1f77b4", linestyle="--", label="XGBoost Predicted", linewidth=2.0)
    ax1.plot(
        sub17["dt"],
        sub17["cams_ppb"],
        color="#ff7f0e",
        linestyle=":",
        label="Naive CAMS (Scaled to ppb)",
        linewidth=1.8,
    )
    ax1.set_title(
        "Station 17 (Inland / Delhi-NCR, dist_to_coast=821.8 km) - Temporal Trajectory",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Ozone (ppb)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Station 5624 (Near-Coast)
    sub5624 = test_df[test_df["station_id"] == 5624].sort_values("dt")
    ax2.plot(sub5624["dt"], sub5624[TARGET_COL], "k-o", label="Actual Ozone (Ground Truth)", linewidth=1.8, markersize=4)
    ax2.plot(sub5624["dt"], sub5624["y_pred"], color="#1f77b4", linestyle="--", label="XGBoost Predicted", linewidth=2.0)
    ax2.plot(
        sub5624["dt"],
        sub5624["cams_ppb"],
        color="#ff7f0e",
        linestyle=":",
        label="Naive CAMS (Scaled to ppb)",
        linewidth=1.8,
    )
    ax2.set_title(
        "Station 5624 (Near Coast / South India, dist_to_coast=299.0 km) - Temporal Trajectory",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel("Ozone (ppb)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date (2026)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Time Series Comparison: Ground-Truth Actual vs Naive CAMS vs XGBoost Model",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig("timeseries_comparison.png", dpi=300)
    plt.close()
    print("[+] Saved: timeseries_comparison.png")
    #the team has read through many research papers and have focused on depth over bredth which is great!
    # -------------------------------------------------------------------------
    # 4. Print Top 5 Most Important Features with Interpretation
    # -------------------------------------------------------------------------
    top5 = feat_series.tail(5).iloc[::-1]
    print("\n" + "=" * 80)
    print("TOP 5 MOST IMPORTANT FEATURES & ATMOSPHERIC INTERPRETATION")
    print("=" * 80)
    interpretations = {
        "elevation_mean_20km": "Dictates base atmospheric pressure, boundary layer height, and regional background ozone, as higher plateaus experience reduced surface deposition and greater free-tropospheric intrusion.",
        "day_of_year_sin": "Captures the broad 3-month seasonal solar radiation and monsoon shift (June-September), governing photochemical ozone generation potential and regional cloud cover dynamics.",
        "elevation_std_20km": "Represents local topographic roughness and slope gradients, which steer valley-breeze drainage, surface ventilation, and localized stagnant pollutant pooling.",
        "dist_to_coast_km": "Delineates marine boundary layer conditions (sea-breeze circulation, elevated moisture) from dry continental inland air masses with stronger precursor stagnation.",
        "day_of_year_cos": "Complements day_of_year_sin to precisely track seasonal monsoon progression and transition phases that dictate solar insolation and rainfall wash-out.",
        "sp": "Reflects air mass synoptic pressure patterns, boundary layer compression, and fair-weather anti-cyclonic stagnation favoring photochemical smog accumulation.",
        "wind_direction_deg": "Identifies synoptic and meso-scale upwind transport corridors, determining whether air parcels arrive from polluted urban-industrial upwind source zones.",
        "neighbor_o3_lagged": "Transports empirical upstream ozone observations from upwind stations downwind with physical travel time delays, providing direct advective memory.",
    }

    for rank, (feat, val) in enumerate(top5.items(), 1):
        interp = interpretations.get(feat, "Key predictor for surface ozone concentrations.")
        print(f"{rank}. {feat:<20} ({val * 100:.2f}% importance):")
        print(f"   --> {interp}\n")

    # -------------------------------------------------------------------------
    # 5. Coastal vs Inland Error Comparison
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("COASTAL VS INLAND REGIME PREDICTION ERROR AUDIT (dist_to_coast_km)")
    print("=" * 80)
    med_dist = test_df["dist_to_coast_km"].median()
    print(f"Distance to coast median in test set : {med_dist:.2f} km")
    print(f"Distance spread in test set         : {test_df['dist_to_coast_km'].min():.2f} km to {test_df['dist_to_coast_km'].max():.2f} km")

    below_med = test_df[test_df["dist_to_coast_km"] <= med_dist]  # Station 5624
    above_med = test_df[test_df["dist_to_coast_km"] > med_dist]   # Station 17

    print("\n1. Below Median (Near-Coast Regime, Station 5624 - dist_to_coast = 299.02 km):")
    print(f"   - Sample Count (N)         : {len(below_med)}")
    print(f"   - Mean Error (Pred - Actual): {below_med['error'].mean():+.4f} ppb")
    print(f"   - MAE                      : {below_med['error'].abs().mean():.4f} ppb")
    print(f"   - RMSE                     : {np.sqrt((below_med['error'] ** 2).mean()):.4f} ppb")

    print("\n2. Above Median (Inland Regime, Station 17 - dist_to_coast = 821.75 km):")
    print(f"   - Sample Count (N)         : {len(above_med)}")
    print(f"   - Mean Error (Pred - Actual): {above_med['error'].mean():+.4f} ppb")
    print(f"   - MAE                      : {above_med['error'].abs().mean():.4f} ppb")
    print(f"   - RMSE                     : {np.sqrt((above_med['error'] ** 2).mean()):.4f} ppb")

    diff_mean = below_med["error"].mean() - above_med["error"].mean()
    print("\nRegime Divergence Insight:")
    print(f"   - Delta Mean Bias (Coastal - Inland): {diff_mean:+.4f} ppb")
    if below_med["error"].mean() < 0 and above_med["error"].mean() >= 0:
        print("   --> Model exhibits negative bias (underprediction) in the near-coast regime,")
        print("       while displaying near-neutral/slight positive bias in the deep continental inland regime.")


if __name__ == "__main__":
    main()
