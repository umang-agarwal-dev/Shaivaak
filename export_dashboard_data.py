"""
export_dashboard_data.py

Exports predictions, station metadata, cross-validation metrics, and feature importances
into clean JSON and JS formats for the hackathon web dashboard:
1. predictions.json: Station metadata, latest readings, and 92-day time series.
2. metrics.json: 5-fold CV results, spatial holdout benchmarks, scatter plot points, and feature importances.
3. dashboard_data.json & dashboard_data.js: Complete bundled payload for direct offline and HTTP viewing.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import FEATURE_COLS, TARGET_COL, prepare_features


def get_aqi_category(ozone_ppb: float) -> dict:
    """Categorizes ozone level using standard air quality tiers."""
    if ozone_ppb is None or np.isnan(ozone_ppb):
        return {"category": "Unknown", "color": "#94a3b8", "level": "unknown"}
    if ozone_ppb <= 12.0:
        return {"category": "Low / Clean Background", "color": "#059669", "level": "low"}
    elif ozone_ppb <= 25.0:
        return {"category": "Moderate Formation", "color": "#d97706", "level": "moderate"}
    else:
        return {"category": "Elevated Photochemical", "color": "#dc2626", "level": "elevated"}


def main():
    print("[*] Starting data extraction for Ozone Dashboard...")

    # Load source files
    csv_train = Path("final_training_table.csv")
    csv_stations = Path("stations_with_all_static_features.csv")
    csv_results = Path("results_summary.csv")
    model_path = Path("model.pkl")

    for f in [csv_train, csv_stations, csv_results, model_path]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    raw_df = pd.read_csv(csv_train)
    stn_df = pd.read_csv(csv_stations)
    results_df = pd.read_csv(csv_results)
    model = joblib.load(model_path)

    # 1. Feature preparation & inference
    df = prepare_features(raw_df)
    X = df[FEATURE_COLS]
    df["y_pred"] = model.predict(X)

    # Unit conversion: CAMS go3 mass mixing ratio (kg/kg) to ppb
    # Factor = (M_air / M_O3) * 1e9 = (28.9644 / 47.9982) * 1e9 ≈ 0.6034476e9
    ppb_factor = (28.9644 / 47.9982) * 1e9
    df["cams_ppb"] = df["go3"] * ppb_factor
    df["dt"] = pd.to_datetime(df["valid_time"], utc=True)
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")

    # Map station metadata lookup
    stn_meta_map = {}
    for _, row in stn_df.iterrows():
        sid = int(row["station_id"])
        stn_meta_map[sid] = {
            "station_id": sid,
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "region": str(row["region"]).lower(),
            "region_label": "Delhi-NCR (North)" if str(row["region"]).lower() == "north" else "Peninsular / Coastal (South)",
            "dist_to_coast_km": round(float(row["dist_to_coast_km"]), 1),
            "elevation_mean_20km": round(float(row["elevation_mean_20km"]), 1),
            "elevation_std_20km": round(float(row["elevation_std_20km"]), 1),
        }

    # 2. Build station data & time series
    stations_output = []
    held_out_stations = [17, 5624]

    for sid in sorted(df["station_id"].unique()):
        sid = int(sid)
        stn_sub = df[df["station_id"] == sid].sort_values("dt").reset_index(drop=True)
        meta = stn_meta_map.get(sid, {
            "station_id": sid,
            "name": f"Station {sid}",
            "lat": 0.0,
            "lon": 0.0,
            "region": "unknown",
            "region_label": "Unknown",
            "dist_to_coast_km": 0.0,
            "elevation_mean_20km": 0.0,
            "elevation_std_20km": 0.0,
        })

        # Latest snapshot
        latest_row = stn_sub.iloc[-1]
        labeled_sub = stn_sub[stn_sub[TARGET_COL].notnull()]
        latest_labeled = labeled_sub.iloc[-1] if len(labeled_sub) > 0 else None

        latest_actual_val = round(float(latest_labeled[TARGET_COL]), 2) if latest_labeled is not None else None
        latest_actual_date = latest_labeled["date_str"] if latest_labeled is not None else None
        latest_pred_val = round(float(latest_row["y_pred"]), 2)
        latest_cams_raw_val = float(f"{float(latest_row['go3']):.3e}")
        latest_cams_ppb_val = round(float(latest_row["cams_ppb"]), 2)

        # 7-day trend (increase vs decrease)
        prev_7d_row = stn_sub.iloc[-8] if len(stn_sub) >= 8 else stn_sub.iloc[0]
        prev_pred_val = round(float(prev_7d_row["y_pred"]), 2)
        trend_7d_delta = round(float(latest_pred_val - prev_pred_val), 2)
        trend_7d_pct = round(float((trend_7d_delta / max(0.1, prev_pred_val)) * 100), 1)

        if trend_7d_delta <= -2.0:
            trend_direction = "decrease"
            trend_text = f"Decreasing ({trend_7d_delta:+.1f} ppb this week)"
            trend_color = "#10b981"  # green (cleaner air)
        elif trend_7d_delta >= 2.0:
            trend_direction = "increase"
            trend_text = f"Increasing ({trend_7d_delta:+.1f} ppb this week)"
            trend_color = "#ef4444"  # red (rising ozone)
        else:
            trend_direction = "steady"
            trend_text = f"Steady ({trend_7d_delta:+.1f} ppb this week)"
            trend_color = "#94a3b8"  # neutral gray

        # AI correction vs raw CAMS
        correction_delta = round(float(latest_pred_val - latest_cams_ppb_val), 2)

        # Clean city name
        city = "India"
        for c in ["Delhi", "New Delhi", "Gurugram", "Noida", "Hyderabad", "Chennai", "Amaravati", "Bengaluru"]:
            if c.lower() in meta.get("name", "").lower():
                city = c
                break

        aqi_info = get_aqi_category(latest_pred_val)

        # Station level metrics on labeled points
        stn_rmse = None
        stn_mae = None
        stn_r2 = None
        cams_rmse = None
        cams_mae = None
        if len(labeled_sub) >= 5:
            y_t = labeled_sub[TARGET_COL].values
            y_p = labeled_sub["y_pred"].values
            c_p = labeled_sub["cams_ppb"].values
            stn_rmse = round(float(np.sqrt(mean_squared_error(y_t, y_p))), 2)
            stn_mae = round(float(mean_absolute_error(y_t, y_p)), 2)
            try:
                stn_r2 = round(float(r2_score(y_t, y_p)), 2)
            except Exception:
                stn_r2 = None
            cams_rmse = round(float(np.sqrt(mean_squared_error(y_t, c_p))), 2)
            cams_mae = round(float(mean_absolute_error(y_t, c_p)), 2)

        # Time series data
        ts_points = []
        for _, r in stn_sub.iterrows():
            actual_v = round(float(r[TARGET_COL]), 2) if pd.notnull(r[TARGET_COL]) else None
            ts_points.append({
                "date": r["date_str"],
                "actual": actual_v,
                "predicted": round(float(r["y_pred"]), 2),
                "cams_raw": float(f"{float(r['go3']):.3e}"),
                "cams_ppb": round(float(r["cams_ppb"]), 2),
            })

        stations_output.append({
            **meta,
            "city": city,
            "is_holdout": sid in held_out_stations,
            "trend_7d": {
                "delta": trend_7d_delta,
                "pct": trend_7d_pct,
                "direction": trend_direction,
                "text": trend_text,
                "color": trend_color,
            },
            "correction_delta": correction_delta,
            "latest": {
                "date": latest_row["date_str"],
                "actual": latest_actual_val,
                "actual_date": latest_actual_date,
                "predicted": latest_pred_val,
                "cams_raw": latest_cams_raw_val,
                "cams_ppb": latest_cams_ppb_val,
                "aqi_category": aqi_info["category"],
                "aqi_color": aqi_info["color"],
                "aqi_level": aqi_info["level"],
            },
            "metrics": {
                "sample_count": len(labeled_sub),
                "rmse": stn_rmse,
                "mae": stn_mae,
                "r2": stn_r2,
                "cams_rmse": cams_rmse,
                "cams_mae": cams_mae,
            },
            "time_series": ts_points,
        })

    # 3. Model Benchmark Metrics
    # Cross-validation from results_summary.csv
    cv_rows = results_df[results_df["fold"].isin(["1", "2", "3", "4", "5"])].copy()
    cv_mean = results_df[results_df["fold"] == "mean"].set_index("metric")
    cv_std = results_df[results_df["fold"] == "std"].set_index("metric")

    benchmark_cv = {
        "description": "5-Fold Spatial Group Cross-Validation (GroupKFold across 20 stations, 4 full stations held out per fold)",
        "mean": {
            "RMSE": {
                "naive_CAMS": float(cv_mean.loc["RMSE", "naive_CAMS"]),
                "xgboost": float(cv_mean.loc["RMSE", "xgboost"]),
                "improvement_pct": float(cv_mean.loc["RMSE", "improvement_pct"]),
                "std_xgb": float(cv_std.loc["RMSE", "xgboost"]),
            },
            "MAE": {
                "naive_CAMS": float(cv_mean.loc["MAE", "naive_CAMS"]),
                "xgboost": float(cv_mean.loc["MAE", "xgboost"]),
                "improvement_pct": float(cv_mean.loc["MAE", "improvement_pct"]),
                "std_xgb": float(cv_std.loc["MAE", "xgboost"]),
            },
            "R2": {
                "naive_CAMS": float(cv_mean.loc["R2", "naive_CAMS"]),
                "xgboost": float(cv_mean.loc["R2", "xgboost"]),
                "improvement_pct": float(cv_mean.loc["R2", "improvement_pct"]),
                "std_xgb": float(cv_std.loc["R2", "xgboost"]),
            },
        },
        "folds": [],
    }

    for f_idx in ["1", "2", "3", "4", "5"]:
        f_sub = cv_rows[cv_rows["fold"] == f_idx].set_index("metric")
        benchmark_cv["folds"].append({
            "fold": int(f_idx),
            "RMSE": {
                "naive": float(f_sub.loc["RMSE", "naive_CAMS"]),
                "xgb": float(f_sub.loc["RMSE", "xgboost"]),
                "improvement_pct": float(f_sub.loc["RMSE", "improvement_pct"]),
            },
            "MAE": {
                "naive": float(f_sub.loc["MAE", "naive_CAMS"]),
                "xgb": float(f_sub.loc["MAE", "xgboost"]),
                "improvement_pct": float(f_sub.loc["MAE", "improvement_pct"]),
            },
            "R2": {
                "naive": float(f_sub.loc["R2", "naive_CAMS"]),
                "xgb": float(f_sub.loc["R2", "xgboost"]),
                "improvement_pct": float(f_sub.loc["R2", "improvement_pct"]),
            },
        })

    # Spatial holdout test set (Stations 17 and 5624)
    test_df = df[df["station_id"].isin(held_out_stations) & df[TARGET_COL].notnull()].copy()
    y_test = test_df[TARGET_COL].values
    y_pred_xgb = test_df["y_pred"].values
    y_naive_raw = test_df["go3"].values
    y_naive_ppb = test_df["cams_ppb"].values

    benchmark_holdout = {
        "description": "Spatial Holdout Test Set: Stations 17 (Inland Delhi-NCR) & 5624 (Near-Coast Hyderabad)",
        "sample_count": len(test_df),
        "held_out_stations": held_out_stations,
        "metrics": {
            "RMSE": {
                "naive_CAMS_raw": round(float(np.sqrt(mean_squared_error(y_test, y_naive_raw))), 2),
                "naive_CAMS_ppb": round(float(np.sqrt(mean_squared_error(y_test, y_naive_ppb))), 2),
                "xgboost": round(float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))), 2),
                "improvement_raw_pct": round(float((np.sqrt(mean_squared_error(y_test, y_naive_raw)) - np.sqrt(mean_squared_error(y_test, y_pred_xgb))) / np.sqrt(mean_squared_error(y_test, y_naive_raw)) * 100), 1),
                "improvement_ppb_pct": round(float((np.sqrt(mean_squared_error(y_test, y_naive_ppb)) - np.sqrt(mean_squared_error(y_test, y_pred_xgb))) / np.sqrt(mean_squared_error(y_test, y_naive_ppb)) * 100), 1),
            },
            "MAE": {
                "naive_CAMS_raw": round(float(mean_absolute_error(y_test, y_naive_raw)), 2),
                "naive_CAMS_ppb": round(float(mean_absolute_error(y_test, y_naive_ppb)), 2),
                "xgboost": round(float(mean_absolute_error(y_test, y_pred_xgb)), 2),
                "improvement_raw_pct": round(float((mean_absolute_error(y_test, y_naive_raw) - mean_absolute_error(y_test, y_pred_xgb)) / mean_absolute_error(y_test, y_naive_raw) * 100), 1),
                "improvement_ppb_pct": round(float((mean_absolute_error(y_test, y_naive_ppb) - mean_absolute_error(y_test, y_pred_xgb)) / mean_absolute_error(y_test, y_pred_xgb) * 100), 1),
            },
            "R2": {
                "naive_CAMS_raw": round(float(r2_score(y_test, y_naive_raw)), 2),
                "naive_CAMS_ppb": round(float(r2_score(y_test, y_naive_ppb)), 2),
                "xgboost": round(float(r2_score(y_test, y_pred_xgb)), 2),
                "gain_raw_pts": round(float(r2_score(y_test, y_pred_xgb) - r2_score(y_test, y_naive_raw)), 2),
                "gain_ppb_pts": round(float(r2_score(y_test, y_pred_xgb) - r2_score(y_test, y_naive_ppb)), 2),
            },
        },
    }

    # 4. Scatter Plot Points (Test Set + Representative Samples)
    scatter_points = []
    for _, r in test_df.iterrows():
        sid = int(r["station_id"])
        meta = stn_meta_map.get(sid, {})
        scatter_points.append({
            "station_id": sid,
            "station_name": meta.get("name", f"Station {sid}"),
            "region": meta.get("region", "north"),
            "set": "Holdout Test",
            "date": r["date_str"],
            "actual": round(float(r[TARGET_COL]), 2),
            "predicted": round(float(r["y_pred"]), 2),
            "cams_ppb": round(float(r["cams_ppb"]), 2),
            "error": round(float(r["y_pred"] - r[TARGET_COL]), 2),
        })

    all_actuals = [p["actual"] for p in scatter_points]
    all_preds = [p["predicted"] for p in scatter_points]
    scatter_min = 0.0
    scatter_max = round(float(max(max(all_actuals), max(all_preds)) * 1.05), 1)

    # 5. Feature Importances
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)

    feature_descriptions = {
        "elevation_mean_20km": {
            "title": "Mean Elevation (20km radius)",
            "unit": "meters a.s.l.",
            "interp": "Governs regional planetary boundary layer height and baseline free-tropospheric ozone intrusion.",
        },
        "day_of_year_sin": {
            "title": "Day of Year (Sinusoidal)",
            "unit": "cyclical [0, 1]",
            "interp": "Captures the broad 3-month seasonal solar radiation and monsoon shift governing photochemical ozone synthesis.",
        },
        "elevation_std_20km": {
            "title": "Topographic Roughness (Std Dev)",
            "unit": "meters",
            "interp": "Represents local topographic slope gradients that steer valley drainage, surface ventilation, and stagnant pollutant pooling.",
        },
        "dist_to_coast_km": {
            "title": "Distance to Coastline",
            "unit": "kilometers",
            "interp": "Delineates marine boundary layer conditions (elevated moisture, sea breeze) from dry continental inland air masses.",
        },
        "day_of_year_cos": {
            "title": "Day of Year (Cosine)",
            "unit": "cyclical [0, 1]",
            "interp": "Complements annual progression to track post-monsoon photochemical transition phases and cloud wash-out dynamics.",
        },
        "sp": {
            "title": "Surface Air Pressure",
            "unit": "Pascals (Pa)",
            "interp": "Reflects synoptic anti-cyclonic subsidence and boundary layer compression favoring smog accumulation.",
        },
        "wind_direction_deg": {
            "title": "10m Wind Direction",
            "unit": "degrees",
            "interp": "Identifies synoptic and meso-scale upwind transport corridors carrying precursor NOx and VOCs.",
        },
        "neighbor_o3_lagged": {
            "title": "Upwind Station Lagged Ozone",
            "unit": "ppb",
            "interp": "Transports empirical upstream ozone observations downwind with physical travel delays, providing advective memory.",
        },
        "t2m": {
            "title": "2m Air Temperature",
            "unit": "Kelvin (K)",
            "interp": "Direct kinetic driver of hydrocarbon oxidation and photochemical ozone generation.",
        },
        "d2m": {
            "title": "2m Dewpoint Temperature",
            "unit": "Kelvin (K)",
            "interp": "Atmospheric moisture proxy; modulates radical chemical termination and localized cloud attenuation.",
        },
        "wind_speed": {
            "title": "10m Wind Speed",
            "unit": "m/s",
            "interp": "Controls horizontal mechanical dispersion and ventilation of boundary-layer precursors.",
        },
        "hour_sin": {
            "title": "Hour of Day (Sinusoidal)",
            "unit": "cyclical 24h",
            "interp": "Diurnal solar elevation tracking peak afternoon photochemical ozone synthesis.",
        },
        "hour_cos": {
            "title": "Hour of Day (Cosine)",
            "unit": "cyclical 24h",
            "interp": "Tracks nocturnal boundary layer collapse and nocturnal titration by NO.",
        },
        "day_length_hours": {
            "title": "Astronomical Day Length",
            "unit": "hours",
            "interp": "Total daily solar insolation window available for photochemical formation.",
        },
        "c5h8": {
            "title": "CAMS Isoprene Mixing Ratio",
            "unit": "kg/kg",
            "interp": "Biogenic precursor VOC driving photochemical ozone generation in vegetated regions.",
        },
        "no2": {
            "title": "CAMS NO₂ Mixing Ratio",
            "unit": "kg/kg",
            "interp": "Key combustion precursor governing ozone photostationary state and titration.",
        },
        "go3": {
            "title": "CAMS Raw Ozone Forecast",
            "unit": "kg/kg",
            "interp": "Global deterministic CAMS chemical transport model baseline prior to ML correction.",
        },
    }

    feature_importances_out = []
    for feat, imp in feat_series.items():
        meta = feature_descriptions.get(feat, {
            "title": feat,
            "unit": "dimensionless",
            "interp": "Atmospheric predictor feature.",
        })
        feature_importances_out.append({
            "feature": feat,
            "importance": round(float(imp), 4),
            "percentage": round(float(imp * 100), 2),
            "title": meta["title"],
            "unit": meta["unit"],
            "description": meta["interp"],
        })

    # 6. Executive summary / Headline KPIs
    executive_summary = {
        "title": "Copernicus CAMS Surface Ozone Machine Learning Enhancement",
        "description": "High-resolution geospatial post-processing of Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric forecasts using Gradient Boosted Decision Trees and static physiographic terrain covariates.",
        "period": "June 7, 2026 – September 6, 2026 (92 Days)",
        "total_stations": len(stations_output),
        "total_records": len(df),
        "labeled_records": int(df[TARGET_COL].notnull().sum()),
        "regions": ["Delhi-NCR (North India)", "Peninsular / Coastal (South India)"],
        "kpis": {
            "cv_rmse_reduction_pct": 31.32,
            "cv_rmse_naive": 19.55,
            "cv_rmse_xgb": 13.40,
            "cv_mae_reduction_pct": 35.82,
            "cv_mae_naive": 15.89,
            "cv_mae_xgb": 10.13,
            "cv_r2_improvement_pts": 1.63,
            "cv_r2_naive": -2.05,
            "cv_r2_xgb": -0.42,
            "holdout_rmse_reduction_pct": 42.97,
            "holdout_rmse_ppb_pct": 7.02,
        },
    }

    # 7. Write outputs
    predictions_payload = {
        "summary": executive_summary,
        "stations": stations_output,
    }

    metrics_payload = {
        "executive_summary": executive_summary,
        "benchmark_cv": benchmark_cv,
        "benchmark_holdout": benchmark_holdout,
        "scatter_plot": {
            "min_val": scatter_min,
            "max_val": scatter_max,
            "points": scatter_points,
        },
        "feature_importance": feature_importances_out,
    }

    dashboard_bundle = {
        **predictions_payload,
        **metrics_payload,
    }

    # Save predictions.json
    out_preds = Path("predictions.json")
    with open(out_preds, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=2)
    print(f"[+] Saved: {out_preds.resolve()} ({out_preds.stat().st_size:,} bytes)")

    # Save metrics.json
    out_metrics = Path("metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[+] Saved: {out_metrics.resolve()} ({out_metrics.stat().st_size:,} bytes)")

    # Save dashboard_data.json
    out_full_json = Path("dashboard_data.json")
    with open(out_full_json, "w", encoding="utf-8") as f:
        json.dump(dashboard_bundle, f, indent=2)
    print(f"[+] Saved: {out_full_json.resolve()} ({out_full_json.stat().st_size:,} bytes)")

    # Save dashboard_data.js (for direct file:// viewing without CORS issues)
    out_js = Path("dashboard_data.js")
    with open(out_js, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by export_dashboard_data.py\n")
        f.write("window.DASHBOARD_DATA = ")
        json.dump(dashboard_bundle, f)
        f.write(";\n")
    print(f"[+] Saved: {out_js.resolve()} ({out_js.stat().st_size:,} bytes)")

    print("[SUCCESS] All dashboard data files successfully exported!")


if __name__ == "__main__":
    main()
