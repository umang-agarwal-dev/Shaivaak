"""
verify_pipeline.py

Comprehensive end-to-end verification script for the ozone prediction data pipeline.
Performs verification across 5 sequential stages and prints a clear pass/fail report:
1. RAW CAMS DATA (existence, variables, NaNs, bounding box, date ranges)
2. STATION METADATA (columns, nulls, feature distribution sanity, bounding boxes)
3. GROUND-TRUTH OZONE DATA (station alignment, duplicates, hourly coverage)
4. BASE TRAINING TABLE (shape, column nulls, target non-null count, temporal alignment, spot check)
5. FINAL TABLE WITH WIND-LAG FEATURE (wind-lag sparsity, recommendations)
6. FINAL SUMMARY (verdict per section and actionable issues list)

Does not silently swallow errors - fails fast with exact diagnostics if any required file
is missing or fails to load.
"""

import sys
import os
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

# Reconfigure stdout for UTF-8 support on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress benign engine warnings from xarray / cfgrib if ecCodes is absent
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray.*")

# ANSI color codes for terminal formatting
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GRAY = "\033[90m"


def format_status(status: str) -> str:
    """Returns color-coded status badge."""
    if status == "PASS":
        return f"{BOLD}{GREEN}[ PASS ]{RESET}"
    elif status == "FAIL":
        return f"{BOLD}{RED}[ FAIL ]{RESET}"
    elif status == "WARNING":
        return f"{BOLD}{YELLOW}[ WARNING ]{RESET}"
    elif status == "CRITICAL":
        return f"{BOLD}{RED}[ CRITICAL FAIL ]{RESET}"
    return f"{BOLD}[ {status} ]{RESET}"


def print_header(title: str, section_num: int | None = None):
    """Prints a styled section header banner."""
    width = 80
    border = "=" * width
    print(f"\n{BOLD}{CYAN}{border}{RESET}")
    if section_num is not None:
        title_str = f"SECTION {section_num}: {title.upper()}"
    else:
        title_str = title.upper()
    print(f"{BOLD}{WHITE} {title_str.center(width - 2)} {RESET}")
    print(f"{BOLD}{CYAN}{border}{RESET}\n")


def load_dataset_strict(file_path: Path, description: str):
    """
    Loads a netCDF or CSV file strictly.
    Exits with a clear diagnostic message if missing or corrupted.
    """
    if not file_path.exists():
        print(f"\n{BOLD}{RED}[FATAL ERROR]{RESET} Missing required {description} file:")
        print(f"  --> Path: {file_path.resolve()}")
        print(f"  Verification stopped. Do not proceed until this file is present.\n")
        sys.exit(1)

    try:
        if file_path.suffix == ".nc":
            import xarray as xr
            ds = xr.open_dataset(file_path)
            return ds
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            return df
        else:
            raise ValueError(f"Unsupported file extension '{file_path.suffix}'")
    except Exception as e:
        print(f"\n{BOLD}{RED}[FATAL ERROR]{RESET} Failed to load {description} at '{file_path.resolve()}':")
        print(f"  --> Error: {type(e).__name__}: {e}")
        print(f"  Verification stopped.\n")
        sys.exit(1)


# ==============================================================================
# SECTION 1: RAW CAMS DATA
# ==============================================================================
def verify_raw_cams_data(north_nc_path: Path, south_nc_path: Path) -> dict:
    print_header("Raw CAMS Data Verification", 1)
    results = {"status": "PASS", "issues": []}

    required_vars = ["go3", "no2", "c5h8", "t2m", "d2m", "sp", "wind_speed", "wind_direction_deg"]
    nan_check_vars = ["go3", "no2", "t2m"]

    # Intended bounding boxes
    intended_boxes = {
        "north": {"lon_min": 72.0, "lon_max": 81.0, "lat_min": 18.0, "lat_max": 29.5},
        "south": {"lon_min": 72.0, "lon_max": 81.0, "lat_min": 12.0, "lat_max": 19.5},
    }

    datasets_to_check = [
        ("North CAMS Dataset", north_nc_path, "north"),
        ("South CAMS Dataset", south_nc_path, "south"),
    ]

    for label, path, region_key in datasets_to_check:
        print(f"[*] Checking {label}: {path.name}")
        ds = load_dataset_strict(path, label)

        print(f"  {GREEN}+ Successfully opened:{RESET} {path.resolve()}")

        # 1. Check data_vars list
        present_vars = list(ds.data_vars.keys())
        print(f"  [*] Total Data Variables: {len(present_vars)}")
        print(f"      Variables: {', '.join(present_vars)}")

        missing_vars = [v for v in required_vars if v not in present_vars]
        if missing_vars:
            print(f"  {RED}[!] Missing required variables:{RESET} {missing_vars}")
            results["issues"].append(f"{label} missing required variables: {missing_vars}")
            results["status"] = "FAIL"
        else:
            print(f"  {GREEN}[+] All 8 required variables present:{RESET} {', '.join(required_vars)}")

        # 2. Check NaNs across go3, no2, t2m
        print("  [*] NaN Audit for critical variables:")
        has_nans = False
        for var in nan_check_vars:
            if var in ds:
                nan_count = int(ds[var].isnull().sum())
                total_points = ds[var].size
                nan_pct = (nan_count / total_points) * 100.0 if total_points > 0 else 0.0
                status_icon = f"{RED}[!] NaNs detected{RESET}" if nan_count > 0 else f"{GREEN}[0.00% NaN - OK]{RESET}"
                print(f"      - {var:<6}: {nan_count:>8} / {total_points:>8} NaNs ({nan_pct:>5.2f}%) {status_icon}")
                if nan_count > 0:
                    has_nans = True
                    results["issues"].append(f"{label} has {nan_count} NaNs ({nan_pct:.2f}%) in '{var}'")
            else:
                print(f"      - {var:<6}: {RED}NOT FOUND in dataset{RESET}")
                has_nans = True

        if has_nans:
            results["status"] = "FAIL"
        else:
            print(f"  {GREEN}[+] Zero NaN values across full go3, no2, and t2m variables in {label}.{RESET}")

        # 3. Lat / Lon / Time range check
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
        lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
        lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())

        time_coord = "valid_time" if "valid_time" in ds else "time"
        time_vals = ds[time_coord].values
        time_min = pd.to_datetime(time_vals.min())
        time_max = pd.to_datetime(time_vals.max())
        num_timesteps = len(time_vals)

        box = intended_boxes[region_key]
        print("  [*] Coordinate Extents & Bounding Box Check:")
        print(f"      - Latitude Range : {lat_min:.2f}N to {lat_max:.2f}N  (Intended: {box['lat_min']}N to {box['lat_max']}N)")
        print(f"      - Longitude Range: {lon_min:.2f}E to {lon_max:.2f}E  (Intended: {box['lon_min']}E to {box['lon_max']}E)")
        print(f"      - Temporal Range : {time_min.strftime('%Y-%m-%d %H:%M:%S UTC')} to {time_max.strftime('%Y-%m-%d %H:%M:%S UTC')} ({num_timesteps} steps)")

        # Verify bounds within intended bounding box
        box_ok = True
        if lat_min < box["lat_min"] or lat_max > box["lat_max"]:
            print(f"      {RED}[!] Latitude range [{lat_min:.2f}, {lat_max:.2f}] exceeds intended [{box['lat_min']}, {box['lat_max']}]{RESET}")
            box_ok = False
        if lon_min < box["lon_min"] or lon_max > box["lon_max"]:
            print(f"      {RED}[!] Longitude range [{lon_min:.2f}, {lon_max:.2f}] exceeds intended [{box['lon_min']}, {box['lon_max']}]{RESET}")
            box_ok = False

        if box_ok:
            print(f"      {GREEN}[+] Geographic bounds fall strictly inside intended {region_key} bounding box.{RESET}")
        else:
            results["issues"].append(f"{label} coordinates fall outside intended bounding box {box}")
            results["status"] = "FAIL"

        ds.close()
        print()

    print(f"--> Section 1 Verdict: {format_status(results['status'])}")
    return results


# ==============================================================================
# SECTION 2: STATION METADATA
# ==============================================================================
def verify_station_metadata(stations_csv_path: Path) -> dict:
    print_header("Station Metadata Verification", 2)
    results = {"status": "PASS", "issues": []}

    df_stn = load_dataset_strict(stations_csv_path, "Station Metadata")
    print(f"[*] Loaded Station Metadata: {stations_csv_path.name}")

    # Required columns
    required_cols = [
        "station_id",
        "name",
        "lat",
        "lon",
        "region",
        "dist_to_coast_km",
        "elevation_mean_20km",
        "elevation_std_20km",
    ]

    missing_cols = [c for c in required_cols if c not in df_stn.columns]
    if missing_cols:
        print(f"{RED}[!] Missing required columns:{RESET} {missing_cols}")
        results["issues"].append(f"Station metadata missing columns: {missing_cols}")
        results["status"] = "FAIL"
    else:
        print(f"{GREEN}[+] All {len(required_cols)} required columns present:{RESET} {', '.join(required_cols)}")

    num_stations = len(df_stn)
    print(f"[*] Row Count (Number of Stations): {BOLD}{num_stations}{RESET}")

    if num_stations == 0:
        print(f"{RED}[!] Station metadata is empty!{RESET}")
        results["issues"].append("Station metadata CSV has 0 rows.")
        results["status"] = "FAIL"
        return results

    # Check for null values in required columns
    print("[*] Checking for null values in station records:")
    null_records = []
    for idx, row in df_stn.iterrows():
        null_in_row = [c for c in required_cols if c in df_stn.columns and pd.isnull(row[c])]
        if null_in_row:
            null_records.append((row.get("station_id", f"row_{idx}"), null_in_row))

    if null_records:
        print(f"  {RED}[!] Found {len(null_records)} station(s) with null values:{RESET}")
        for stn_id, cols in null_records:
            print(f"      - Station ID {stn_id}: Null in {cols}")
            results["issues"].append(f"Station {stn_id} has nulls in columns {cols}")
        results["status"] = "FAIL"
    else:
        print(f"  {GREEN}[+] Zero null values found across all required columns for all stations.{RESET}")

    # Sanity checks on physical features
    print("\n[*] Sanity check on static feature realistic ranges:")
    check_features = [
        ("dist_to_coast_km", 0.0, 2000.0, "km", "India maximum inland distance ~2000 km"),
        ("elevation_mean_20km", -500.0, 9000.0, "m", "Elevation should be within Earth extremes"),
        ("elevation_std_20km", 0.0, 5000.0, "m", "Std dev of elevation must be non-negative"),
    ]

    print(f"  {'Feature':<22} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Sanity Range':<25} | Status")
    print("  " + "-" * 90)

    for col, min_valid, max_valid, unit, note in check_features:
        if col in df_stn.columns:
            vals = df_stn[col].dropna()
            f_min = vals.min()
            f_max = vals.max()
            f_mean = vals.mean()

            is_sane = (f_min >= min_valid) and (f_max <= max_valid)
            status_badge = f"{GREEN}OK{RESET}" if is_sane else f"{RED}OUT OF RANGE{RESET}"

            print(f"  {col:<22} | {f_min:>10.2f} | {f_max:>10.2f} | {f_mean:>10.2f} | {f'[{min_valid:.0f}, {max_valid:.0f}] {unit}':<25} | {status_badge}")
            if not is_sane:
                results["issues"].append(f"Feature '{col}' outside realistic range: [{f_min:.2f}, {f_max:.2f}] ({note})")
                results["status"] = "FAIL"
        else:
            print(f"  {col:<22} | Column Missing")

    # Confirm every station's lat/lon falls inside intended bounding boxes
    print("\n[*] Confirming station coordinates fall inside intended regional bounding boxes:")
    box_violations = []
    for idx, row in df_stn.iterrows():
        stn_id = row.get("station_id", f"idx_{idx}")
        lat = float(row["lat"])
        lon = float(row["lon"])
        region = str(row.get("region", "")).strip().lower()

        # North: 72-81E, 18-29.5N; South: 72-81E, 12-19.5N
        in_north = (72.0 <= lon <= 81.0) and (18.0 <= lat <= 29.5)
        in_south = (72.0 <= lon <= 81.0) and (12.0 <= lat <= 19.5)

        if region == "north":
            valid = in_north
            box_label = "North Box (72-81E, 18-29.5N)"
        elif region == "south":
            valid = in_south
            box_label = "South Box (72-81E, 12-19.5N)"
        else:
            valid = in_north or in_south
            box_label = "Either North or South Box"

        status_str = f"{GREEN}[INSIDE]{RESET}" if valid else f"{RED}[OUTSIDE]{RESET}"
        print(f"  - Station {stn_id} ({row.get('name', '')}): lat={lat:.4f}, lon={lon:.4f}, region='{region}' -> {box_label} {status_str}")

        if not valid:
            box_violations.append((stn_id, lat, lon, region))

    if box_violations:
        print(f"  {RED}[!] {len(box_violations)} station(s) fall outside their regional bounding box.{RESET}")
        for stn_id, lat, lon, region in box_violations:
            results["issues"].append(f"Station {stn_id} ({lat:.4f}, {lon:.4f}) outside {region} bounding box")
        results["status"] = "FAIL"
    else:
        print(f"  {GREEN}[+] Every station's coordinates fall strictly inside intended bounding boxes.{RESET}")

    if num_stations < 3 and results["status"] == "PASS":
        print(f"\n  {YELLOW}[NOTE] Only {num_stations} station found in metadata. Multi-station spatial features like wind-lag will be limited.{RESET}")

    print(f"\n--> Section 2 Verdict: {format_status(results['status'])}")
    return results


# ==============================================================================
# SECTION 3: GROUND-TRUTH OZONE DATA
# ==============================================================================
def verify_ground_truth_o3(o3_csv_path: Path, stations_csv_path: Path) -> dict:
    print_header("Ground-Truth Ozone Data Verification", 3)
    results = {"status": "PASS", "issues": []}

    df_o3 = load_dataset_strict(o3_csv_path, "Ground-Truth O3 Data")
    df_stn = load_dataset_strict(stations_csv_path, "Station Metadata")

    total_rows = len(df_o3)
    print(f"[*] Loaded O3 Measurements: {o3_csv_path.name}")
    print(f"  - Row Count: {BOLD}{total_rows:,}{RESET}")

    if total_rows == 0:
        print(f"{RED}[!] o3_measurements.csv is empty!{RESET}")
        results["issues"].append("Ground-truth o3_measurements.csv contains 0 rows.")
        results["status"] = "FAIL"
        return results

    # Required columns check
    for req in ["station_id", "datetime_utc", "o3_value"]:
        if req not in df_o3.columns:
            print(f"{RED}[!] Missing required column in o3_measurements.csv: {req}{RESET}")
            results["issues"].append(f"o3_measurements.csv missing column: {req}")
            results["status"] = "FAIL"
            return results

    # Date range and unique station IDs
    df_o3["dt"] = pd.to_datetime(df_o3["datetime_utc"], utc=True)
    min_date = df_o3["dt"].min()
    max_date = df_o3["dt"].max()
    unique_stations = df_o3["station_id"].unique()
    num_unique_stns = len(unique_stations)

    print(f"  - Date Range Covered        : {min_date.strftime('%Y-%m-%d %H:%M:%S UTC')} to {max_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  - Unique Station Count      : {BOLD}{num_unique_stns}{RESET} -> {unique_stations.tolist()}")

    # Cross-reference station IDs with station metadata
    metadata_station_ids = set(df_stn["station_id"].unique())
    o3_station_ids = set(unique_stations)

    unmatched_in_o3 = o3_station_ids - metadata_station_ids
    if unmatched_in_o3:
        print(f"  {RED}[!] Station IDs in o3_measurements not found in metadata CSV:{RESET} {list(unmatched_in_o3)}")
        results["issues"].append(f"Stations in o3_measurements missing in metadata: {list(unmatched_in_o3)}")
        results["status"] = "FAIL"
    else:
        print(f"  {GREEN}[+] Every station_id in o3_measurements exists in stations_with_all_static_features.csv.{RESET}")

    unmatched_in_meta = metadata_station_ids - o3_station_ids
    if unmatched_in_meta:
        print(f"  {YELLOW}[!] Stations in metadata without any O3 measurements:{RESET} {list(unmatched_in_meta)}")
        results["issues"].append(f"Metadata stations missing measurements: {list(unmatched_in_meta)}")
        if results["status"] != "FAIL":
            results["status"] = "WARNING"

    # Check for duplicate (station_id, datetime_utc) rows
    dup_mask = df_o3.duplicated(subset=["station_id", "datetime_utc"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        print(f"  {RED}[!] Found {dup_count} duplicate (station_id, datetime_utc) rows!{RESET}")
        print("      Sample duplicates:")
        print(df_o3[dup_mask].head(4)[["station_id", "datetime_utc", "o3_value"]])
        results["issues"].append(f"o3_measurements.csv contains {dup_count} duplicate (station_id, datetime_utc) rows")
        results["status"] = "FAIL"
    else:
        print(f"  {GREEN}[+] Zero duplicate (station_id, datetime_utc) rows found.{RESET}")

    # Hourly coverage completeness per station
    print("\n[*] % Expected Hourly Timestamps Present per Station:")
    print(f"  {'Station ID':<12} | {'Start Time (UTC)':<20} | {'End Time (UTC)':<20} | {'Span (hrs)':>10} | {'Actual Rows':>12} | {'Coverage %':>10} | Gap Assessment")
    print("  " + "-" * 105)

    for stn_id in unique_stations:
        stn_data = df_o3[df_o3["station_id"] == stn_id]
        stn_min = stn_data["dt"].min()
        stn_max = stn_data["dt"].max()
        span_hours = int(round((stn_max - stn_min).total_seconds() / 3600.0)) + 1
        actual_rows = len(stn_data)
        coverage_pct = (actual_rows / span_hours) * 100.0 if span_hours > 0 else 0.0

        if coverage_pct >= 85.0:
            cov_status = f"{GREEN}EXCELLENT (Dense){RESET}"
        elif coverage_pct >= 50.0:
            cov_status = f"{YELLOW}MODERATE (Moderate gaps){RESET}"
        else:
            cov_status = f"{RED}LOW COVERAGE (Significant gaps){RESET}"

        print(
            f"  {stn_id:<12} | {stn_min.strftime('%Y-%m-%d %H:%M'):<20} | {stn_max.strftime('%Y-%m-%d %H:%M'):<20} | "
            f"{span_hours:>10} | {actual_rows:>12} | {coverage_pct:>9.2f}% | {cov_status}"
        )

        if coverage_pct < 50.0:
            results["issues"].append(
                f"Station {stn_id} has low hourly temporal coverage ({coverage_pct:.2f}%: {actual_rows} present of {span_hours} expected hours)"
            )
            if results["status"] != "FAIL":
                results["status"] = "WARNING"

    print(f"\n--> Section 3 Verdict: {format_status(results['status'])}")
    return results


# ==============================================================================
# SECTION 4: BASE TRAINING TABLE
# ==============================================================================
def verify_base_training_table(base_csv_path: Path, cams_start: pd.Timestamp | None, cams_end: pd.Timestamp | None) -> dict:
    print_header("Base Training Table Verification", 4)
    results = {"status": "PASS", "issues": []}

    df_base = load_dataset_strict(base_csv_path, "Base Training Table")
    total_rows, total_cols = df_base.shape
    print(f"[*] Loaded Base Training Table: {base_csv_path.name}")
    print(f"  - Shape (rows, columns): {BOLD}({total_rows}, {total_cols}){RESET}")

    if total_rows == 0:
        print(f"{RED}[!] base_training_table.csv is empty!{RESET}")
        results["issues"].append("base_training_table.csv has 0 rows.")
        results["status"] = "FAIL"
        return results

    # Full column list
    print(f"\n[*] Full Column List ({total_cols} columns):")
    print(f"    {', '.join(df_base.columns)}")

    # Null value percentage per column
    print("\n[*] Column Null Value Analysis (% Null per Column):")
    print(f"  {'Column Name':<25} | {'Non-Null Count':>14} | {'Null Count':>10} | {'% Null':>8} | Status / Flag")
    print("  " + "-" * 80)

    high_null_cols = []
    for col in df_base.columns:
        null_count = int(df_base[col].isnull().sum())
        non_null_count = total_rows - null_count
        null_pct = (null_count / total_rows) * 100.0

        if null_pct > 20.0:
            status = f"{RED}[FLAGGED: >20% NULL CONCERN]{RESET}"
            high_null_cols.append((col, null_pct))
        elif null_pct > 0.0:
            status = f"{YELLOW}[PARTIAL NULL]{RESET}"
        else:
            status = f"{GREEN}[100% COMPLETE]{RESET}"

        print(f"  {col:<25} | {non_null_count:>14} | {null_count:>10} | {null_pct:>7.2f}% | {status}")

    if high_null_cols:
        for col, pct in high_null_cols:
            results["issues"].append(f"Column '{col}' is {pct:.2f}% null (>20% concern threshold)")
        if results["status"] != "FAIL":
            results["status"] = "WARNING"

    # Target o3_value check
    print("\n[*] Target Variable Audit ('o3_value'):")
    if "o3_value" not in df_base.columns:
        print(f"  {RED}[CRITICAL] Target column 'o3_value' is completely missing!{RESET}")
        results["issues"].append("Target column 'o3_value' missing from base training table.")
        results["status"] = "FAIL"
    else:
        target_non_null = int(df_base["o3_value"].notnull().sum())
        print(f"  - Exact Non-Null Count     : {BOLD}{target_non_null}{RESET} / {total_rows} rows")
        print(f"  - Non-Null Training Ratio  : {BOLD}{(target_non_null / total_rows) * 100.0:.2f}%{RESET}")

        if target_non_null == 0:
            print(f"  {RED}[CRITICAL BLOCKER] Exactly 0 non-null rows in target 'o3_value'!{RESET}")
            print("  --> Cause: Temporal mismatch between CAMS dataset and OpenAQ ground-truth timestamps during merge.")
            print("  --> Impact: Model training cannot run with zero labeled samples.")
            results["issues"].append(
                "CRITICAL: Target 'o3_value' has 0 non-null values in base_training_table.csv. "
                "No training labels are available for model training."
            )
            results["status"] = "FAIL"
        elif target_non_null < 50:
            print(f"  {YELLOW}[WARNING] Very few non-null target samples ({target_non_null}) available for training.{RESET}")
            results["issues"].append(f"Low target sample count ({target_non_null} rows).")
            if results["status"] != "FAIL":
                results["status"] = "WARNING"
        else:
            print(f"  {GREEN}[+] Target variable has sufficient non-null rows ({target_non_null} / {total_rows} -> {(target_non_null / total_rows) * 100.0:.1f}%).{RESET}")

    # Valid time check & duplicates
    print("\n[*] Temporal Alignment & Duplicate Check:")
    if "valid_time" in df_base.columns:
        df_base["dt"] = pd.to_datetime(df_base["valid_time"], utc=True)
        base_min_time = df_base["dt"].min()
        base_max_time = df_base["dt"].max()
        print(f"  - Valid Time Range in Base Table: {base_min_time.strftime('%Y-%m-%d %H:%M:%S UTC')} to {base_max_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        if cams_start is not None and cams_end is not None:
            time_match = (base_min_time >= cams_start) and (base_max_time <= cams_end)
            if time_match:
                print(f"  {GREEN}[+] valid_time values fall strictly within intended CAMS date range [{cams_start.date()} to {cams_end.date()}].{RESET}")
            else:
                print(f"  {RED}[!] valid_time range [{base_min_time}, {base_max_time}] falls outside CAMS range [{cams_start}, {cams_end}]{RESET}")
                results["issues"].append("base_training_table valid_time falls outside CAMS date range.")
                results["status"] = "FAIL"

        dup_check = df_base.duplicated(subset=["station_id", "valid_time"], keep=False)
        dup_cnt = int(dup_check.sum())
        if dup_cnt > 0:
            print(f"  {RED}[!] Found {dup_cnt} duplicate (station_id, valid_time) rows!{RESET}")
            results["issues"].append(f"base_training_table contains {dup_cnt} duplicate (station_id, valid_time) rows.")
            results["status"] = "FAIL"
        else:
            print(f"  {GREEN}[+] Zero duplicate (station_id, valid_time) pairs found.{RESET}")
    else:
        print(f"  {RED}[!] 'valid_time' column missing from base table!{RESET}")
        results["issues"].append("base_training_table missing 'valid_time' column.")
        results["status"] = "FAIL"

    # Spot-check 3 random rows
    print("\n[*] Spot-Check: Inspecting 3 Random Rows in Full (Physical Sanity Evaluation):")
    sample_size = min(3, total_rows)
    sample_df = df_base.sample(n=sample_size, random_state=42)

    cols_to_print = [c for c in df_base.columns if c != "dt"]

    for i, (idx, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\n  --- Spot-Check Sample #{i} (Row Index {idx}) ---")
        for col in cols_to_print:
            val = row[col]
            comment = ""
            if col == "t2m" and pd.notnull(val):
                celsius = val - 273.15
                comment = f"({celsius:.2f} deg C -> {'Sane for India summer/monsoon' if 15 <= celsius <= 50 else 'Unusual'})"
            elif col == "wind_speed" and pd.notnull(val):
                comment = f"({'Valid non-negative m/s' if val >= 0 else 'INVALID NEGATIVE SPEED'})"
            elif col == "o3_value":
                comment = f"({'NULL / Unmatched label' if pd.isnull(val) else f'{val} (ground truth)'})"
            elif col == "sp" and pd.notnull(val):
                comment = f"({val/100:.1f} hPa -> {'Plausible surface pressure' if 80000 <= val <= 105000 else 'Unusual'})"
            elif col == "day_length_hours" and pd.notnull(val):
                comment = f"({val:.2f} hours)"
            elif col == "dist_to_coast_km" and pd.notnull(val):
                comment = f"({val:.1f} km)"

            if isinstance(val, (float, np.floating)) and (abs(val) < 1e-4 or abs(val) > 1e5):
                val_str = f"{val:.6e}"
            else:
                val_str = str(val)

            print(f"    {col:<24} : {val_str:<22} {GRAY}{comment}{RESET}")

    print(f"\n--> Section 4 Verdict: {format_status(results['status'])}")
    return results


# ==============================================================================
# SECTION 5: FINAL TABLE WITH WIND-LAG FEATURE
# ==============================================================================
def verify_final_training_table(final_csv_path: Path) -> dict:
    print_header("Final Table with Wind-Lag Feature Verification", 5)
    results = {"status": "PASS", "issues": []}

    if not final_csv_path.exists():
        print(f"{YELLOW}[!] final_training_table.csv does not exist yet at:{RESET}")
        print(f"    {final_csv_path.resolve()}")
        print(f"    Skipping wind-lag verification until add_wind_lag_feature.py is run.")
        results["status"] = "WARNING"
        results["issues"].append("final_training_table.csv not found.")
        return results

    df_final = load_dataset_strict(final_csv_path, "Final Training Table")
    print(f"[*] Loaded Final Training Table: {final_csv_path.name}")
    print(f"  - Shape (rows, cols): {BOLD}{df_final.shape}{RESET}")

    target_feature = "neighbor_o3_lagged"
    if target_feature not in df_final.columns:
        print(f"  {RED}[!] Column '{target_feature}' is missing from {final_csv_path.name}!{RESET}")
        results["issues"].append(f"'{target_feature}' column missing from final table.")
        results["status"] = "FAIL"
        return results

    total_rows = len(df_final)
    non_null_count = int(df_final[target_feature].notnull().sum())
    non_null_pct = (non_null_count / total_rows) * 100.0 if total_rows > 0 else 0.0

    print(f"\n[*] '{target_feature}' Non-Null Audit:")
    print(f"  - Total Rows                : {total_rows}")
    print(f"  - Non-Null Lagged O3 Rows   : {BOLD}{non_null_count}{RESET} / {total_rows}")
    print(f"  - Non-Null Percentage       : {BOLD}{non_null_pct:.2f}%{RESET}")

    if non_null_pct < 10.0:
        print(f"\n  {BOLD}{YELLOW}[WARNING] Sparsity Alert:{RESET} {target_feature} coverage is only {non_null_pct:.2f}% (< 10.0% threshold)!")
        print("  --> The wind-lag transport feature may be too sparse to provide predictive value.")
        print("  --> Recommendations to address sparsity:")
        print("      1. Widen the neighbor search radius (e.g., expand from 100 km to 150 km or 200 km).")
        print("      2. Widen the wind-direction tolerance angle (e.g., from +/-45 deg to +/-60 deg or +/-90 deg).")
        print("      3. Add more monitoring stations in the surrounding geographical region (Delhi-NCR network).")
        print("      4. Ensure target o3_value measurements are populated so lagged observations actually exist.")
        results["issues"].append(
            f"Wind-lag feature '{target_feature}' is very sparse ({non_null_pct:.2f}% non-null). "
            f"Widen search radius, loosen wind-direction tolerance, or add neighbor stations."
        )
        results["status"] = "WARNING"
    else:
        print(f"  {GREEN}[+] Wind-lag feature has acceptable non-null coverage ({non_null_pct:.2f}%).{RESET}")

    print(f"\n--> Section 5 Verdict: {format_status(results['status'])}")
    return results


# ==============================================================================
# SECTION 6: FINAL SUMMARY & DIAGNOSTIC REPORT
# ==============================================================================
def print_final_summary(section_results: dict):
    print_header("Comprehensive Pipeline Verification Summary", 6)

    sections = [
        ("1. Raw CAMS Data", section_results["s1"]),
        ("2. Station Metadata", section_results["s2"]),
        ("3. Ground-Truth Ozone Data", section_results["s3"]),
        ("4. Base Training Table", section_results["s4"]),
        ("5. Final Wind-Lag Table", section_results["s5"]),
    ]

    print(f"  {'Verification Stage':<45} | Status")
    print("  " + "=" * 65)
    has_critical_fail = False
    has_warning = False

    for name, res in sections:
        status = res["status"]
        if status == "FAIL":
            has_critical_fail = True
        elif status == "WARNING":
            has_warning = True
        print(f"  {name:<45} | {format_status(status)}")
    print("  " + "=" * 65)

    # Collect all identified issues
    all_issues = []
    for name, res in sections:
        for issue in res["issues"]:
            all_issues.append((name, issue))

    print("\n" + BOLD + "Detailed Diagnostics & Required Fixes Before Training:" + RESET)
    if not all_issues:
        print(f"  {GREEN}[+] No issues detected. The pipeline is fully validated and ready for model training!{RESET}")
    else:
        for idx, (sec_name, issue) in enumerate(all_issues, 1):
            color = RED if "CRITICAL" in issue or "0 non-null" in issue or "missing" in issue.lower() else YELLOW
            print(f"  {BOLD}{idx}. [{sec_name}]{RESET} {color}{issue}{RESET}")

    # Overall Pipeline Readiness Assessment
    print("\n" + "=" * 80)
    if has_critical_fail:
        print(f"  {BOLD}{RED}OVERALL PIPELINE VERDICT: BLOCKED (CANNOT PROCEED TO MODEL TRAINING){RESET}")
        print("  Fix the critical issues above (especially target variable alignment) before training.")
    elif has_warning:
        print(f"  {BOLD}{YELLOW}OVERALL PIPELINE VERDICT: PROCEED WITH CAUTION (WARNINGS PRESENT){RESET}")
        print("  Review the highlighted warnings and recommendations before training.")
    else:
        print(f"  {BOLD}{GREEN}OVERALL PIPELINE VERDICT: READY FOR MODEL TRAINING{RESET}")
    print("=" * 80 + "\n")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    base_dir_default = Path(__file__).resolve().parent
    default_north = "north_3months_derived.nc" if (base_dir_default / "north_3months_derived.nc").exists() else "north_2weeks_derived.nc"
    default_south = "south_3months_derived.nc" if (base_dir_default / "south_3months_derived.nc").exists() else "south_2weeks_derived.nc"

    parser = argparse.ArgumentParser(
        description="Verify end-to-end data pipeline integrity for ozone prediction before model training."
    )
    parser.add_argument("--base-dir", default=None, help="Root project directory (default: current directory)")
    parser.add_argument("--north-cams", default=default_north, help=f"North CAMS derived netCDF file (default: {default_north})")
    parser.add_argument("--south-cams", default=default_south, help=f"South CAMS derived netCDF file (default: {default_south})")
    parser.add_argument("--stations", default="stations_with_all_static_features.csv", help="Station metadata CSV")
    parser.add_argument("--o3", default="o3_measurements.csv", help="Ground-truth O3 measurements CSV")
    parser.add_argument("--base-table", default="base_training_table.csv", help="Base training table CSV")
    parser.add_argument("--final-table", default="final_training_table.csv", help="Final training table CSV")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(__file__).resolve().parent

    north_cams_path = base_dir / args.north_cams
    south_cams_path = base_dir / args.south_cams
    stations_path = base_dir / args.stations
    o3_path = base_dir / args.o3
    base_table_path = base_dir / args.base_table
    final_table_path = base_dir / args.final_table

    print(f"\n{BOLD}OZONE PREDICTION PIPELINE VERIFICATION RUNNER{RESET}")
    print(f"Working Directory: {base_dir}\n")

    section_results = {}

    # Section 1: Raw CAMS Data
    section_results["s1"] = verify_raw_cams_data(north_cams_path, south_cams_path)

    # Determine CAMS date range for section 4 validation
    cams_start, cams_end = None, None
    try:
        import xarray as xr
        with xr.open_dataset(north_cams_path) as ds:
            time_coord = "valid_time" if "valid_time" in ds else "time"
            t_vals = ds[time_coord].values
            cams_start = pd.to_datetime(t_vals.min(), utc=True)
            cams_end = pd.to_datetime(t_vals.max(), utc=True)
    except Exception:
        pass

    # Section 2: Station Metadata
    section_results["s2"] = verify_station_metadata(stations_path)

    # Section 3: Ground-Truth Ozone Data
    section_results["s3"] = verify_ground_truth_o3(o3_path, stations_path)

    # Section 4: Base Training Table
    section_results["s4"] = verify_base_training_table(base_table_path, cams_start, cams_end)

    # Section 5: Final Table with Wind-Lag Feature
    section_results["s5"] = verify_final_training_table(final_table_path)

    # Section 6: Final Summary
    print_final_summary(section_results)

    # Return non-zero exit code if any stage critically failed
    if any(res["status"] == "FAIL" for res in section_results.values()):
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
