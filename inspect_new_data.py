#!/usr/bin/env python3
"""
inspect_new_data.py

Inspects newly pulled 3-month CAMS historical NetCDF datasets:
1. Opens data_mlev.nc and data_sfc.nc from regional folders (north_3months and south_3months).
2. Prints forecast_reference_time range, forecast_period values, and total timesteps.
3. Computes and prints actual valid_time range (forecast_reference_time + forecast_period).
4. Confirms required variables (go3, no2, c5h8, u10, v10, t2m, d2m, sp) are present.
5. Confirms lat/lon bounding boxes match expected bounds:
   - North: 72-81E, 18-29.5N
   - South: 72-81E, 12-19.5N
6. Prints a concise summary of the valid time range for pulling ground-truth data:
   "3-month CAMS data covers [START DATE] to [END DATE]"
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Suppress backend warnings (e.g. cfgrib if missing eccodes)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray.backends.plugins")

# Required variables across datasets
REQUIRED_VARS = ["go3", "no2", "c5h8", "u10", "v10", "t2m", "d2m", "sp"]

# Expected bounding boxes: (lon_min, lat_min, lon_max, lat_max)
EXPECTED_BBOX = {
    "north": {
        "lon_min": 72.0,
        "lon_max": 81.0,
        "lat_min": 18.0,
        "lat_max": 29.5,
    },
    "south": {
        "lon_min": 72.0,
        "lon_max": 81.0,
        "lat_min": 12.0,
        "lat_max": 19.5,
    },
}


def find_region_folders(base_dir: Path) -> dict[str, Path]:
    """Finds north and south 3-month directories."""
    candidates_north = [base_dir / "north_3months", base_dir / "north_3month"]
    candidates_south = [base_dir / "south_3months", base_dir / "south_3month"]

    north_dir = next((p for p in candidates_north if p.is_dir()), None)
    south_dir = next((p for p in candidates_south if p.is_dir()), None)

    # Fallback to search if directory names vary
    if not north_dir:
        found = [p for p in base_dir.glob("*north*3month*") if p.is_dir()]
        if found:
            north_dir = found[0]
    if not south_dir:
        found = [p for p in base_dir.glob("*south*3month*") if p.is_dir()]
        if found:
            south_dir = found[0]

    return {"north": north_dir, "south": south_dir}


def inspect_dataset(region_name: str, region_dir: Path) -> dict:
    """Inspects NetCDF files in a region folder and validates schema/bounds."""
    print(f"\n{'='*70}")
    print(f"REGION: {region_name.upper()} ({region_dir.name})")
    print(f"Path: {region_dir.resolve()}")
    print(f"{'='*70}")

    mlev_path = region_dir / "data_mlev.nc"
    sfc_path = region_dir / "data_sfc.nc"

    if not mlev_path.exists():
        raise FileNotFoundError(f"Missing {mlev_path}")
    if not sfc_path.exists():
        raise FileNotFoundError(f"Missing {sfc_path}")

    # 1. Open datasets
    ds_mlev = xr.open_dataset(mlev_path)
    ds_sfc = xr.open_dataset(sfc_path)

    # Coords & Times
    ref_time_mlev = ds_mlev.forecast_reference_time.values
    ref_time_sfc = ds_sfc.forecast_reference_time.values
    assert (ref_time_mlev == ref_time_sfc).all(), f"{region_name}: Reference times differ between mlev and sfc!"

    period_mlev = ds_mlev.forecast_period.values
    period_sfc = ds_sfc.forecast_period.values
    assert (period_mlev == period_sfc).all(), f"{region_name}: Forecast periods differ between mlev and sfc!"

    num_ref_timesteps = len(ref_time_mlev)
    ref_min = pd.to_datetime(ref_time_mlev.min())
    ref_max = pd.to_datetime(ref_time_mlev.max())

    # Format periods
    period_deltas = [pd.to_timedelta(p) for p in period_mlev]
    period_strs = [str(p) for p in period_deltas]

    # 2. Print forecast_reference_time and forecast_period
    print("\n[1] FORECAST REFERENCE TIME & PERIODS:")
    print(f"  * Reference Time Min  : {ref_min.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  * Reference Time Max  : {ref_max.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  * Timestep Count      : {num_ref_timesteps} daily initialization steps")
    print(f"  * Available Periods   : {period_strs}")

    # 3. Compute actual valid_time range (forecast_reference_time + forecast_period)
    vt_grid = ds_mlev.forecast_reference_time + ds_mlev.forecast_period
    vt_min_overall = pd.to_datetime(vt_grid.values.min())
    vt_max_overall = pd.to_datetime(vt_grid.values.max())

    # Range for standard 1-day lead time used in CAMS pipeline (forecast_period == 1 days)
    p_1day = np.timedelta64(1, "D")
    has_1day = p_1day in period_mlev
    vt_min_1day = pd.to_datetime((ds_mlev.forecast_reference_time + p_1day).values.min()) if has_1day else None
    vt_max_1day = pd.to_datetime((ds_mlev.forecast_reference_time + p_1day).values.max()) if has_1day else None

    print("\n[2] VALID TIME RANGE (forecast_reference_time + forecast_period):")
    print(f"  * Overall Valid Time Min : {vt_min_overall.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  * Overall Valid Time Max : {vt_max_overall.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if has_1day:
        print(f"  * 1-Day Lead Valid Min   : {vt_min_1day.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  * 1-Day Lead Valid Max   : {vt_max_1day.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  * 1-Day Lead Timesteps   : {num_ref_timesteps}")

    # 4. Check variables
    mlev_vars = list(ds_mlev.data_vars)
    sfc_vars = list(ds_sfc.data_vars)
    combined_vars = set(mlev_vars) | set(sfc_vars)

    print("\n[3] VARIABLES VERIFICATION:")
    print(f"  * Model Level (mlev) Vars : {mlev_vars}")
    print(f"  * Surface (sfc) Vars     : {sfc_vars}")
    all_vars_ok = True
    for var in REQUIRED_VARS:
        present = var in combined_vars
        status = "OK [PASS]" if present else "MISSING [FAIL]"
        loc = "mlev" if var in mlev_vars else ("sfc" if var in sfc_vars else "NONE")
        print(f"    - {var:18s}: {status} (found in {loc})")
        if not present:
            all_vars_ok = False

    # 5. Check lat/lon bounding box
    lat_vals = ds_mlev.latitude.values
    lon_vals = ds_mlev.longitude.values
    lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())

    exp = EXPECTED_BBOX[region_name]
    lat_ok = (lat_min >= exp["lat_min"] - 1e-4) and (lat_max <= exp["lat_max"] + 1e-4)
    lon_ok = (lon_min >= exp["lon_min"] - 1e-4) and (lon_max <= exp["lon_max"] + 1e-4)
    bbox_ok = lat_ok and lon_ok

    print("\n[4] BOUNDING BOX VERIFICATION:")
    print(f"  * Expected Bounds : Lon [{exp['lon_min']:.1f}E, {exp['lon_max']:.1f}E], Lat [{exp['lat_min']:.1f}N, {exp['lat_max']:.1f}N]")
    print(f"  * Actual Bounds   : Lon [{lon_min:.2f}E, {lon_max:.2f}E], Lat [{lat_min:.2f}N, {lat_max:.2f}N]")
    print(f"  * Grid Shape      : {len(lat_vals)} lats x {len(lon_vals)} lons (res: ~0.4 deg)")
    print(f"  * Bounding Box OK : {'YES [PASS]' if bbox_ok else 'NO [MISMATCH]'}")

    ds_mlev.close()
    ds_sfc.close()

    return {
        "region": region_name,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "vt_min_overall": vt_min_overall,
        "vt_max_overall": vt_max_overall,
        "vt_min_1day": vt_min_1day,
        "vt_max_1day": vt_max_1day,
        "num_timesteps": num_ref_timesteps,
        "period_strs": period_strs,
        "all_vars_ok": all_vars_ok,
        "bbox_ok": bbox_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect new 3-month CAMS historical NetCDF datasets.")
    parser.add_argument("--north-dir", type=str, default=None, help="Path to north_3months directory")
    parser.add_argument("--south-dir", type=str, default=None, help="Path to south_3months directory")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    folders = find_region_folders(base_dir)

    north_dir = Path(args.north_dir) if args.north_dir else folders["north"]
    south_dir = Path(args.south_dir) if args.south_dir else folders["south"]

    if not north_dir or not north_dir.exists():
        print(f"ERROR: Could not locate north 3-month directory in {base_dir}")
        sys.exit(1)
    if not south_dir or not south_dir.exists():
        print(f"ERROR: Could not locate south 3-month directory in {base_dir}")
        sys.exit(1)

    print("======================================================================")
    print("           CAMS 3-MONTH HISTORICAL DATA INSPECTION REPORT             ")
    print("======================================================================")

    res_north = inspect_dataset("north", north_dir)
    res_south = inspect_dataset("south", south_dir)

    # Verify both regions align in time
    time_aligned = (
        res_north["ref_min"] == res_south["ref_min"]
        and res_north["ref_max"] == res_south["ref_max"]
        and res_north["vt_min_1day"] == res_south["vt_min_1day"]
        and res_north["vt_max_1day"] == res_south["vt_max_1day"]
    )

    all_passed = (
        res_north["all_vars_ok"]
        and res_south["all_vars_ok"]
        and res_north["bbox_ok"]
        and res_south["bbox_ok"]
        and time_aligned
    )

    start_date_1day = res_north["vt_min_1day"].strftime("%Y-%m-%d")
    end_date_1day = res_north["vt_max_1day"].strftime("%Y-%m-%d")

    start_date_all = res_north["vt_min_overall"].strftime("%Y-%m-%d")
    end_date_all = res_north["vt_max_overall"].strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print("                        FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"All variables present       : {'YES [PASS]' if (res_north['all_vars_ok'] and res_south['all_vars_ok']) else 'NO [FAIL]'}")
    print(f"All bounding boxes valid   : {'YES [PASS]' if (res_north['bbox_ok'] and res_south['bbox_ok']) else 'NO [FAIL]'}")
    print(f"Regions temporally aligned : {'YES [PASS]' if time_aligned else 'NO [MISMATCH]'}")
    print(f"Total reference timesteps  : {res_north['num_timesteps']}")
    print(f"Forecast periods available : {res_north['period_strs']}")
    print("----------------------------------------------------------------------")
    print(f"3-month CAMS data covers {start_date_1day} to {end_date_1day}")
    if end_date_all != end_date_1day:
        print(f"(Note: With full 2-day forecast period included, coverage extends to {end_date_all})")
    print("----------------------------------------------------------------------")
    print("Recommended ground-truth date range to pull next:")
    print(f"  Start : {start_date_1day}T00:00:00Z")
    print(f"  End   : {end_date_1day}T23:59:59Z (or {end_date_all}T00:00:00Z)")
    print(f"{'='*70}\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
