"""
build_base_table.py

Builds the base training table for ozone modeling by:
1. Loading CAMS netCDF datasets (north and south regions) and station metadata with static features.
2. Selecting the appropriate regional CAMS dataset for each station based on region / latitude.
3. Bilinearly interpolating all CAMS variables to the exact station point.
4. Converting each station's interpolated time series to a DataFrame.
5. Computing day_length_hours per station per date using the astral library.
6. Concatenating all stations into a single base DataFrame.
7. Left-joining static features (dist_to_coast_km, elevation_mean_20km, elevation_std_20km).
8. Left-joining o3_measurements.csv on (station_id, matching timestamp rounded to nearest hour).
9. Saving the resulting table to base_training_table.csv.
10. Printing final table shape and non-null value counts per column.
"""

import os
import sys
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from astral import LocationInfo
from astral.sun import sun


# Required CAMS variables to extract
CAMS_VARIABLES = [
    "go3",
    "no2",
    "c5h8",
    "t2m",
    "d2m",
    "sp",
    "wind_speed",
    "wind_direction_deg",
]

# Static feature columns to include
STATIC_FEATURE_COLS = [
    "dist_to_coast_km",
    "elevation_mean_20km",
    "elevation_std_20km",
]


def load_cams_datasets(north_path: Path, south_path: Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Loads north and south CAMS derived netCDF datasets."""
    if not north_path.exists():
        raise FileNotFoundError(f"North CAMS dataset not found: {north_path.resolve()}")
    if not south_path.exists():
        raise FileNotFoundError(f"South CAMS dataset not found: {south_path.resolve()}")

    print(f"[*] Loading CAMS north dataset from: {north_path.name}")
    ds_north = xr.open_dataset(north_path)
    print(f"[*] Loading CAMS south dataset from: {south_path.name}")
    ds_south = xr.open_dataset(south_path)

    return ds_north, ds_south


def load_stations_data(base_dir: Path, custom_path: str | None = None) -> pd.DataFrame:
    """
    Loads station metadata and static features, ensuring required static feature columns
    are present (falling back to stations_with_terrain.csv or stations_with_coast_dist.csv if needed).
    """
    if custom_path:
        stn_path = Path(custom_path)
        if not stn_path.exists():
            raise FileNotFoundError(f"Specified stations CSV not found: {custom_path}")
    else:
        candidates = [
            base_dir / "stations_with_all_static_features.csv",
            base_dir / "stations_with_terrain.csv",
            base_dir / "stations_with_coast_dist.csv",
            base_dir / "stations_metadata.csv",
        ]
        stn_path = None
        for c in candidates:
            if c.exists():
                stn_path = c
                break
        if stn_path is None:
            raise FileNotFoundError("Could not find any stations CSV file.")

    print(f"[*] Loading station metadata from: {stn_path.name}")
    df_stn = pd.read_csv(stn_path)

    # Check for missing static feature columns and try to fill from companion CSVs
    missing_static = [col for col in STATIC_FEATURE_COLS if col not in df_stn.columns]
    if missing_static:
        companion_files = [
            base_dir / "stations_with_terrain.csv",
            base_dir / "stations_with_coast_dist.csv",
        ]
        for comp in companion_files:
            if comp.exists() and comp != stn_path:
                df_comp = pd.read_csv(comp)
                for col in missing_static:
                    if col in df_comp.columns and col not in df_stn.columns:
                        merged_col = df_comp.set_index("station_id")[col]
                        df_stn[col] = df_stn["station_id"].map(merged_col)

        still_missing = [col for col in STATIC_FEATURE_COLS if col not in df_stn.columns]
        if still_missing:
            print(f"  [!] Note: Static feature columns {still_missing} not found; initializing with NaN.")
            for col in still_missing:
                df_stn[col] = np.nan

    return df_stn


@lru_cache(maxsize=10000)
def compute_day_length_hours(lat: float, lon: float, date_obj) -> float:
    """
    Computes daylight duration in hours for a given lat, lon, and date using Astral.
    """
    try:
        loc = LocationInfo(name="Station", region="India", timezone="UTC", latitude=lat, longitude=lon)
        s = sun(loc.observer, date=date_obj)
        sunrise = s["sunrise"]
        sunset = s["sunset"]
        duration_hours = (sunset - sunrise).total_seconds() / 3600.0
        return float(round(duration_hours, 4))
    except Exception as e:
        print(f"  [!] Warning computing day length for ({lat}, {lon}) on {date_obj}: {e}")
        return np.nan


def interpolate_station_cams(
    ds_north: xr.Dataset,
    ds_south: xr.Dataset,
    station_row: pd.Series,
) -> pd.DataFrame:
    """
    Selects the regional CAMS dataset, bilinearly interpolates all variables
    to the station's lat/lon, and computes day_length_hours for each timestamp.
    """
    stn_id = station_row["station_id"]
    stn_name = station_row.get("name", f"Station_{stn_id}")
    lat = float(station_row["lat"])
    lon = float(station_row["lon"])
    region = str(station_row.get("region", "")).strip().lower()

    # Determine region dataset
    if region == "north":
        selected_ds = ds_north
        region_label = "north"
    elif region == "south":
        selected_ds = ds_south
        region_label = "south"
    else:
        # Geographic fallback based on latitude
        if lat >= 18.0:
            selected_ds = ds_north
            region_label = "north (inferred)"
        else:
            selected_ds = ds_south
            region_label = "south (inferred)"

    print(f"  [*] Station [{stn_id}] '{stn_name}' ({lat:.4f}, {lon:.4f}) -> using {region_label} CAMS dataset")

    # Bilinear interpolation of all variables at the exact station coordinate
    stn_interp = selected_ds.interp(latitude=lat, longitude=lon, method="linear")

    # Available variables to extract
    vars_to_extract = [v for v in CAMS_VARIABLES if v in stn_interp.data_vars]
    df_stn = stn_interp[vars_to_extract].to_dataframe().reset_index()

    # Ensure station_id is set
    df_stn["station_id"] = stn_id

    # Ensure valid_time is retained
    if "valid_time" not in df_stn.columns:
        if "forecast_reference_time" in df_stn.columns and "forecast_period" in df_stn.columns:
            df_stn["valid_time"] = df_stn["forecast_reference_time"] + df_stn["forecast_period"]
        elif "time" in df_stn.columns:
            df_stn["valid_time"] = df_stn["time"]
        else:
            raise KeyError(f"Could not determine valid_time coordinate for station {stn_id}")

    # Standardize valid_time as ISO string or timestamp
    df_stn["valid_time"] = pd.to_datetime(df_stn["valid_time"], utc=True)

    # Compute day_length_hours per row date using Astral
    day_lengths = []
    for vt in df_stn["valid_time"]:
        row_date = vt.date()
        dl = compute_day_length_hours(lat, lon, row_date)
        day_lengths.append(dl)

    df_stn["day_length_hours"] = day_lengths

    # Reorder base columns
    desired_cols = ["station_id", "valid_time"] + vars_to_extract + ["day_length_hours"]
    return df_stn[desired_cols]


def main():
    parser = argparse.ArgumentParser(
        description="Build base training table combining CAMS predictions, static features, and OpenAQ ground truth."
    )
    base_dir = Path(__file__).resolve().parent
    default_north = "north_3months_derived.nc" if (base_dir / "north_3months_derived.nc").exists() else "north_2weeks_derived.nc"
    default_south = "south_3months_derived.nc" if (base_dir / "south_3months_derived.nc").exists() else "south_2weeks_derived.nc"

    parser.add_argument(
        "--stations",
        default=None,
        help="Path to station static features CSV (default: auto-detect stations_with_all_static_features.csv)",
    )
    parser.add_argument(
        "--north-cams",
        default=default_north,
        help=f"Path to north region derived CAMS dataset (default: {default_north})",
    )
    parser.add_argument(
        "--south-cams",
        default=default_south,
        help=f"Path to south region derived CAMS dataset (default: {default_south})",
    )
    parser.add_argument(
        "--o3",
        default="o3_measurements.csv",
        help="Path to ground truth o3_measurements.csv",
    )
    parser.add_argument(
        "--output",
        default="base_training_table.csv",
        help="Output CSV path (default: base_training_table.csv)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("                     BUILD BASE TRAINING TABLE")
    print("=" * 75)

    # 1. Load CAMS datasets and stations CSV
    north_path = base_dir / args.north_cams
    south_path = base_dir / args.south_cams
    ds_north, ds_south = load_cams_datasets(north_path, south_path)

    df_stations = load_stations_data(base_dir, args.stations)
    print(f"[*] Found {len(df_stations)} station(s) to process.")

    # 2, 3, 4, 5. Interpolate CAMS variables and calculate day_length_hours per station
    print("\n[*] Interpolating CAMS variables to exact station coordinates...")
    station_dfs = []
    for idx, row in df_stations.iterrows():
        df_stn_cams = interpolate_station_cams(ds_north, ds_south, row)
        station_dfs.append(df_stn_cams)

    # 6. Concatenate all stations into one long DataFrame
    df_base = pd.concat(station_dfs, ignore_index=True)
    print(f"\n[*] Concatenated base CAMS records: {len(df_base)} total rows across {len(station_dfs)} station(s).")

    # 7. Left-join static features on station_id
    print(f"[*] Joining static features: {STATIC_FEATURE_COLS}...")
    static_df = df_stations[["station_id"] + STATIC_FEATURE_COLS].drop_duplicates(subset=["station_id"])
    df_base = df_base.merge(static_df, on="station_id", how="left")

    # 8. Left-join o3_measurements.csv on (station_id, matching timestamp rounded to nearest hour)
    o3_path = base_dir / args.o3
    if o3_path.exists():
        print(f"[*] Loading ground-truth measurements from: {o3_path.name}")
        df_o3 = pd.read_csv(o3_path)

        # Round timestamps to nearest hour
        df_base["_time_round"] = pd.to_datetime(df_base["valid_time"], format="ISO8601", utc=True).dt.round("h")
        df_o3["_time_round"] = pd.to_datetime(df_o3["datetime_utc"], format="ISO8601", utc=True).dt.round("h")

        # In case multiple measurements round to the same hour, take the average o3_value
        df_o3_clean = (
            df_o3.groupby(["station_id", "_time_round"], as_index=False)["o3_value"]
            .mean()
        )

        print(f"[*] Merging ground-truth o3_value (nearest-hour alignment)...")
        df_base = df_base.merge(
            df_o3_clean[["station_id", "_time_round", "o3_value"]],
            on=["station_id", "_time_round"],
            how="left",
        )
        df_base = df_base.drop(columns=["_time_round"])
    else:
        print(f"  [!] Warning: o3 ground-truth file not found at {o3_path.resolve()}. Adding empty o3_value column.")
        df_base["o3_value"] = np.nan

    # Format valid_time back to clean ISO8601 string for CSV export
    df_base["valid_time"] = df_base["valid_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 9. Save result as base_training_table.csv
    output_path = base_dir / args.output
    df_base.to_csv(output_path, index=False)
    print(f"\n[+] Saved base training table to:\n    {output_path.resolve()}")

    # 10. Print shape and count of non-null values per column
    print("\n" + "=" * 75)
    print("                    TABLE SUMMARY & DATA AUDIT")
    print("=" * 75)
    print(f"Final Table Shape: {df_base.shape} (rows: {df_base.shape[0]}, columns: {df_base.shape[1]})")
    print("\nNon-Null Value Counts per Column:")
    print(f"  {'Column Name':<25} | {'Non-Null Count':<16} | {'Null Count':<12} | {'Data Type':<15}")
    print("  " + "-" * 73)
    for col in df_base.columns:
        non_null = int(df_base[col].notnull().sum())
        null_count = int(df_base[col].isnull().sum())
        dtype_str = str(df_base[col].dtype)
        print(f"  {col:<25} | {non_null:<16} | {null_count:<12} | {dtype_str:<15}")
    print("=" * 75 + "\n")

    # Close CAMS datasets
    ds_north.close()
    ds_south.close()


if __name__ == "__main__":
    main()
