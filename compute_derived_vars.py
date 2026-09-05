"""
compute_derived_vars.py

Processes CAMS netCDF datasets for North and South regional domains:
1. Opens data_mlev.nc and data_sfc.nc for each region.
2. Squeezes the model_level dimension.
3. Computes valid_time coordinate = forecast_reference_time + forecast_period.
4. Validates coordinate alignment between surface and model level files.
5. Merges datasets with compat='no_conflicts' and selects forecast_period == '1 days'.
6. Computes derived wind variables:
   - wind_speed = sqrt(u10^2 + v10^2)
   - wind_direction_deg = (270 - degrees(atan2(v10, u10))) % 360
7. Saves north_3months_derived.nc and south_3months_derived.nc.
"""

import os
import argparse
import warnings
from pathlib import Path
import numpy as np
import xarray as xr

# Suppress engine warnings from plugins
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray.backends.plugins")


def load_region(folder: str | Path) -> xr.Dataset:
    """
    Loads and merges data_mlev.nc and data_sfc.nc for a given region folder.
    
    Steps:
    - Squeezes the 'model_level' dimension if present.
    - Ensures 'valid_time' coordinate is computed.
    - Validates forecast_reference_time and forecast_period alignment.
    - Merges datasets with compat='no_conflicts'.
    - Selects forecast_period == '1 days' only.
    """
    folder_path = Path(folder)
    ds_mlev = xr.open_dataset(folder_path / "data_mlev.nc", engine="netcdf4")
    ds_sfc = xr.open_dataset(folder_path / "data_sfc.nc", engine="netcdf4")

    if "model_level" in ds_mlev.dims:
        ds_mlev = ds_mlev.squeeze("model_level", drop=True)

    # Compute actual valid time = reference time + forecast lead time
    for ds in [ds_mlev, ds_sfc]:
        if "valid_time" not in ds.coords:
            ds.coords["valid_time"] = ds.forecast_reference_time + ds.forecast_period

    # Sanity checks on coordinates
    assert (ds_mlev.forecast_reference_time.values == ds_sfc.forecast_reference_time.values).all(), (
        f"{folder}: reference time mismatch"
    )
    assert (ds_mlev.forecast_period.values == ds_sfc.forecast_period.values).all(), (
        f"{folder}: forecast period mismatch"
    )

    # Merge datasets with no conflicts and select forecast_period == '1 days'
    ds_merged = xr.merge([ds_mlev, ds_sfc], compat="no_conflicts")
    ds_merged = ds_merged.sel(forecast_period="1 days")

    return ds_merged


def compute_derived_wind_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes derived wind variables using vectorized xarray/numpy operations:
    - wind_speed = sqrt(u10^2 + v10^2)
    - wind_direction_deg = (270 - degrees(atan2(v10, u10))) % 360
      (meteorological convention: direction the wind is blowing FROM, 0=North, 90=East)
    """
    u10 = ds["u10"]
    v10 = ds["v10"]

    # 10m Wind Speed (m/s)
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_speed.attrs["units"] = "m s**-1"
    wind_speed.attrs["long_name"] = "10 metre wind speed"

    # 10m Wind Direction (degrees from North)
    wind_dir_rad = np.arctan2(v10, u10)
    wind_direction_deg = (270.0 - np.rad2deg(wind_dir_rad)) % 360.0
    wind_direction_deg.attrs["units"] = "degrees"
    wind_direction_deg.attrs["long_name"] = "10 metre wind direction (from)"

    ds["wind_speed"] = wind_speed
    ds["wind_direction_deg"] = wind_direction_deg

    return ds


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Compute derived CAMS variables (wind speed, wind direction).")
    parser.add_argument(
        "--north-dir",
        default=str(base_dir / "north_3months"),
        help="Path to north regional directory (default: ./north_3months)",
    )
    parser.add_argument(
        "--south-dir",
        default=str(base_dir / "south_3months"),
        help="Path to south regional directory (default: ./south_3months)",
    )
    parser.add_argument(
        "--north-out",
        default=str(base_dir / "north_3months_derived.nc"),
        help="Path to save north derived netCDF (default: ./north_3months_derived.nc)",
    )
    parser.add_argument(
        "--south-out",
        default=str(base_dir / "south_3months_derived.nc"),
        help="Path to save south derived netCDF (default: ./south_3months_derived.nc)",
    )
    args = parser.parse_args()

    north_dir = Path(args.north_dir)
    south_dir = Path(args.south_dir)
    output_north = Path(args.north_out)
    output_south = Path(args.south_out)

    print("=" * 70)
    print("           COMPUTE DERIVED VARIABLES (3-MONTH CAMS DATA)")
    print("=" * 70)

    print(f"\n[*] Loading and processing North region from: {north_dir.name}...")
    ds_north = load_region(north_dir)
    ds_north = compute_derived_wind_vars(ds_north)

    print(f"[*] Loading and processing South region from: {south_dir.name}...")
    ds_south = load_region(south_dir)
    ds_south = compute_derived_wind_vars(ds_south)

    print(f"\n[*] Saving North derived dataset to: {output_north.name}...")
    ds_north.to_netcdf(output_north)
    print(f"[+] Saved {output_north.name} ({output_north.stat().st_size / (1024*1024):.2f} MB)")

    print(f"[*] Saving South derived dataset to: {output_south.name}...")
    ds_south.to_netcdf(output_south)
    print(f"[+] Saved {output_south.name} ({output_south.stat().st_size / (1024*1024):.2f} MB)")

    print("\n--- Summary and Verification ---")
    print(f"North 3 Months Derived:")
    print(f"  Shape/Dims : {dict(ds_north.sizes)}")
    print(f"  Valid Time : {ds_north.valid_time.values.min()} to {ds_north.valid_time.values.max()}")
    print(f"  wind_speed (first 5) : {ds_north['wind_speed'].values.flatten()[:5].round(3)}")
    print(f"  wind_dir   (first 5) : {ds_north['wind_direction_deg'].values.flatten()[:5].round(1)}")

    print(f"\nSouth 3 Months Derived:")
    print(f"  Shape/Dims : {dict(ds_south.sizes)}")
    print(f"  Valid Time : {ds_south.valid_time.values.min()} to {ds_south.valid_time.values.max()}")
    print(f"  wind_speed (first 5) : {ds_south['wind_speed'].values.flatten()[:5].round(3)}")
    print(f"  wind_dir   (first 5) : {ds_south['wind_direction_deg'].values.flatten()[:5].round(1)}")

    print("\n[+] Derived variable computation completed successfully.\n")


if __name__ == "__main__":
    main()
