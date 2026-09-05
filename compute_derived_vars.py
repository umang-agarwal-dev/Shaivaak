import os
from pathlib import Path
import numpy as np
import xarray as xr


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

    print("Loading and processing north_2weeks...")
    ds_north = load_region(base_dir / "north_2weeks")
    ds_north = compute_derived_wind_vars(ds_north)

    print("Loading and processing south_2weeks...")
    ds_south = load_region(base_dir / "south_2weeks")
    ds_south = compute_derived_wind_vars(ds_south)

    output_north = base_dir / "north_2weeks_derived.nc"
    output_south = base_dir / "south_2weeks_derived.nc"

    print(f"Saving north dataset to {output_north.name}...")
    ds_north.to_netcdf(output_north)

    print(f"Saving south dataset to {output_south.name}...")
    ds_south.to_netcdf(output_south)

    print("\n--- Sanity Checks (First 5 Values) ---")
    print("North 2 Weeks:")
    print("  wind_speed (first 5 values):", ds_north["wind_speed"].values.flatten()[:5])
    print("  wind_direction_deg (first 5 values):", ds_north["wind_direction_deg"].values.flatten()[:5])

    print("\nSouth 2 Weeks:")
    print("  wind_speed (first 5 values):", ds_south["wind_speed"].values.flatten()[:5])
    print("  wind_direction_deg (first 5 values):", ds_south["wind_direction_deg"].values.flatten()[:5])
    print("\nProcessing complete successfully.")


if __name__ == "__main__":
    main()
