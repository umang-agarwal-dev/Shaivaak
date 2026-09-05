import xarray as xr

def load_region(folder):
    ds_mlev = xr.open_dataset(f"{folder}/data_mlev.nc", engine="netcdf4")
    ds_sfc = xr.open_dataset(f"{folder}/data_sfc.nc", engine="netcdf4")

    if "model_level" in ds_mlev.dims:
        ds_mlev = ds_mlev.squeeze("model_level", drop=True)

    # compute actual valid time = reference time + forecast lead time
    for ds in [ds_mlev, ds_sfc]:
        if "valid_time" not in ds.coords:
            ds.coords["valid_time"] = ds.forecast_reference_time + ds.forecast_period

    # sanity check on the real coordinates instead
    assert (ds_mlev.forecast_reference_time.values == ds_sfc.forecast_reference_time.values).all(), f"{folder}: reference time mismatch"
    assert (ds_mlev.forecast_period.values == ds_sfc.forecast_period.values).all(), f"{folder}: forecast period mismatch"

    return xr.merge([ds_mlev, ds_sfc])

ds_north = load_region("north_2weeks")
ds_south = load_region("south_2weeks")

print(ds_north.data_vars)
print(ds_north.coords)