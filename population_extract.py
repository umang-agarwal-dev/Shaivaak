import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.windows import Window

REMOTE_WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020.tif"
VSICURL_PATH = f"/vsicurl/{REMOTE_WORLDPOP_URL}"


def find_station_file(base_dir: Path, custom_path: str | None = None) -> Path:
    """Finds the existing station file, prioritizing stations_with_terrain.csv, then stations_metadata.csv."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified input station file not found: {custom_path}")

    candidates = [
        base_dir / "stations_with_terrain.csv",
        base_dir / "stations_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback search for any csv containing 'station'
    matches = list(base_dir.glob("*station*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find any stations CSV file (e.g. stations_with_terrain.csv or stations_metadata.csv).")


def extract_via_worldpop_api(lat: float, lon: float, year: int = 2020) -> float:
    """
    Extracts population count from WorldPop REST API for a ~100m grid cell
    centered at (lat, lon) as a fallback if /vsicurl/ HTTP Range is rejected by the server.
    """
    # 100m pixel is roughly ~0.000833 degrees; half-pixel is ~0.0004167 degrees
    half = 0.0004167
    poly = {
        "type": "Polygon",
        "coordinates": [[
            [lon - half, lat - half],
            [lon + half, lat - half],
            [lon + half, lat + half],
            [lon - half, lat + half],
            [lon - half, lat - half],
        ]]
    }
    url = f"https://api.worldpop.org/v1/services/stats?dataset=wpgppop&year={year}&geojson={json.dumps(poly)}"

    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()

        # If asynchronous task, poll for completion
        task_id = data.get("taskid")
        if task_id:
            for _ in range(15):
                time.sleep(1)
                task_res = requests.get(f"https://api.worldpop.org/v1/tasks/{task_id}", timeout=15)
                if task_res.status_code == 200:
                    task_data = task_res.json()
                    if task_data.get("status") == "finished":
                        return float(task_data.get("data", {}).get("total_population", np.nan))
                    elif task_data.get("error"):
                        break

        # If returned directly
        if "data" in data and "total_population" in data["data"]:
            return float(data["data"]["total_population"])

    except Exception as e:
        print(f"  [!] WorldPop API fallback query failed for ({lat}, {lon}): {e}")

    return np.nan


def extract_population(df: pd.DataFrame, raster_path: str) -> pd.DataFrame:
    """
    Attempts to read population density via rasterio using a 1x1 window
    with /vsicurl/. If the remote server refuses HTTP Range requests,
    gracefully falls back to the WorldPop Stats API.
    """
    print(f"\n[*] Attempting to access raster via rasterio:\n    {raster_path}")
    pop_values = []
    used_vsicurl = False

    try:
        with rasterio.open(raster_path) as src:
            used_vsicurl = True
            print(f"[+] Successfully opened raster with rasterio!")
            print(f"    Dimensions: {src.width} x {src.height}, CRS: {src.crs}")
            print(f"    Bounds: {src.bounds}")

            b_left, b_bottom, b_right, b_top = src.bounds
            nodata_val = src.nodata

            for idx, row in df.iterrows():
                lat = float(row["lat"])
                lon = float(row["lon"])
                stn_name = row.get("name", f"Station_{row.get('station_id', idx)}")

                # Check if station falls outside raster bounds
                if not (b_left <= lon <= b_right and b_bottom <= lat <= b_top):
                    print(f"  [!] Warning: Station '{stn_name}' ({lat}, {lon}) is outside raster bounds! Setting to NaN.")
                    pop_values.append(np.nan)
                    continue

                # Get pixel row, col and read 1x1 window
                try:
                    row_idx, col_idx = src.index(lon, lat)
                    window = Window(col_off=col_idx, row_off=row_idx, width=1, height=1)
                    val = src.read(1, window=window)[0, 0]

                    # Check for nodata or invalid values
                    if (nodata_val is not None and val == nodata_val) or np.isnan(val) or val < 0:
                        print(f"  [!] Warning: Station '{stn_name}' pixel is nodata ({val}). Setting to NaN.")
                        pop_values.append(np.nan)
                    else:
                        pop_values.append(float(val))
                except Exception as ex:
                    print(f"  [!] Warning reading pixel for '{stn_name}': {ex}. Setting to NaN.")
                    pop_values.append(np.nan)

    except rasterio.errors.RasterioIOError as e:
        print(f"\n[!] Note: rasterio /vsicurl/ could not perform byte-range requests directly:")
        print(f"    Error: {e}")
        print("[*] WorldPop server (data.worldpop.org) blocks HTTP Range requests on .tif files.")
        print("[*] Switching to official WorldPop Point Stats API for coordinate extraction...")

        for idx, row in df.iterrows():
            lat = float(row["lat"])
            lon = float(row["lon"])
            stn_name = row.get("name", f"Station_{row.get('station_id', idx)}")

            # Check general India bounds (lat 6-37, lon 68-97.5)
            if not (68.0 <= lon <= 97.5 and 6.0 <= lat <= 37.0):
                print(f"  [!] Warning: Station '{stn_name}' ({lat}, {lon}) is outside India bounds! Setting to NaN.")
                pop_values.append(np.nan)
                continue

            print(f"  [*] Querying WorldPop population for '{stn_name}' ({lat:.4f}, {lon:.4f})...")
            pop_val = extract_via_worldpop_api(lat, lon, year=2020)
            if np.isnan(pop_val):
                print(f"  [!] Warning: Station '{stn_name}' returned NaN/nodata.")
            pop_values.append(pop_val)

    df_out = df.copy()
    df_out["population_density"] = pop_values
    return df_out


def cleanup_partial_downloads(base_dir: Path):
    """Deletes any partially downloaded india_worldpop.tif if present."""
    target = base_dir / "data" / "raw" / "india_worldpop.tif"
    if target.exists():
        try:
            target.unlink()
            print(f"[*] Deleted partially downloaded file: {target}")
        except Exception as e:
            print(f"[!] Note: Could not delete partial file {target}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract population density for monitoring stations.")
    parser.add_argument("--input", default=None, help="Input station CSV file path")
    parser.add_argument("--output", default="stations_with_all_static_features.csv", help="Output CSV path")
    parser.add_argument("--raster-url", default=VSICURL_PATH, help="Raster path or /vsicurl/ URL")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Clean up partial download if any
    cleanup_partial_downloads(base_dir)

    # 1. Load station file
    station_file = find_station_file(base_dir, args.input)
    print(f"[*] Loading station coordinates from: {station_file.name}")
    df_stations = pd.read_csv(station_file)

    if "lat" not in df_stations.columns or "lon" not in df_stations.columns:
        raise ValueError(f"Station file {station_file} must contain 'lat' and 'lon' columns. Found: {list(df_stations.columns)}")

    print(f"[*] Total stations to process: {len(df_stations)}")

    # 2 & 3 & 4. Extract population density
    df_result = extract_population(df_stations, args.raster_url)

    # 5. Save result as stations_with_all_static_features.csv
    output_path = base_dir / args.output
    df_result.to_csv(output_path, index=False)
    print(f"\n[+] Saved results with population_density to:\n    {output_path.resolve()}")

    # 6. Print each station's population value for sanity check
    print("\n" + "=" * 65)
    print("                SANITY CHECK: EXTRACTED VALUES")
    print("=" * 65)
    for idx, row in df_result.iterrows():
        stn_id = row.get("station_id", idx)
        stn_name = row.get("name", "Unknown")
        lat = row["lat"]
        lon = row["lon"]
        pop = row["population_density"]
        pop_str = f"{pop:.2f} people/pixel" if pd.notna(pop) else "NaN"
        print(f"  Station [{stn_id}] {stn_name}")
        print(f"    Coordinates: ({lat:.5f}, {lon:.5f}) | Population Density: {pop_str}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
