"""
dem_elevation.py

Downloads Copernicus GLO-30 DEM tiles covering monitoring stations from AWS Open Data
(s3://copernicus-dem-30m / https://copernicus-dem-30m.s3.amazonaws.com/), merges adjacent
tiles if a station's radius spans tile boundaries, and computes 20km elevation statistics
(elevation_mean_20km, elevation_std_20km).
"""

import os
import sys
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import rasterio
from rasterio.windows import from_bounds
import rasterio.merge

COP_DEM_S3_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
KM_PER_DEGREE_LAT = 111.0  # 1 degree latitude ≈ 111 km

# Reference benchmark locations for validation
BENCHMARK_STATIONS = [
    {"name": "Bengaluru (Inland Plateau)", "lat": 12.9716, "lon": 77.5946, "expected": "~900m"},
    {"name": "Chennai (Coastal)", "lat": 13.0827, "lon": 80.2707, "expected": "Near sea level (0-20m)"},
    {"name": "Mumbai (Coastal)", "lat": 18.9220, "lon": 72.8340, "expected": "Near sea level (0-20m)"},
    {"name": "Delhi (Inland Plain)", "lat": 28.6139, "lon": 77.2090, "expected": "~215-225m"},
]


def find_station_file(base_dir: Path, custom_path: str | None = None) -> Path:
    """Finds input station CSV file, prioritizing stations_with_coast_dist.csv."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified input station file not found: {custom_path}")

    candidates = [
        base_dir / "stations_with_coast_dist.csv",
        base_dir / "stations_with_all_static_features.csv",
        base_dir / "stations_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(base_dir.glob("*station*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find any stations CSV file (e.g. stations_with_coast_dist.csv).")


def get_tile_name(lat_deg: int, lon_deg: int) -> str:
    """
    Generates Copernicus GLO-30 tile name for a given 1x1 degree lower-left integer coordinate.
    Example: (28, 77) -> 'Copernicus_DSM_COG_10_N28_00_E077_00_DEM'
    """
    lat_prefix = "N" if lat_deg >= 0 else "S"
    lon_prefix = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{lat_prefix}{abs(lat_deg):02d}_00_{lon_prefix}{abs(lon_deg):03d}_00_DEM"


def compute_station_bounding_box(lat: float, lon: float, radius_km: float = 20.0) -> dict:
    """
    Computes geographic bounding box (min_lon, min_lat, max_lon, max_lat) and required
    Copernicus DEM 1x1 degree tile names for a given station coordinate and radius.
    """
    # 1 degree latitude ≈ 111 km
    delta_lat = radius_km / KM_PER_DEGREE_LAT
    # Longitude adjusted by cos(latitude)
    cos_lat = math.cos(math.radians(lat))
    delta_lon = radius_km / (KM_PER_DEGREE_LAT * max(cos_lat, 0.01))

    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon

    # Determine 1x1 degree grid cells intersecting the box
    lat_min_idx = int(math.floor(min_lat))
    lat_max_idx = int(math.floor(max_lat))
    lon_min_idx = int(math.floor(min_lon))
    lon_max_idx = int(math.floor(max_lon))

    tiles = []
    for l_idx in range(lat_min_idx, lat_max_idx + 1):
        for m_idx in range(lon_min_idx, lon_max_idx + 1):
            tiles.append(get_tile_name(l_idx, m_idx))

    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "tiles": sorted(list(set(tiles))),
    }


def download_dem_tile(tile_name: str, dest_dir: Path, force: bool = False) -> Path | None:
    """
    Downloads a single Copernicus GLO-30 DEM tile COG from AWS Open Data bucket
    into dest_dir if not already present.
    Returns Path to local tif file, or None if tile does not exist (e.g. open ocean).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / f"{tile_name}.tif"

    if target_path.exists() and not force:
        file_size = target_path.stat().st_size
        if file_size > 1024:  # Valid non-empty file
            return target_path

    url = f"{COP_DEM_S3_BASE_URL}/{tile_name}/{tile_name}.tif"
    part_path = dest_dir / f"{tile_name}.tif.part"

    print(f"[*] Downloading tile {tile_name} from AWS S3:")
    print(f"    URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            print(f"  [!] Tile {tile_name} returned 404 Not Found on S3 (likely open ocean/no land).")
            return None
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB chunk

        with open(part_path, "wb") as f, tqdm(
            desc=tile_name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))

        # Rename partial file to final target
        if part_path.exists():
            if target_path.exists():
                target_path.unlink()
            part_path.rename(target_path)

        file_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"  [+] Saved {target_path.name} ({file_mb:.2f} MB)")
        return target_path

    except Exception as ex:
        if part_path.exists():
            part_path.unlink()
        print(f"  [!] Error downloading {tile_name}: {ex}")
        raise ex


def extract_elevation_for_station(
    lat: float,
    lon: float,
    tile_paths: list[Path],
    radius_km: float = 20.0,
) -> tuple[float, float]:
    """
    Extracts elevation values in a ~20km window around (lat, lon).
    If multiple tiles are provided, merges them with rasterio.merge before extracting.
    Filters nodata and computes mean and std.
    """
    bbox = compute_station_bounding_box(lat, lon, radius_km=radius_km)
    min_lon, min_lat = bbox["min_lon"], bbox["min_lat"]
    max_lon, max_lat = bbox["max_lon"], bbox["max_lat"]

    if not tile_paths:
        return np.nan, np.nan

    # Open relevant tile(s)
    opened_datasets = []
    try:
        for tp in tile_paths:
            opened_datasets.append(rasterio.open(tp))

        if len(opened_datasets) == 1:
            src = opened_datasets[0]
            win = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
            elevation_data = src.read(1, window=win)
            nodata_val = src.nodata
        else:
            # Multi-tile case: merge tiles covering the 20km radius window
            mosaic, _ = rasterio.merge.merge(
                opened_datasets,
                bounds=(min_lon, min_lat, max_lon, max_lat),
            )
            elevation_data = mosaic[0]
            nodata_val = opened_datasets[0].nodata

    finally:
        for ds in opened_datasets:
            ds.close()

    # Filter out nodata values
    # Nodata can be None, -32768, -99999, etc.
    valid_mask = np.isfinite(elevation_data)
    if nodata_val is not None:
        valid_mask &= (elevation_data != nodata_val)
    # Generic safeguard for DEM nodata fill values
    valid_mask &= (elevation_data > -10000)

    valid_values = elevation_data[valid_mask]

    if valid_values.size == 0:
        return np.nan, np.nan

    elev_mean = float(np.mean(valid_values))
    elev_std = float(np.std(valid_values))

    return round(elev_mean, 2), round(elev_std, 2)


def process_stations(
    df: pd.DataFrame,
    dem_dir: Path,
    radius_km: float = 20.0,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Determines needed tiles for all stations in df, downloads only needed tiles,
    and extracts elevation mean and std.
    """
    # 1. Determine all tiles needed across all stations
    station_bboxes = []
    all_needed_tiles = set()

    for idx, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        stn_bbox = compute_station_bounding_box(lat, lon, radius_km=radius_km)
        station_bboxes.append(stn_bbox)
        all_needed_tiles.update(stn_bbox["tiles"])

    print(f"\n[*] Step 1: Determined required DEM tiles for {len(df)} station(s):")
    print(f"    Unique 1x1 DEM tiles required ({len(all_needed_tiles)}): {sorted(list(all_needed_tiles))}")

    # 2. Download only the needed tiles into dem_dir
    print(f"\n[*] Step 2: Downloading required DEM tiles into {dem_dir.resolve()}...")
    tile_paths_map = {}
    for tile in sorted(list(all_needed_tiles)):
        path = download_dem_tile(tile, dem_dir, force=force_download)
        if path is not None:
            tile_paths_map[tile] = path

    # 3. For each station, extract elevation stats
    print(f"\n[*] Step 3: Extracting ~{radius_km:.0f}km elevation window for each station...")
    elev_means = []
    elev_stds = []

    for idx, row in df.iterrows():
        stn_id = row.get("station_id", idx)
        stn_name = row.get("name", f"Station_{stn_id}")
        lat = float(row["lat"])
        lon = float(row["lon"])
        needed_tile_names = station_bboxes[idx]["tiles"]

        # Resolve local paths
        relevant_paths = [tile_paths_map[t] for t in needed_tile_names if t in tile_paths_map]

        if not relevant_paths:
            print(f"  [!] Warning: No DEM tiles found for '{stn_name}' ({lat}, {lon}). Setting to NaN.")
            elev_means.append(np.nan)
            elev_stds.append(np.nan)
            continue

        if len(relevant_paths) > 1:
            print(f"  [*] Station [{stn_id}] '{stn_name}' spans {len(relevant_paths)} tiles: {[p.name for p in relevant_paths]}")
            print(f"      Merging tiles with rasterio.merge before extraction...")

        e_mean, e_std = extract_elevation_for_station(
            lat, lon, relevant_paths, radius_km=radius_km
        )
        elev_means.append(e_mean)
        elev_stds.append(e_std)

    df_result = df.copy()
    df_result["elevation_mean_20km"] = elev_means
    df_result["elevation_std_20km"] = elev_stds
    return df_result


def run_benchmarks(dem_dir: Path, radius_km: float = 20.0):
    """
    Runs elevation extraction on standard benchmark cities across India to verify
    accuracy (Bengaluru ~900m, Chennai/Mumbai near sea level, Delhi ~218m).
    """
    print("\n" + "=" * 70)
    print("              REFERENCE SANITY BENCHMARK VALIDATION")
    print("=" * 70)
    df_bench = pd.DataFrame(BENCHMARK_STATIONS)

    # Determine tiles for benchmarks
    bench_tiles = set()
    bench_bboxes = []
    for idx, row in df_bench.iterrows():
        b = compute_station_bounding_box(row["lat"], row["lon"], radius_km=radius_km)
        bench_bboxes.append(b)
        bench_tiles.update(b["tiles"])

    print(f"[*] Benchmark locations require {len(bench_tiles)} DEM tiles: {sorted(list(bench_tiles))}")
    tile_paths = {}
    for t in sorted(list(bench_tiles)):
        p = download_dem_tile(t, dem_dir)
        if p:
            tile_paths[t] = p

    print("\nResults for Reference Cities:")
    print(f"  {'City / Station':<28} | {'Lat, Lon':<18} | {'Mean Elev':<11} | {'Std Elev':<10} | {'Expected':<22}")
    print("  " + "-" * 95)
    for idx, row in df_bench.iterrows():
        req_tiles = bench_bboxes[idx]["tiles"]
        paths = [tile_paths[t] for t in req_tiles if t in tile_paths]
        e_mean, e_std = extract_elevation_for_station(row["lat"], row["lon"], paths, radius_km=radius_km)
        coords = f"{row['lat']:.4f}, {row['lon']:.4f}"
        print(f"  {row['name']:<28} | {coords:<18} | {e_mean:8.2f} m | {e_std:7.2f} m | {row['expected']:<22}")
    print("=" * 70 + "\n")


def print_station_summary(df: pd.DataFrame):
    """Prints formatted sanity check summary of extracted elevation statistics."""
    print("\n" + "=" * 75)
    print("                STATION ELEVATION SUMMARY & SANITY CHECKS")
    print("=" * 75)
    print(f"Total stations processed: {len(df)}")
    if len(df) > 0 and "elevation_mean_20km" in df.columns:
        print(f"Min  elevation mean   : {df['elevation_mean_20km'].min():.2f} m")
        print(f"Max  elevation mean   : {df['elevation_mean_20km'].max():.2f} m")
        print(f"Avg  elevation mean   : {df['elevation_mean_20km'].mean():.2f} m")

    print("\nStation Elevation Breakdown:")
    for idx, row in df.iterrows():
        stn_id = row.get("station_id", idx)
        stn_name = row.get("name", "Unknown")
        lat = row.get("lat", np.nan)
        lon = row.get("lon", np.nan)
        e_mean = row.get("elevation_mean_20km", np.nan)
        e_std = row.get("elevation_std_20km", np.nan)
        print(f"  Station [{stn_id}] {stn_name}")
        print(f"    Coordinates: ({lat:.5f}, {lon:.5f})")
        print(f"    Elevation 20km: Mean = {e_mean:.2f} m | Std = {e_std:.2f} m")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Copernicus GLO-30 DEM elevation features for monitoring stations."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input station CSV path (default: auto-detect stations_with_coast_dist.csv)",
    )
    parser.add_argument(
        "--output",
        default="stations_with_terrain.csv",
        help="Output CSV path (default: stations_with_terrain.csv)",
    )
    parser.add_argument(
        "--dem-dir",
        default=None,
        help="Directory to save/load downloaded DEM tiles (default: ./data/raw/dem)",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=20.0,
        help="Extraction radius in km around station (default: 20.0)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download existing DEM tiles",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run reference sanity benchmarks (Bengaluru, Chennai, Mumbai, Delhi)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dem_dir = Path(args.dem_dir) if args.dem_dir else base_dir / "data" / "raw" / "dem"

    # 1. Load station file
    station_file = find_station_file(base_dir, args.input)
    print(f"[*] Loading input station data from:\n    {station_file.resolve()}")
    df_stations = pd.read_csv(station_file)

    for col in ["lat", "lon"]:
        if col not in df_stations.columns:
            raise ValueError(
                f"Station CSV {station_file.name} is missing required column '{col}'. Columns found: {list(df_stations.columns)}"
            )

    # 2 & 3 & 4. Process stations (determine tiles, download, merge & extract)
    df_terrain = process_stations(
        df_stations,
        dem_dir=dem_dir,
        radius_km=args.radius_km,
        force_download=args.force_download,
    )

    # 5. Save output CSV
    output_path = base_dir / args.output
    df_terrain.to_csv(output_path, index=False)
    print(f"[+] Saved updated station data with elevation features to:\n    {output_path.resolve()}")

    # Also save as stations_with_all_static_features.csv (population_density dropped from static features)
    all_static_path = base_dir / "stations_with_all_static_features.csv"
    df_terrain.to_csv(all_static_path, index=False)
    print(f"[+] Synced final static features table (without population_density) to:\n    {all_static_path.resolve()}")

    # 6. Sanity check summary
    print_station_summary(df_terrain)

    # Optional benchmark validation
    if args.benchmark:
        run_benchmarks(dem_dir=dem_dir, radius_km=args.radius_km)


if __name__ == "__main__":
    main()
