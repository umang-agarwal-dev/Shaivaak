import os
import io
import sys
import zipfile
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import shapely

COASTLINE_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
TARGET_CRS = "EPSG:32644"  # WGS 84 / UTM zone 44N (India)


def download_and_extract_coastline(dest_dir: Path, force: bool = False) -> Path:
    """
    Downloads and unzips ne_10m_coastline.zip into dest_dir.
    Returns path to the extracted .shp file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    shp_path = dest_dir / "ne_10m_coastline.shp"

    if shp_path.exists() and not force:
        print(f"[*] Coastline shapefile already exists: {shp_path.name}")
        return shp_path

    print(f"[*] Downloading Natural Earth 10m coastline from:\n    {COASTLINE_URL}")
    response = requests.get(COASTLINE_URL, timeout=60)
    response.raise_for_status()

    print(f"[*] Extracting shapefile components into {dest_dir.resolve()}...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dest_dir)

    if not shp_path.exists():
        raise FileNotFoundError(f"Expected shapefile not found at {shp_path}")

    print(f"[+] Successfully extracted {shp_path.name} ({shp_path.stat().st_size / (1024*1024):.2f} MB)")
    return shp_path


def load_and_prepare_coastline(shp_path: Path) -> shapely.Geometry:
    """
    Loads coastline shapefile with geopandas, filters to South Asia / Indian Ocean
    extent to prevent UTM projection breakdown on global anti-meridians,
    reprojects to EPSG:32644 (UTM 44N), and merges into unified geometry.
    """
    print(f"[*] Loading coastline with GeoPandas from {shp_path.name}...")
    gdf_coast = gpd.read_file(shp_path)
    print(f"[*] Total global coastline features: {len(gdf_coast)} | Source CRS: {gdf_coast.crs}")

    # Clip to broad region surrounding India / Indian Ocean (lon 40-120, lat -15 to 45)
    # This avoids UTM distortion/infinities from points on opposite side of globe
    print(f"[*] Filtering to regional extent around India (lon [40, 120], lat [-15, 45])...")
    gdf_regional = gdf_coast.cx[40.0:120.0, -15.0:45.0].copy()
    print(f"[*] Regional coastline features: {len(gdf_regional)}")

    # Reproject to EPSG:32644 (UTM zone 44N)
    print(f"[*] Reprojecting coastline to {TARGET_CRS} (UTM zone 44N)...")
    gdf_utm = gdf_regional.to_crs(TARGET_CRS)
    gdf_utm = gdf_utm[gdf_utm.is_valid & ~gdf_utm.is_empty]

    # Combine all coastline geometries into a single geometry for fast distance calculation
    print(f"[*] Dissolving coastline geometries into unified spatial index...")
    coastline_geom = shapely.unary_union(gdf_utm.geometry)
    return coastline_geom


def find_station_file(base_dir: Path, custom_path: str | None = None) -> Path:
    """Finds input station CSV file."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified input station file not found: {custom_path}")

    candidates = [
        base_dir / "stations_with_all_static_features.csv",
        base_dir / "stations_with_terrain.csv",
        base_dir / "stations_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(base_dir.glob("*station*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find any stations CSV file (e.g. stations_metadata.csv).")


def run_sanity_benchmark(coastline_geom: shapely.Geometry):
    """Calculates and prints distance to coast for benchmark reference cities."""
    benchmarks = {
        "Mumbai (Coastal)": (18.9220, 72.8340),
        "Chennai (Coastal)": (13.0827, 80.2707),
        "Bengaluru (Inland)": (12.9716, 77.5946),
        "Delhi (Inland)": (28.6139, 77.2090),
    }

    print("\n--- Reference Validation Benchmark ---")
    for city, (lat, lon) in benchmarks.items():
        pt_wgs = gpd.GeoSeries([shapely.Point(lon, lat)], crs="EPSG:4326")
        pt_utm = pt_wgs.to_crs(TARGET_CRS).iloc[0]
        dist_km = pt_utm.distance(coastline_geom) / 1000.0
        print(f"  {city:20s}: {dist_km:6.2f} km")
    print("--------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Compute distance to coastline for monitoring stations.")
    parser.add_argument("--input", default=None, help="Input station CSV (default: auto-detect)")
    parser.add_argument("--output", default="stations_with_coast_dist.csv", help="Output CSV path")
    parser.add_argument("--force-download", action="store_true", help="Force re-download coastline zip")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Step 1: Download & unzip shapefile
    coastline_dir = base_dir / "data" / "raw" / "coastline"
    shp_path = download_and_extract_coastline(coastline_dir, force=args.force_download)

    # Step 2 & 3: Load with geopandas and reproject to EPSG:32644
    coastline_geom = load_and_prepare_coastline(shp_path)

    # Step 4: Load station file
    station_file = find_station_file(base_dir, args.input)
    print(f"\n[*] Loading station data from: {station_file.name}")
    df_stations = pd.read_csv(station_file)

    for col in ["lat", "lon"]:
        if col not in df_stations.columns:
            raise ValueError(f"Station file must contain '{col}' column. Found: {list(df_stations.columns)}")

    # Step 5: Reproject stations to EPSG:32644 and compute distance in km
    print(f"[*] Computing minimum distance to coastline for {len(df_stations)} station(s)...")
    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["lon"], df_stations["lat"]),
        crs="EPSG:4326",
    )
    gdf_stations_utm = gdf_stations.to_crs(TARGET_CRS)

    # Distance in UTM is in meters, divide by 1000 for km
    dist_km = gdf_stations_utm.geometry.distance(coastline_geom) / 1000.0
    df_stations["dist_to_coast_km"] = dist_km.round(2)

    # Step 6: Save output file
    output_path = base_dir / args.output
    df_stations.to_csv(output_path, index=False)
    print(f"[+] Saved updated station data with dist_to_coast_km to:\n    {output_path.resolve()}")

    # Step 7: Print min, max, and mean distance
    print("\n" + "=" * 60)
    print("           COASTAL DISTANCE SUMMARY & SANITY CHECKS")
    print("=" * 60)
    print(f"Total stations processed: {len(df_stations)}")
    print(f"Min  distance to coast : {df_stations['dist_to_coast_km'].min():.2f} km")
    print(f"Max  distance to coast : {df_stations['dist_to_coast_km'].max():.2f} km")
    print(f"Mean distance to coast : {df_stations['dist_to_coast_km'].mean():.2f} km")

    print("\nStations Details:")
    for idx, row in df_stations.iterrows():
        stn_id = row.get("station_id", idx)
        stn_name = row.get("name", "Unknown")
        d_km = row["dist_to_coast_km"]
        print(f"  Station [{stn_id}] {stn_name}: {d_km:.2f} km")

    # Run reference city checks
    run_sanity_benchmark(coastline_geom)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
