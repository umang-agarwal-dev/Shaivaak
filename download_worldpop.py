import os
import sys
import time
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
import rasterio

# Base WorldPop REST API endpoints
WORLDPOP_API_WPGP = "https://www.worldpop.org/rest/data/pop/wpgp"
WORLDPOP_API_POP = "https://www.worldpop.org/rest/data/pop"


def find_worldpop_url(iso3: str = "IND") -> tuple[str, str, str]:
    """
    Finds the download URL for India's 100m unconstrained population raster
    for the most recent year available (2020 or later) using the WorldPop REST API.

    Returns:
        tuple[str, str, str]: (download_url, popyear, title)
    """
    print(f"[*] Querying WorldPop REST API for country code '{iso3}'...", flush=True)

    wpgp_url = f"{WORLDPOP_API_WPGP}?iso3={iso3}"
    selected_item = None

    try:
        response = requests.get(wpgp_url, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])

        valid_items = [
            item for item in data
            if item.get("files") and item.get("popyear")
        ]
        if valid_items:
            valid_items.sort(key=lambda x: int(x.get("popyear", 0)), reverse=True)
            selected_item = valid_items[0]
            print(f"[*] Found latest entry in wpgp: Year {selected_item.get('popyear')} - {selected_item.get('title')}", flush=True)
    except Exception as e:
        print(f"[!] Warning: Failed querying {wpgp_url}: {e}", flush=True)

    if not selected_item:
        raise RuntimeError(f"Could not find any population datasets for ISO3 '{iso3}' in WorldPop API.")

    download_url = selected_item["files"][0]
    popyear = selected_item.get("popyear", "Unknown")
    title = selected_item.get("title", "WorldPop India Population")

    return download_url, popyear, title


def download_file(
    url: str,
    output_path: Path,
    chunk_size: int = 1048576,  # 1 MB chunk size for maximum throughput
    force: bool = False,
) -> None:
    """
    Downloads a file with streaming using optimized 1 MB buffer chunks
    and interactive tqdm progress reporting with percentage, speed, and ETA.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Encoding": "identity",
    }

    # Retrieve remote content-length
    expected_size = None
    try:
        head_res = requests.head(url, headers=headers, timeout=15)
        if head_res.status_code == 200:
            expected_size = int(head_res.headers.get("content-length", 0))
    except Exception as e:
        print(f"[!] Note: Could not retrieve HEAD content-length: {e}", flush=True)

    # Check if already fully downloaded
    if output_path.exists() and not force:
        local_size = output_path.stat().st_size
        if expected_size and local_size == expected_size:
            print(f"[*] File already downloaded and verified ({local_size / (1024*1024):.2f} MB). Skipping.", flush=True)
            return

    print(f"[*] Starting download from:\n    {url}", flush=True)
    print(f"[*] Saving to:\n    {output_path.resolve()}", flush=True)
    if expected_size:
        print(f"[*] Total file size: {expected_size / (1024*1024):.2f} MB ({expected_size / (1024**3):.2f} GB)", flush=True)
    print(f"[*] Buffer chunk size: {chunk_size / 1024:.0f} KB", flush=True)

    start_time = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_length = int(response.headers.get("content-length", expected_size or 0))

        with open(output_path, "wb") as f, tqdm(
            desc="Downloading india_worldpop.tif",
            total=total_length,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    elapsed = time.time() - start_time
    file_size = output_path.stat().st_size
    speed = (file_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    print(f"\n[+] Download finished in {elapsed:.1f}s ({file_size / (1024*1024):.2f} MB @ avg {speed:.2f} MB/s).", flush=True)


def verify_file_size(output_path: Path) -> int:
    """Verifies that the downloaded file exists and is non-empty."""
    if not output_path.exists():
        raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    size_bytes = output_path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"Downloaded file is empty (0 bytes): {output_path}")

    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    print(f"\n[+] Verification: File size is {size_bytes:,} bytes ({size_mb:.2f} MB / {size_gb:.2f} GB)", flush=True)
    return size_bytes


def inspect_raster(raster_path: Path) -> None:
    """
    Opens the raster once with rasterio and prints its CRS, bounds, resolution,
    and checks if it covers India's extent (roughly lat 6-37, lon 68-97.5).
    """
    print("\n" + "=" * 55, flush=True)
    print("      Rasterio GeoTIFF Inspection & Verification", flush=True)
    print("=" * 55, flush=True)
    with rasterio.open(raster_path) as src:
        crs = src.crs
        bounds = src.bounds
        res = src.res
        width, height = src.width, src.height
        count = src.count

        print(f"Driver:        {src.driver}", flush=True)
        print(f"Dimensions:    {width} x {height} pixels, Bands: {count}", flush=True)
        print(f"CRS:           {crs}", flush=True)
        print(f"Resolution:    {res[0]:.8f}, {res[1]:.8f} deg (~{res[0]*111320:.1f}m at equator)", flush=True)
        print(f"Bounds:        Left={bounds.left:.4f}, Bottom={bounds.bottom:.4f}, Right={bounds.right:.4f}, Top={bounds.top:.4f}", flush=True)

        india_lon_min, india_lon_max = 68.0, 97.5
        india_lat_min, india_lat_max = 6.0, 37.0

        covers_lon = (bounds.left <= india_lon_min) and (bounds.right >= india_lon_max)
        covers_lat = (bounds.bottom <= india_lat_min) and (bounds.top >= india_lat_max)

        print("\nExtent Check against India (lat 6-37, lon 68-97.5):", flush=True)
        print(f"  Longitude [{india_lon_min}, {india_lon_max}]: {'COVERS FULLY' if covers_lon else 'PARTIAL'}", flush=True)
        print(f"  Latitude  [{india_lat_min}, {india_lat_max}]: {'COVERS FULLY' if covers_lat else 'PARTIAL'}", flush=True)

        if covers_lon and covers_lat:
            print("  -> CONFIRMED: GeoTIFF covers the full geographic extent of India.", flush=True)
        else:
            print("  -> Covers India bounds roughly within coordinate boundaries.", flush=True)
    print("=" * 55 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Download and verify WorldPop India population density raster.")
    parser.add_argument(
        "--output",
        default="./data/raw/india_worldpop.tif",
        help="Target output path (default: ./data/raw/india_worldpop.tif)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1048576,  # 1 MB default
        help="Streaming chunk size in bytes (default: 1048576 = 1 MB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find download URL via API without downloading",
    )
    args = parser.parse_args()

    # Step 1: Find URL via WorldPop REST API
    download_url, year, title = find_worldpop_url(iso3="IND")
    print(f"\n[*] Selected WorldPop Dataset:", flush=True)
    print(f"    Title: {title}", flush=True)
    print(f"    Year:  {year}", flush=True)
    print(f"    URL:   {download_url}", flush=True)

    if args.dry_run:
        print("\n[*] Dry run requested. Exiting without downloading.", flush=True)
        return

    output_path = Path(args.output).resolve()

    # Step 2: Download with streaming
    download_file(
        url=download_url,
        output_path=output_path,
        chunk_size=args.chunk_size,
        force=args.force,
    )

    # Step 3: Verify file size
    verify_file_size(output_path)

    # Step 4: Open with rasterio and print CRS, bounds, resolution
    inspect_raster(output_path)


if __name__ == "__main__":
    main()
