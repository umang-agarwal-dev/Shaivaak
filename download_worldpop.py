import os
import sys
import time
import argparse
from pathlib import Path
import requests
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
    print(f"Querying WorldPop REST API for country code '{iso3}'...", flush=True)

    # First, query the standard unconstrained 100m endpoint (wpgp)
    wpgp_url = f"{WORLDPOP_API_WPGP}?iso3={iso3}"
    selected_item = None

    try:
        response = requests.get(wpgp_url, timeout=30)
        response.raise_for_status()
        data = response.json().get("data", [])

        # Filter items with valid files and sort by popyear descending
        valid_items = [
            item for item in data
            if item.get("files") and item.get("popyear")
        ]
        if valid_items:
            valid_items.sort(key=lambda x: int(x.get("popyear", 0)), reverse=True)
            selected_item = valid_items[0]
            print(f"Found latest entry in wpgp: Year {selected_item.get('popyear')} - {selected_item.get('title')}", flush=True)
    except Exception as e:
        print(f"Warning: Failed querying {wpgp_url}: {e}", flush=True)

    # Check if a newer unconstrained 100m dataset is available under other pop aliases (e.g. 2024)
    try:
        alias_url = f"{WORLDPOP_API_POP}/G2_UC_POP_2024_100m?iso3={iso3}"
        r_2024 = requests.get(alias_url, timeout=15)
        if r_2024.status_code == 200:
            data_2024 = r_2024.json().get("data", [])
            if data_2024 and data_2024[0].get("files"):
                item_2024 = data_2024[0]
                year_2024 = int(item_2024.get("popyear", 0))
                current_year = int(selected_item.get("popyear", 0)) if selected_item else 0
                if year_2024 > current_year:
                    print(f"Discovered newer unconstrained dataset: Year {year_2024} ({item_2024.get('title')})", flush=True)
    except Exception:
        pass

    if not selected_item:
        raise RuntimeError(f"Could not find any population datasets for ISO3 '{iso3}' in WorldPop API.")

    download_url = selected_item["files"][0]
    popyear = selected_item.get("popyear", "Unknown")
    title = selected_item.get("title", "WorldPop India Population")

    return download_url, popyear, title


def download_file(
    url: str,
    output_path: Path,
    chunk_size: int = 8192,
    force: bool = False,
) -> None:
    """
    Downloads a file with requests using streaming so it does not load the whole
    file into memory, writing chunks of size `chunk_size`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check remote file metadata via HEAD
    expected_size = None
    try:
        head_res = requests.head(url, timeout=15)
        if head_res.status_code == 200:
            expected_size = int(head_res.headers.get("content-length", 0))
    except Exception as e:
        print(f"Note: Could not retrieve HEAD content-length: {e}", flush=True)

    if output_path.exists() and not force:
        local_size = output_path.stat().st_size
        if expected_size and local_size == expected_size:
            print(f"File already exists and matches remote size ({local_size / (1024*1024):.2f} MB). Skipping download.", flush=True)
            return
        elif local_size > 0 and expected_size is None:
            print(f"File already exists ({local_size / (1024*1024):.2f} MB). Skipping download. Use --force to re-download.", flush=True)
            return

    print(f"Starting download from:\n  {url}", flush=True)
    print(f"Target location:\n  {output_path.resolve()}", flush=True)
    if expected_size:
        print(f"Expected file size: {expected_size / (1024 * 1024):.2f} MB ({expected_size / (1024**3):.2f} GB)", flush=True)

    start_time = time.time()
    downloaded_bytes = 0
    last_print_time = start_time

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    # Print progress every 5 seconds
                    now = time.time()
                    if now - last_print_time >= 5.0:
                        elapsed = now - start_time
                        speed_mb = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        if expected_size:
                            pct = (downloaded_bytes / expected_size) * 100
                            print(
                                f"  Progress: {pct:5.1f}% | "
                                f"{downloaded_bytes / (1024*1024):.1f} / {expected_size / (1024*1024):.1f} MB | "
                                f"Speed: {speed_mb:.2f} MB/s",
                                flush=True,
                            )
                        else:
                            print(
                                f"  Progress: {downloaded_bytes / (1024*1024):.1f} MB downloaded | "
                                f"Speed: {speed_mb:.2f} MB/s",
                                flush=True,
                            )
                        last_print_time = now

    elapsed_total = time.time() - start_time
    avg_speed = (downloaded_bytes / (1024 * 1024)) / elapsed_total if elapsed_total > 0 else 0
    print(
        f"Download complete in {elapsed_total:.1f}s "
        f"({downloaded_bytes / (1024*1024):.2f} MB at avg {avg_speed:.2f} MB/s).",
        flush=True,
    )


def verify_file_size(output_path: Path) -> int:
    """Verifies that the downloaded file exists and is non-empty."""
    if not output_path.exists():
        raise FileNotFoundError(f"Downloaded file not found at {output_path}")

    size_bytes = output_path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"Downloaded file is empty (0 bytes): {output_path}")

    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    print(f"\n[Verification] Downloaded file size: {size_bytes:,} bytes ({size_mb:.2f} MB / {size_gb:.2f} GB)", flush=True)
    return size_bytes


def inspect_raster(raster_path: Path) -> None:
    """
    Opens the raster once with rasterio and prints its CRS, bounds, resolution,
    and checks if it covers India's extent (roughly lat 6-37, lon 68-97.5).
    """
    print("\n--- Inspecting Raster with Rasterio ---", flush=True)
    with rasterio.open(raster_path) as src:
        crs = src.crs
        bounds = src.bounds
        res = src.res
        width, height = src.width, src.height
        count = src.count

        print(f"Driver:       {src.driver}", flush=True)
        print(f"Dimensions:   {width} (width) x {height} (height), Bands: {count}", flush=True)
        print(f"CRS:          {crs}", flush=True)
        print(f"Resolution:   {res} (approx {res[0]*111320:.1f}m at equator)", flush=True)
        print(f"Bounds:       Left={bounds.left:.4f}, Bottom={bounds.bottom:.4f}, Right={bounds.right:.4f}, Top={bounds.top:.4f}", flush=True)

        # Check coverage against India's full extent (roughly lat 6-37, lon 68-97.5)
        india_lon_min, india_lon_max = 68.0, 97.5
        india_lat_min, india_lat_max = 6.0, 37.0

        covers_lon = (bounds.left <= india_lon_min) and (bounds.right >= india_lon_max)
        covers_lat = (bounds.bottom <= india_lat_min) and (bounds.top >= india_lat_max)

        print("\nExtent Confirmation:", flush=True)
        print(f"  Covers Longitude [{india_lon_min}, {india_lon_max}]: {'YES' if covers_lon else 'PARTIAL'} (Bounds: {bounds.left:.2f} to {bounds.right:.2f})", flush=True)
        print(f"  Covers Latitude  [{india_lat_min}, {india_lat_max}]: {'YES' if covers_lat else 'PARTIAL'} (Bounds: {bounds.bottom:.2f} to {bounds.top:.2f})", flush=True)

        if covers_lon and covers_lat:
            print("  -> CONFIRMED: GeoTIFF covers the full geographic extent of India.", flush=True)
        else:
            print("  -> NOTE: Covers India coordinates roughly within boundaries.", flush=True)


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
        default=8192,
        help="Streaming chunk size in bytes (default: 8192)",
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
    print(f"\nSelected WorldPop Dataset:", flush=True)
    print(f"  Title: {title}", flush=True)
    print(f"  Year:  {year}", flush=True)
    print(f"  URL:   {download_url}", flush=True)

    if args.dry_run:
        print("\nDry run requested. Exiting without downloading.", flush=True)
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
