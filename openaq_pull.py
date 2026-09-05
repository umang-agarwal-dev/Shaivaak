"""
openaq_pull.py

Pulls ground-truth ozone (O3) monitoring station metadata and hourly measurements from OpenAQ v3 API:
1. Queries OpenAQ v3 API for ozone stations in target regional bounding boxes:
   - North India: 72.0E to 81.0E, 18.0N to 29.5N (including Delhi-NCR)
   - South India: 72.0E to 81.0E, 12.0N to 19.5N
2. Filters stations whose operational time window overlaps the 3-month CAMS date range:
   2026-06-07T00:00:00Z to 2026-09-06T23:59:59Z.
3. Identifies the active O3 sensor per station for the 2026 CAMS window.
4. Fetches hourly measurements using datetime_from/datetime_to and date_from/date_to parameters.
5. Saves stations_metadata.csv and o3_measurements.csv.
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import requests
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Regional Bounding boxes
BOUNDING_BOXES = {
    "delhi_ncr": "76.8,28.2,77.6,28.9",
    "north": "72.0,18.0,81.0,29.5",
    "south": "72.0,12.0,81.0,19.5",
}

DEFAULT_CAMS_START = "2026-06-07T00:00:00Z"
DEFAULT_CAMS_END = "2026-09-06T23:59:59Z"


def load_env_file(base_dir: Path):
    """Loads OPENAQ_API_KEY from .env if present."""
    env_path = base_dir / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAQ_API_KEY=") and not os.environ.get("OPENAQ_API_KEY"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            os.environ["OPENAQ_API_KEY"] = key
        except Exception as e:
            print(f"Note: Error reading .env: {e}")


def get_cams_date_range(base_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Detects CAMS start and end timestamps from derived datasets or raw 3-month data."""
    # 1. Check for 3-month derived datasets
    for fname in ["north_3months_derived.nc", "south_3months_derived.nc"]:
        nc_path = base_dir / fname
        if nc_path.exists():
            try:
                import xarray as xr
                with xr.open_dataset(nc_path) as ds:
                    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                    if time_coord in ds:
                        t_min = pd.to_datetime(ds[time_coord].min().values, utc=True)
                        t_max = pd.to_datetime(ds[time_coord].max().values, utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                        return t_min, t_max
            except Exception as e:
                print(f"Note: Could not inspect {fname}: {e}")

    # 2. Check for raw 3-month NetCDF datasets (compute valid_time with 1-day lead)
    for folder in ["north_3months", "south_3months"]:
        mlev_path = base_dir / folder / "data_mlev.nc"
        if mlev_path.exists():
            try:
                import xarray as xr
                with xr.open_dataset(mlev_path) as ds:
                    p_1day = pd.Timedelta(days=1)
                    vt = ds.forecast_reference_time + p_1day
                    t_min = pd.to_datetime(vt.values.min(), utc=True)
                    t_max = pd.to_datetime(vt.values.max(), utc=True) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    return t_min, t_max
            except Exception as e:
                print(f"Note: Could not inspect {folder}/data_mlev.nc: {e}")

    # 3. Check for 2-week derived datasets
    for fname in ["north_2weeks_derived.nc", "south_2weeks_derived.nc"]:
        nc_path = base_dir / fname
        if nc_path.exists():
            try:
                import xarray as xr
                with xr.open_dataset(nc_path) as ds:
                    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                    if time_coord in ds:
                        t_min = pd.to_datetime(ds[time_coord].min().values, utc=True)
                        t_max = pd.to_datetime(ds[time_coord].max().values, utc=True)
                        return t_min, t_max
            except Exception as e:
                print(f"Note: Could not inspect {fname}: {e}")

    return pd.to_datetime(DEFAULT_CAMS_START, utc=True), pd.to_datetime(DEFAULT_CAMS_END, utc=True)


def parse_dt_str(dt_val) -> str | None:
    """Extracts UTC ISO string from datetime object or dict."""
    if isinstance(dt_val, dict):
        return dt_val.get("utc") or dt_val.get("local")
    return str(dt_val) if dt_val is not None else None


def pace_rate_limit(response: requests.Response, min_delay: float = 0.5):
    """Dynamically monitors OpenAQ rate limits and sleeps appropriately."""
    remaining = response.headers.get("X-Ratelimit-Remaining")
    reset_secs = response.headers.get("X-Ratelimit-Reset")

    if remaining is not None:
        try:
            rem = int(remaining)
            if rem < 4:
                wait_time = int(reset_secs) + 1 if reset_secs else 5
                print(f"  [i] Rate limit nearly exhausted ({rem} remaining). Sleeping {wait_time}s until reset...", flush=True)
                time.sleep(wait_time)
                return
        except ValueError:
            pass

    time.sleep(min_delay)


def fetch_locations_for_box(
    region: str, bbox: str, api_key: str, session: requests.Session
) -> list[dict]:
    """
    Queries GET /v3/locations?bbox={bbox}&parameters_id={pid}&limit=1000
    for parameters 3 (O3 mass) and 10 (O3 ppm).
    """
    headers = {"X-API-Key": api_key}
    records = []
    seen_ids = set()

    for pid in [3, 10]:
        url = f"https://api.openaq.org/v3/locations?bbox={bbox}&parameters_id={pid}&limit=1000"
        time.sleep(0.4)

        try:
            response = session.get(url, headers=headers, timeout=30)
            if response.status_code == 429:
                print("  [!] Rate limit reached (429). Sleeping 10s...", flush=True)
                time.sleep(10)
                response = session.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                continue
            data = response.json()
        except Exception as e:
            print(f"Error fetching locations for {region} (param {pid}): {e}")
            continue

        results = data.get("results", [])
        for item in results:
            stn_id = item.get("id")
            if stn_id in seen_ids:
                continue
            seen_ids.add(stn_id)

            coords = item.get("coordinates") or {}
            lat = coords.get("latitude") if isinstance(coords, dict) else item.get("latitude")
            lon = coords.get("longitude") if isinstance(coords, dict) else item.get("longitude")

            if lat is None or lon is None:
                continue

            dt_first = parse_dt_str(item.get("datetimeFirst") or item.get("datetime_first"))
            dt_last = parse_dt_str(item.get("datetimeLast") or item.get("datetime_last"))

            # Determine regional label: north if lat >= 18.0, south if lat < 18.0
            reg_label = "north" if float(lat) >= 18.0 else "south"

            # Check for embedded sensors list
            embedded_sensors = item.get("sensors", [])

            records.append({
                "station_id": stn_id,
                "name": item.get("name"),
                "lat": float(lat),
                "lon": float(lon),
                "region": reg_label,
                "datetime_first": dt_first,
                "datetime_last": dt_last,
                "sensors": embedded_sensors,
            })

    return records


def get_active_o3_sensor(
    location_record: dict,
    cams_start_iso: str,
    cams_end_iso: str,
    api_key: str,
    session: requests.Session
) -> tuple[int | None, str]:
    """
    Finds the active modern O3 sensor for the station that has observations in the CAMS window.
    Uses embedded sensor details if available, or falls back to /v3/locations/{id}/sensors.
    Returns (sensor_id, units).
    """
    station_id = location_record["station_id"]
    sensors = location_record.get("sensors") or []

    if not sensors:
        headers = {"X-API-Key": api_key}
        url = f"https://api.openaq.org/v3/locations/{station_id}/sensors"
        time.sleep(0.3)
        try:
            r = session.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                sensors = r.json().get("results", [])
        except Exception:
            pass

    o3_candidates = []
    for s in sensors:
        param = s.get("parameter")
        param_name = ""
        param_units = None
        p_id = None
        if isinstance(param, dict):
            param_name = str(param.get("name", "")).lower()
            param_units = param.get("units")
            p_id = param.get("id")
        elif isinstance(param, str):
            param_name = param.lower()

        if not p_id:
            p_id = s.get("parameter_id")

        if param_name == "o3" or p_id in [3, 10, 32]:
            o3_candidates.append({
                "sensor_id": s.get("id"),
                "units": param_units or s.get("units") or "µg/m³",
            })

    if not o3_candidates:
        return None, "µg/m³"

    # In OpenAQ v3, the highest sensor ID corresponds to the most modern sensor series (12,000,000+)
    best_candidate = max(o3_candidates, key=lambda c: c["sensor_id"])
    return best_candidate["sensor_id"], best_candidate["units"]


def fetch_hourly_measurements_for_sensor(
    station_id: int | str,
    sensor_id: int | str,
    datetime_from_iso: str,
    datetime_to_iso: str,
    api_key: str,
    session: requests.Session,
    default_units: str = "µg/m³",
) -> list[dict]:
    """
    Calls GET https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly
    with datetime_from and datetime_to in ISO format, handling pagination and rate limits.
    """
    headers = {"X-API-Key": api_key}
    measurements = []
    page = 1
    limit = 1000

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly"
        params = {
            "datetime_from": datetime_from_iso,
            "datetime_to": datetime_to_iso,
            "date_from": datetime_from_iso,
            "date_to": datetime_to_iso,
            "limit": limit,
            "page": page,
        }

        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                print("  [!] Rate limit reached (429). Sleeping 10s and retrying...", flush=True)
                time.sleep(10)
                response = session.get(url, headers=headers, params=params, timeout=30)

            response.raise_for_status()
            data = response.json()
            pace_rate_limit(response, min_delay=0.4)
        except requests.exceptions.RequestException as e:
            print(f"  [!] Error fetching hourly measurements for sensor {sensor_id} (page {page}): {e}")
            break

        results = data.get("results", [])
        meta = data.get("meta", {})

        for item in results:
            dt_utc = None
            period = item.get("period", {})
            if isinstance(period, dict) and "datetimeTo" in period:
                dt_to = period["datetimeTo"]
                dt_utc = dt_to.get("utc") if isinstance(dt_to, dict) else str(dt_to)
            elif isinstance(period, dict) and "datetimeFrom" in period:
                dt_from = period["datetimeFrom"]
                dt_utc = dt_from.get("utc") if isinstance(dt_from, dict) else str(dt_from)
            elif "datetime" in item:
                dt_obj = item["datetime"]
                dt_utc = dt_obj.get("utc") or dt_obj.get("local") if isinstance(dt_obj, dict) else str(dt_obj)

            val = item.get("value")
            units = None
            if isinstance(item.get("parameter"), dict):
                units = item["parameter"].get("units")
            if not units:
                units = item.get("units") or default_units

            if dt_utc is not None and val is not None and val >= 0:
                measurements.append({
                    "station_id": station_id,
                    "datetime_utc": dt_utc,
                    "o3_value": val,
                    "units": units,
                })

        found = meta.get("found")
        total_pages = math.ceil(found / limit) if found is not None and limit > 0 else 1

        if not results or page >= total_pages or len(results) < limit:
            break

        page += 1

    return measurements


def save_progress(
    kept_stations: list[dict],
    all_measurements: list[dict],
    metadata_file: Path,
    measurements_file: Path,
):
    """Saves current stations and measurements incrementally."""
    if kept_stations:
        cols_meta = ["station_id", "name", "lat", "lon", "region", "datetime_first", "datetime_last"]
        df_meta = pd.DataFrame(kept_stations)[cols_meta].drop_duplicates(subset=["station_id"])
        df_meta.to_csv(metadata_file, index=False)

    if all_measurements:
        cols_meas = ["station_id", "datetime_utc", "o3_value", "units"]
        df_meas = pd.DataFrame(all_measurements)[cols_meas].drop_duplicates(subset=["station_id", "datetime_utc"])
        df_meas.to_csv(measurements_file, index=False)


def main():
    base_dir = Path(__file__).resolve().parent
    load_env_file(base_dir)

    parser = argparse.ArgumentParser(description="Pull ground-truth OpenAQ v3 ozone station data.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAQ_API_KEY"),
        help="OpenAQ API key (defaults to OPENAQ_API_KEY environment variable or .env file)",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help=f"Start date (ISO format, defaults to detected CAMS start or {DEFAULT_CAMS_START})",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help=f"End date (ISO format, defaults to detected CAMS end or {DEFAULT_CAMS_END})",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Maximum number of stations to process (default: None, processes all qualifying stations)",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print(
            "ERROR: OpenAQ API key not found. Please set OPENAQ_API_KEY in .env or pass --api-key.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    cams_start_detected, cams_end_detected = get_cams_date_range(base_dir)
    cams_start = pd.to_datetime(args.date_from, utc=True) if args.date_from else cams_start_detected
    cams_end = pd.to_datetime(args.date_to, utc=True) if args.date_to else cams_end_detected

    cams_start_iso = cams_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    cams_end_iso = cams_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 70)
    print("           OPENAQ V3 GROUND-TRUTH OZONE DATA INGESTION")
    print("=" * 70)
    print(f"Target 3-Month Window: {cams_start_iso} to {cams_end_iso}")
    if args.max_stations:
        print(f"Max Stations Limit   : {args.max_stations}")
    else:
        print("Max Stations Limit   : All qualifying stations")

    session = requests.Session()

    # Step 1: Query stations from Delhi-NCR, North, and South bounding boxes
    print("\n[*] Querying OpenAQ locations across regional bounding boxes...")
    all_locations = []

    delhi_locs = fetch_locations_for_box("delhi_ncr", BOUNDING_BOXES["delhi_ncr"], api_key, session)
    print(f"  -> Retrieved {len(delhi_locs)} candidate stations from Delhi-NCR sub-box.")
    all_locations.extend(delhi_locs)

    north_locs = fetch_locations_for_box("north", BOUNDING_BOXES["north"], api_key, session)
    print(f"  -> Retrieved {len(north_locs)} candidate stations from North India box.")
    all_locations.extend(north_locs)

    south_locs = fetch_locations_for_box("south", BOUNDING_BOXES["south"], api_key, session)
    print(f"  -> Retrieved {len(south_locs)} candidate stations from South India box.")
    all_locations.extend(south_locs)

    # Deduplicate locations by station_id
    df_stations = pd.DataFrame(all_locations).drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    print(f"\n[*] Total unique candidate stations found: {len(df_stations)}")

    # Step 2: Filter stations overlapping the 3-month CAMS data window
    df_stations["dt_first_parsed"] = pd.to_datetime(df_stations["datetime_first"], utc=True, errors="coerce")
    df_stations["dt_last_parsed"] = pd.to_datetime(df_stations["datetime_last"], utc=True, errors="coerce")

    overlap_mask = (
        (df_stations["dt_last_parsed"] >= cams_start) &
        (df_stations["dt_first_parsed"] <= cams_end)
    )
    df_filtered = df_stations[overlap_mask].copy().reset_index(drop=True)

    # Filter to coordinates within North or South bounding bounds
    in_north = (df_filtered["lat"] >= 18.0) & (df_filtered["lat"] <= 29.5) & (df_filtered["lon"] >= 72.0) & (df_filtered["lon"] <= 81.0)
    in_south = (df_filtered["lat"] >= 12.0) & (df_filtered["lat"] <= 19.5) & (df_filtered["lon"] >= 72.0) & (df_filtered["lon"] <= 81.0)
    df_filtered = df_filtered[in_north | in_south].copy().reset_index(drop=True)

    # Ensure region label matches geographic latitude
    df_filtered["region"] = ["north" if lat >= 18.0 else "south" for lat in df_filtered["lat"]]

    n_north = (df_filtered["region"] == "north").sum()
    n_south = (df_filtered["region"] == "south").sum()
    print(f"[*] Stations overlapping 3-month window: {len(df_filtered)} (North: {n_north}, South: {n_south})")

    # Prioritize Delhi-NCR stations, then North, then South
    def is_delhi_station(name: str, lat: float, lon: float) -> bool:
        if lat is not None and lon is not None:
            if 28.3 <= lat <= 28.9 and 76.9 <= lon <= 77.5:
                return True
        if isinstance(name, str):
            n_low = name.lower()
            if "delhi" in n_low and "lucknow" not in n_low:
                return True
        return False

    df_filtered["is_delhi"] = [
        is_delhi_station(r.get("name", ""), r.get("lat"), r.get("lon"))
        for _, r in df_filtered.iterrows()
    ]

    # Sort so Delhi stations are at the top, then balanced across regions
    df_filtered = df_filtered.sort_values(
        by=["is_delhi", "region", "station_id"], ascending=[False, False, True]
    ).reset_index(drop=True)

    if args.max_stations and args.max_stations > 0 and len(df_filtered) > args.max_stations:
        print(f"[*] Selecting balanced cross-regional subset of {args.max_stations} stations (requested by --max-stations)...")
        half = args.max_stations // 2
        df_north = df_filtered[df_filtered["region"] == "north"]
        df_south = df_filtered[df_filtered["region"] == "south"]

        n_take_south = min(len(df_south), half)
        n_take_north = min(len(df_north), args.max_stations - n_take_south)
        if (n_take_north + n_take_south) < args.max_stations and len(df_south) > n_take_south:
            n_take_south = min(len(df_south), args.max_stations - n_take_north)

        df_selected = pd.concat([df_north.head(n_take_north), df_south.head(n_take_south)]).reset_index(drop=True)
    else:
        df_selected = df_filtered.copy()

    # Step 3: Discover active O3 sensors & fetch hourly measurements
    print(f"\n[*] Fetching hourly ozone measurements ({cams_start_iso} to {cams_end_iso})...")
    metadata_file = base_dir / "stations_metadata.csv"
    measurements_file = base_dir / "o3_measurements.csv"

    kept_stations = []
    all_measurements = []

    for idx, row in df_selected.iterrows():
        stn_id = row["station_id"]
        name = row["name"]
        lat = row["lat"]
        lon = row["lon"]
        region = row["region"]
        print(f"[{idx + 1}/{len(df_selected)}] Station: '{name}' (ID: {stn_id}, {region}, {lat:.3f}N, {lon:.3f}E)", flush=True)

        sensor_id, units = get_active_o3_sensor(row, cams_start_iso, cams_end_iso, api_key, session)
        if not sensor_id:
            print(f"  [-] No active 2026 O3 sensor found. Skipping.", flush=True)
            continue

        meas = fetch_hourly_measurements_for_sensor(
            station_id=stn_id,
            sensor_id=sensor_id,
            datetime_from_iso=cams_start_iso,
            datetime_to_iso=cams_end_iso,
            api_key=api_key,
            session=session,
            default_units=units,
        )

        if not meas:
            print(f"  [-] Zero measurement records returned for sensor {sensor_id}. Skipping.", flush=True)
            continue

        print(f"  [+] Active sensor: {sensor_id} | Retrieved: {len(meas)} hourly rows (Cumulative: {len(all_measurements) + len(meas):,})", flush=True)
        all_measurements.extend(meas)
        kept_stations.append(row)

        # Autosave every 10 stations
        if len(kept_stations) % 10 == 0:
            save_progress(kept_stations, all_measurements, metadata_file, measurements_file)

    if not kept_stations:
        print("\n[!] No measurements could be retrieved for any station. Exiting.")
        sys.exit(1)

    # Step 4: Final save output files
    save_progress(kept_stations, all_measurements, metadata_file, measurements_file)
    df_kept_stations = pd.DataFrame(kept_stations)
    df_measurements = pd.DataFrame(all_measurements).drop_duplicates(subset=["station_id", "datetime_utc"])

    print(f"\n[+] Final Saved {len(df_kept_stations)} stations to: {metadata_file.name}")
    print(f"[+] Final Saved {len(df_measurements)} hourly ozone measurements to: {measurements_file.name}")

    # Summary
    n_kept_north = (df_kept_stations["region"] == "north").sum()
    n_kept_south = (df_kept_stations["region"] == "south").sum()

    print("\n" + "=" * 70)
    print("                     INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total Stations Qualifying : {len(df_filtered)}")
    print(f"Total Stations Kept       : {len(df_kept_stations)} (North: {n_kept_north}, South: {n_kept_south})")
    print(f"Total Hourly Measurements : {len(df_measurements):,}")
    if not df_measurements.empty:
        print(f"Observation Date Range    : {df_measurements['datetime_utc'].min()} to {df_measurements['datetime_utc'].max()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
