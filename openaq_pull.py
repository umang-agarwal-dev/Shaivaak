import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

# Bounding boxes for India regions (west, south, east, north)
BOUNDING_BOXES = {
    "north": "72.0,18.0,81.0,29.5",
    "south": "72.0,12.0,81.0,19.5",
}

# Default CAMS 2-week window dates based on valid_time range in CAMS datasets
DEFAULT_CAMS_START = "2026-08-20T00:00:00Z"
DEFAULT_CAMS_END = "2026-09-06T00:00:00Z"


def get_cams_date_range(base_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Detects CAMS start and end timestamps from existing derived datasets,
    falling back to DEFAULT_CAMS_START and DEFAULT_CAMS_END.
    """
    for fname in ["north_2weeks_derived.nc", "south_2weeks_derived.nc"]:
        nc_path = base_dir / fname
        if nc_path.exists():
            try:
                import xarray as xr
                with xr.open_dataset(nc_path) as ds:
                    if "valid_time" in ds.coords:
                        t_min = pd.to_datetime(ds.valid_time.min().values, utc=True)
                        t_max = pd.to_datetime(ds.valid_time.max().values, utc=True)
                        return t_min, t_max
            except Exception as e:
                print(f"Note: Could not inspect {fname}: {e}")

    return pd.to_datetime(DEFAULT_CAMS_START, utc=True), pd.to_datetime(DEFAULT_CAMS_END, utc=True)


def parse_dt_str(dt_val) -> str | None:
    """Extracts UTC ISO string from datetime object or string."""
    if isinstance(dt_val, dict):
        return dt_val.get("utc") or dt_val.get("local")
    return str(dt_val) if dt_val is not None else None


def fetch_locations_for_box(
    region: str, bbox: str, api_key: str, session: requests.Session
) -> list[dict]:
    """
    Queries GET /v3/locations?bbox={bbox}&parameters_id=10&limit=1000
    for a given bounding box.
    """
    url = f"https://api.openaq.org/v3/locations?bbox={bbox}&parameters_id=10&limit=1000"
    headers = {"X-API-Key": api_key}
    time.sleep(1)  # 1-second delay between requests to avoid rate limiting

    print(f"Querying OpenAQ locations for {region} box (bbox={bbox})...")
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching locations for {region} box: {e}")
        return []

    results = data.get("results", [])
    records = []
    for item in results:
        coords = item.get("coordinates") or {}
        lat = coords.get("latitude") if isinstance(coords, dict) else item.get("latitude")
        lon = coords.get("longitude") if isinstance(coords, dict) else item.get("longitude")

        dt_first = parse_dt_str(item.get("datetimeFirst") or item.get("datetime_first"))
        dt_last = parse_dt_str(item.get("datetimeLast") or item.get("datetime_last"))

        records.append({
            "station_id": item.get("id"),
            "name": item.get("name"),
            "lat": lat,
            "lon": lon,
            "region": region,
            "datetime_first": dt_first,
            "datetime_last": dt_last,
        })

    print(f"Retrieved {len(records)} station locations for {region} box.")
    return records


def get_o3_sensor_details(
    station_id: int | str, api_key: str, session: requests.Session
) -> tuple[int | None, str | None]:
    """
    Calls GET https://api.openaq.org/v3/locations/{id} to find the sensor ID
    where parameter.name == 'o3'.
    Returns (sensor_id, units).
    """
    url = f"https://api.openaq.org/v3/locations/{station_id}"
    headers = {"X-API-Key": api_key}
    time.sleep(1)  # 1-second delay

    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Error fetching location detail for station {station_id}: {e}")
        return None, None

    # OpenAQ v3 can return {'results': [location]} or location object directly
    loc = data.get("results", [{}])[0] if isinstance(data.get("results"), list) and data["results"] else data
    sensors = loc.get("sensors", [])

    # If sensors list is empty in the location details, query /locations/{id}/sensors as fallback
    if not sensors:
        time.sleep(1)
        try:
            sensors_url = f"https://api.openaq.org/v3/locations/{station_id}/sensors"
            sensors_res = session.get(sensors_url, headers=headers, timeout=30)
            if sensors_res.status_code == 200:
                sensors = sensors_res.json().get("results", [])
        except Exception:
            pass

    for s in sensors:
        param = s.get("parameter")
        param_name = ""
        param_units = None
        if isinstance(param, dict):
            param_name = str(param.get("name", "")).lower()
            param_units = param.get("units")
        elif isinstance(param, str):
            param_name = param.lower()

        if param_name == "o3" or s.get("parameter_id") == 10:
            return s.get("id"), param_units or s.get("units") or "µg/m³"

    return None, None


def fetch_hourly_measurements_for_sensor(
    station_id: int | str,
    sensor_id: int | str,
    date_from: str,
    date_to: str,
    api_key: str,
    session: requests.Session,
    default_units: str = "µg/m³",
) -> list[dict]:
    """
    Calls GET https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly
    with date_from and date_to, handling pagination via the meta field.
    Returns list of dicts with: station_id, datetime_utc, o3_value, units.
    """
    headers = {"X-API-Key": api_key}
    measurements = []
    page = 1
    limit = 1000

    while True:
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly"
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "page": page,
        }
        time.sleep(1)  # 1-second delay between requests

        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"  [!] Error fetching hourly measurements for sensor {sensor_id} (page {page}): {e}")
            break

        results = data.get("results", [])
        meta = data.get("meta", {})

        for item in results:
            # Datetime extraction
            dt_utc = None
            if "datetime" in item:
                dt_obj = item["datetime"]
                dt_utc = dt_obj.get("utc") or dt_obj.get("local") if isinstance(dt_obj, dict) else dt_obj
            elif "period" in item and isinstance(item["period"], dict):
                period = item["period"]
                dt_to = period.get("datetimeTo")
                dt_from = period.get("datetimeFrom")
                if isinstance(dt_to, dict):
                    dt_utc = dt_to.get("utc")
                elif isinstance(dt_to, str):
                    dt_utc = dt_to
                elif isinstance(dt_from, dict):
                    dt_utc = dt_from.get("utc")
                elif isinstance(dt_from, str):
                    dt_utc = dt_from

            val = item.get("value")
            units = None
            if isinstance(item.get("parameter"), dict):
                units = item["parameter"].get("units")
            if not units:
                units = item.get("units") or default_units

            measurements.append({
                "station_id": station_id,
                "datetime_utc": dt_utc,
                "o3_value": val,
                "units": units,
            })

        # Pagination check based on "meta" field
        total_pages = None
        if "pages" in meta:
            total_pages = meta["pages"]
        elif "totalPages" in meta:
            total_pages = meta["totalPages"]
        elif "found" in meta and meta["found"] is not None and limit > 0:
            total_pages = math.ceil(meta["found"] / limit)

        if not results:
            break
        if total_pages is not None and page >= total_pages:
            break
        if len(results) < limit:
            break

        page += 1

    return measurements


def main():
    parser = argparse.ArgumentParser(description="Pull ground-truth OpenAQ v3 ozone station data.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAQ_API_KEY"),
        help="OpenAQ API key (defaults to OPENAQ_API_KEY environment variable)",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print(
            "ERROR: OpenAQ API key not found. Please set the OPENAQ_API_KEY environment variable "
            "or pass it via --api-key <your_key>.\n"
            "Example in PowerShell: $env:OPENAQ_API_KEY='your_key_here'",
            file=sys.stderr,
        )
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    cams_start, cams_end = get_cams_date_range(base_dir)
    print(f"CAMS data window: {cams_start.isoformat()} to {cams_end.isoformat()}")

    session = requests.Session()

    # Step 1: Query locations for both bounding boxes
    all_locations = []
    for region, bbox in BOUNDING_BOXES.items():
        records = fetch_locations_for_box(region, bbox, api_key, session)
        all_locations.extend(records)

    if not all_locations:
        print("No stations retrieved from OpenAQ API. Exiting.")
        sys.exit(0)

    # Step 2: Parse into DataFrame
    df_stations = pd.DataFrame(all_locations)
    # Deduplicate in case a station falls into both bounding boxes
    df_stations = df_stations.drop_duplicates(subset=["station_id"]).reset_index(drop=True)
    print(f"\nTotal unique stations fetched: {len(df_stations)}")

    # Step 3: Filter stations overlapping the CAMS data window
    # datetime_last is after CAMS_START and datetime_first is before CAMS_END
    df_stations["dt_first_parsed"] = pd.to_datetime(df_stations["datetime_first"], utc=True, errors="coerce")
    df_stations["dt_last_parsed"] = pd.to_datetime(df_stations["datetime_last"], utc=True, errors="coerce")

    overlap_mask = (
        (df_stations["dt_last_parsed"] > cams_start) &
        (df_stations["dt_first_parsed"] < cams_end)
    )
    df_filtered = df_stations[overlap_mask].copy()
    df_filtered = df_filtered.drop(columns=["dt_first_parsed", "dt_last_parsed"]).reset_index(drop=True)

    print(f"Stations overlapping CAMS window kept: {len(df_filtered)}")

    # Step 4 & 5: Find o3 sensor ID and fetch hourly measurements
    all_measurements = []
    date_from_str = cams_start.strftime("%Y-%m-%d")
    date_to_str = cams_end.strftime("%Y-%m-%d")

    print(f"\nFetching hourly ozone measurements for each kept station ({date_from_str} to {date_to_str})...")
    for idx, row in df_filtered.iterrows():
        station_id = row["station_id"]
        station_name = row["name"]
        print(f"[{idx + 1}/{len(df_filtered)}] Processing station: '{station_name}' (ID: {station_id})...")

        sensor_id, units = get_o3_sensor_details(station_id, api_key, session)
        if not sensor_id:
            print(f"  [-] No active O3 sensor found for station '{station_name}'. Skipping.")
            continue

        print(f"  [+] Found O3 sensor ID {sensor_id}. Fetching hourly measurements...")
        station_meas = fetch_hourly_measurements_for_sensor(
            station_id=station_id,
            sensor_id=sensor_id,
            date_from=date_from_str,
            date_to=date_to_str,
            api_key=api_key,
            session=session,
            default_units=units or "µg/m³",
        )
        print(f"  [+] Retrieved {len(station_meas)} hourly measurement records.")
        all_measurements.extend(station_meas)

    # Step 6: Save output files
    metadata_file = base_dir / "stations_metadata.csv"
    measurements_file = base_dir / "o3_measurements.csv"

    # Save stations metadata
    cols_meta = ["station_id", "name", "lat", "lon", "region", "datetime_first", "datetime_last"]
    df_filtered[cols_meta].to_csv(metadata_file, index=False)
    print(f"\nSaved station metadata to: {metadata_file.name}")

    # Save hourly measurements
    df_measurements = pd.DataFrame(all_measurements)
    cols_meas = ["station_id", "datetime_utc", "o3_value", "units"]
    if df_measurements.empty:
        df_measurements = pd.DataFrame(columns=cols_meas)
    else:
        # Reorder columns
        df_measurements = df_measurements[cols_meas]

    df_measurements.to_csv(measurements_file, index=False)
    print(f"Saved O3 hourly measurements to: {measurements_file.name}")

    # Step 7: Print summary
    print("\n" + "=" * 45)
    print(f"Total stations kept: {len(df_filtered)}")
    print(f"Total hourly measurement rows retrieved: {len(df_measurements)}")
    print("=" * 45)


if __name__ == "__main__":
    main()
