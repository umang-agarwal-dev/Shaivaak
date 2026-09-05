"""
add_wind_lag_feature.py

Computes wind-directed lagged ozone transport feature (neighbor_o3_lagged):
1. Finds all other stations B within 100km of station A (Haversine distance).
2. Checks whether wind_direction_deg at station A points FROM station B's direction
   (bearing from B to A is within +-45 degrees of wind_direction_deg).
3. For qualifying neighbor stations, computes lag_hours = distance_km / (wind_speed_m_s * 3.6),
   rounded to the nearest whole hour.
4. Looks up station B's o3_value at (t - lag_hours) from base_training_table.csv.
5. If multiple qualifying neighbors exist, averages their lagged o3_value.
6. Adds column neighbor_o3_lagged (NaN if no qualifying neighbor found).
7. Saves the result as final_training_table.csv.
8. Prints summary statistics of non-null neighbor_o3_lagged rows.

Uses numpy vectorized operations over station pairs and time series arrays.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def compute_haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Computes NxN pairwise great-circle distance matrix in km using Haversine formula.
    """
    earth_radius_km = 6371.0
    phi = np.radians(lats)
    lam = np.radians(lons)

    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi[:, None]) * np.cos(phi[None, :]) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c


def compute_bearing_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Computes NxN pairwise forward azimuth / initial bearing matrix in degrees from B to A.
    Matrix entry [i, j] represents the bearing from station j (B) to station i (A).
    Bearing range is [0, 360).
    """
    phi = np.radians(lats)
    lam = np.radians(lons)

    # Station A is index i (row), Station B is index j (column)
    phi_a = phi[:, None]
    lam_a = lam[:, None]
    phi_b = phi[None, :]
    lam_b = lam[None, :]

    dlam = lam_a - lam_b
    y = np.sin(dlam) * np.cos(phi_a)
    x = np.cos(phi_b) * np.sin(phi_a) - np.sin(phi_b) * np.cos(phi_a) * np.cos(dlam)

    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return bearing


def compute_angular_difference(angle1: float | np.ndarray, angle2: float | np.ndarray) -> np.ndarray:
    """
    Computes minimum circular difference between two angles in degrees (range: [0, 180]).
    """
    return np.abs((angle1 - angle2 + 180.0) % 360.0 - 180.0)


def add_wind_lag_feature(
    df_base: pd.DataFrame,
    df_stations: pd.DataFrame,
    max_dist_km: float = 100.0,
    angle_threshold_deg: float = 45.0,
    df_o3: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculates neighbor_o3_lagged for each row (station A, timestamp t)
    using vectorized station pair filtering and time-series array indexing.
    """
    df = df_base.copy()

    # Ensure timestamp is parsed as UTC datetime
    df["_valid_time_dt"] = pd.to_datetime(df["valid_time"], format="ISO8601", utc=True)

    # Get unique stations present in the base table
    unique_stn_ids = df["station_id"].unique()
    n_stations = len(unique_stn_ids)

    print(f"[*] Base table has {len(df)} rows across {n_stations} unique station(s).")

    # Match coordinates for unique stations from df_stations
    stn_coord_map = {}
    for _, row in df_stations.iterrows():
        sid = row["station_id"]
        if "lat" in row and "lon" in row and pd.notna(row["lat"]) and pd.notna(row["lon"]):
            stn_coord_map[sid] = (float(row["lat"]), float(row["lon"]))

    missing_coords = [sid for sid in unique_stn_ids if sid not in stn_coord_map]
    if missing_coords:
        raise ValueError(f"Missing coordinates for station IDs: {missing_coords} in stations metadata.")

    stn_id_list = list(unique_stn_ids)
    stn_to_idx = {sid: idx for idx, sid in enumerate(stn_id_list)}
    lats = np.array([stn_coord_map[sid][0] for sid in stn_id_list])
    lons = np.array([stn_coord_map[sid][1] for sid in stn_id_list])

    # Compute pairwise distance matrix and bearing matrix
    dist_matrix = compute_haversine_matrix(lats, lons)
    bearing_matrix = compute_bearing_matrix(lats, lons)

    # Build neighbor candidate mapping: for each station A, list of (neighbor B, dist_km, bearing_B_to_A)
    # Exclude self (A == B) and distances > max_dist_km
    neighbor_candidates = {sid: [] for sid in stn_id_list}
    total_neighbor_pairs = 0

    for i, stn_a in enumerate(stn_id_list):
        for j, stn_b in enumerate(stn_id_list):
            if i == j:
                continue
            dist_ij = dist_matrix[i, j]
            if dist_ij <= max_dist_km:
                bearing_ba = bearing_matrix[i, j]
                neighbor_candidates[stn_a].append({
                    "neighbor_id": stn_b,
                    "dist_km": dist_ij,
                    "bearing_b_to_a": bearing_ba,
                })
                total_neighbor_pairs += 1

    print(f"[*] Found {total_neighbor_pairs} directional neighbor pair(s) within {max_dist_km} km.")

    # Create fast O(1) lookup dictionary: (station_id, timestamp_utc) -> o3_value
    o3_lookup = {}
    for sid, t_val, o3 in zip(df["station_id"], df["_valid_time_dt"], df["o3_value"]):
        if pd.notna(o3):
            o3_lookup[(sid, t_val)] = float(o3)

    # Optional nearest lookup index from high-resolution o3_measurements.csv
    o3_by_station = {}
    if df_o3 is not None and not df_o3.empty and "datetime_utc" in df_o3.columns and "o3_value" in df_o3.columns:
        df_o3_clean = df_o3.copy()
        df_o3_clean["_dt"] = pd.to_datetime(df_o3_clean["datetime_utc"], format="ISO8601", utc=True).dt.tz_convert(None)
        df_o3_clean = df_o3_clean.dropna(subset=["_dt", "o3_value", "station_id"])
        for sid, grp in df_o3_clean.groupby("station_id"):
            sorted_grp = grp.sort_values("_dt").drop_duplicates("_dt")
            o3_by_station[sid] = (sorted_grp["_dt"].values, sorted_grp["o3_value"].values.astype(float))

    # Pre-allocate array for neighbor_o3_lagged
    neighbor_o3_lagged = np.full(len(df), np.nan, dtype=np.float64)

    # If no candidate pairs exist (e.g. single station or isolated stations)
    if total_neighbor_pairs == 0:
        print(f"  [!] Note: No neighbor stations within {max_dist_km} km. All neighbor_o3_lagged will be NaN.")
        df["neighbor_o3_lagged"] = neighbor_o3_lagged
        df = df.drop(columns=["_valid_time_dt"])
        return df

    # Group row indices by station_id
    grouped_indices = df.groupby("station_id").groups

    # Process each station A
    for stn_a, row_indices in grouped_indices.items():
        candidates = neighbor_candidates.get(stn_a, [])
        if not candidates:
            continue

        n_rows_a = len(row_indices)
        idx_array = row_indices.values

        # Extract station A's time series arrays
        times_a = df["_valid_time_dt"].iloc[idx_array].values
        wind_speeds = df["wind_speed"].iloc[idx_array].values
        wind_dirs = df["wind_direction_deg"].iloc[idx_array].values

        # Accumulator for valid neighbor readings per row of station A: list of lists
        accum_o3_vals = [[] for _ in range(n_rows_a)]

        for cand in candidates:
            stn_b = cand["neighbor_id"]
            dist_km = cand["dist_km"]
            bearing_ba = cand["bearing_b_to_a"]

            # Vectorized angular difference between wind direction and bearing B -> A
            diff = compute_angular_difference(bearing_ba, wind_dirs)

            # Qualifying condition: within +-45 deg of wind direction and positive wind speed
            qualifies = (diff <= angle_threshold_deg) & (wind_speeds > 0)
            qual_pos = np.where(qualifies)[0]

            if len(qual_pos) == 0:
                continue

            # Compute lag_hours = round(dist_km / (wind_speed * 3.6))
            ws_qual = wind_speeds[qual_pos]
            lag_hours = np.round(dist_km / (ws_qual * 3.6)).astype(int)

            # Compute lagged target timestamp: t - lag_hours
            target_times = pd.to_datetime(times_a[qual_pos] - pd.to_timedelta(lag_hours, unit="h"), utc=True)

            # Lookup station B's o3_value at (t - lag_hours)
            for k, pos_in_a in enumerate(qual_pos):
                t_target = target_times[k]
                val = o3_lookup.get((stn_b, t_target))
                if (val is None or np.isnan(val)) and stn_b in o3_by_station:
                    times_b, vals_b = o3_by_station[stn_b]
                    target_dt64 = np.datetime64(t_target.tz_convert(None))
                    pos = np.searchsorted(times_b, target_dt64)
                    best_diff = 3600.0  # within 1 hour tolerance
                    best_v = None
                    for c_pos in (pos - 1, pos):
                        if 0 <= c_pos < len(times_b):
                            diff_sec = abs((times_b[c_pos] - target_dt64) / np.timedelta64(1, "s"))
                            if diff_sec <= best_diff:
                                best_diff = diff_sec
                                best_v = vals_b[c_pos]
                    val = best_v

                if val is not None and not np.isnan(val):
                    accum_o3_vals[pos_in_a].append(val)

        # Compute average of qualifying neighbor lagged values for each row of station A
        for pos_in_a, vals in enumerate(accum_o3_vals):
            if len(vals) > 0:
                original_row_idx = idx_array[pos_in_a]
                neighbor_o3_lagged[original_row_idx] = float(np.mean(vals))

    df["neighbor_o3_lagged"] = neighbor_o3_lagged
    df = df.drop(columns=["_valid_time_dt"])
    return df


def print_summary(df: pd.DataFrame):
    """Prints statistics and sanity check on neighbor_o3_lagged column."""
    total_rows = len(df)
    non_null_count = int(df["neighbor_o3_lagged"].notnull().sum())
    null_count = total_rows - non_null_count
    pct_matched = (non_null_count / total_rows * 100.0) if total_rows > 0 else 0.0

    print("\n" + "=" * 75)
    print("               WIND-LAGGED OZONE FEATURE SUMMARY")
    print("=" * 75)
    print(f"Total Rows in Final Table       : {total_rows}")
    print(f"Non-Null neighbor_o3_lagged Rows: {non_null_count} ({pct_matched:.2f}%)")
    print(f"Null / Unmatched Rows           : {null_count} ({100.0 - pct_matched:.2f}%)")

    if non_null_count > 0:
        val_series = df["neighbor_o3_lagged"].dropna()
        print(f"Min   neighbor_o3_lagged        : {val_series.min():.6f}")
        print(f"Max   neighbor_o3_lagged        : {val_series.max():.6f}")
        print(f"Mean  neighbor_o3_lagged        : {val_series.mean():.6f}")
        print(f"Std   neighbor_o3_lagged        : {val_series.std():.6f}")
    else:
        print("Note: 0 non-null values because no neighbor stations were within 100km,")
        print("      or upwind neighbors did not have ground-truth o3 readings at the lagged times.")

    print("\nSample Rows:")
    sample_cols = [
        "station_id",
        "valid_time",
        "wind_speed",
        "wind_direction_deg",
        "o3_value",
        "neighbor_o3_lagged",
    ]
    avail_cols = [c for c in sample_cols if c in df.columns]
    print(df[avail_cols].head(5).to_string(index=False))
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute wind-lagged ozone transport feature from neighboring stations."
    )
    parser.add_argument(
        "--input",
        default="base_training_table.csv",
        help="Input base training table CSV (default: base_training_table.csv)",
    )
    parser.add_argument(
        "--stations",
        default="stations_with_all_static_features.csv",
        help="Input stations metadata CSV (default: stations_with_all_static_features.csv)",
    )
    parser.add_argument(
        "--output",
        default="final_training_table.csv",
        help="Output CSV path (default: final_training_table.csv)",
    )
    parser.add_argument(
        "--max-dist-km",
        type=float,
        default=100.0,
        help="Maximum neighbor radius in km (default: 100.0)",
    )
    parser.add_argument(
        "--angle-threshold-deg",
        type=float,
        default=45.0,
        help="Bearing difference threshold in degrees (default: 45.0)",
    )
    parser.add_argument(
        "--o3",
        default="o3_measurements.csv",
        help="Optional path to ground-truth o3_measurements.csv for hourly lookups (default: o3_measurements.csv)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    stations_path = base_dir / args.stations
    output_path = base_dir / args.output
    o3_path = base_dir / args.o3

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")
    if not stations_path.exists():
        raise FileNotFoundError(f"Stations file not found: {stations_path.resolve()}")

    print(f"[*] Loading base training table from: {input_path.name}")
    df_base = pd.read_csv(input_path)

    print(f"[*] Loading station metadata from: {stations_path.name}")
    df_stations = pd.read_csv(stations_path)

    df_o3 = None
    if o3_path.exists():
        print(f"[*] Loading high-resolution ground truth from: {o3_path.name}")
        df_o3 = pd.read_csv(o3_path)

    df_final = add_wind_lag_feature(
        df_base=df_base,
        df_stations=df_stations,
        max_dist_km=args.max_dist_km,
        angle_threshold_deg=args.angle_threshold_deg,
        df_o3=df_o3,
    )

    df_final.to_csv(output_path, index=False)
    print(f"\n[+] Saved final training table with neighbor_o3_lagged to:\n    {output_path.resolve()}")

    print_summary(df_final)


if __name__ == "__main__":
    main()
