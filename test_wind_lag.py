"""
test_wind_lag.py

Unit tests and validation for add_wind_lag_feature.py:
Tests multi-station network within 100km with known geometry, wind directions,
calculated lags, and verifies that neighbor_o3_lagged computes expected values.
"""

import numpy as np
import pandas as pd
from add_wind_lag_feature import (
    compute_haversine_matrix,
    compute_bearing_matrix,
    compute_angular_difference,
    add_wind_lag_feature,
)


def test_geometry_and_angles():
    # Point 1: Delhi North Campus (28.657, 77.158)
    # Point 2: Delhi South / Okhla (28.530, 77.270) ~18 km SE
    lats = np.array([28.657, 28.530])
    lons = np.array([77.158, 77.270])

    dist = compute_haversine_matrix(lats, lons)
    assert dist.shape == (2, 2)
    assert dist[0, 0] == 0.0
    assert 15.0 < dist[0, 1] < 20.0, f"Unexpected distance: {dist[0, 1]}"

    # Bearing from Point 2 (South-East) to Point 1 (North-West)
    bearing = compute_bearing_matrix(lats, lons)
    # Bearing from index 1 to index 0: row 0, col 1
    b_21 = bearing[0, 1]
    assert 310.0 < b_21 < 330.0, f"Unexpected bearing: {b_21}"

    # Angular difference tests
    assert compute_angular_difference(10.0, 350.0) == 20.0
    assert compute_angular_difference(0.0, 45.0) == 45.0
    assert compute_angular_difference(0.0, 50.0) == 50.0
    assert compute_angular_difference(180.0, 180.0) == 0.0
    print("[+] Geometry and angle tests passed.")


def test_wind_lag_calculation():
    """
    Sets up 3 stations:
    Station A (Target): (28.65, 77.15)
    Station B (Neighbor 1): (28.53, 77.27) ~18 km SE of A. Bearing B -> A is ~321 deg.
    Station C (Neighbor 2): (28.75, 77.05) ~14 km NW of A. Bearing C -> A is ~141 deg.

    Timestamps: 3 consecutive hours: t0, t1, t2.
    Wind at Station A:
    - At t0: wind_dir = 320 deg (from B's direction), wind_speed = 5 m/s (18 km/h).
      Lag from B = round(18 / 18) = 1 hour.
      At t0, Station A looks up B at (t0 - 1h).
    - At t1: wind_dir = 140 deg (from C's direction), wind_speed = 3.89 m/s (14 km/h).
      Lag from C = round(14 / 14) = 1 hour.
      At t1, Station A looks up C at (t1 - 1h).
    - At t2: wind_dir = 50 deg (from NE, neither B nor C qualify).
      At t2, neighbor_o3_lagged should be NaN.
    """
    t_base = pd.Timestamp("2026-08-25 10:00:00", tz="UTC")
    t0 = t_base
    t1 = t_base + pd.Timedelta(hours=1)
    t2 = t_base + pd.Timedelta(hours=2)
    t_minus_1 = t_base - pd.Timedelta(hours=1)

    df_stations = pd.DataFrame([
        {"station_id": 101, "lat": 28.65, "lon": 77.15, "name": "Station A"},
        {"station_id": 102, "lat": 28.53, "lon": 77.27, "name": "Station B"},
        {"station_id": 103, "lat": 28.75, "lon": 77.05, "name": "Station C"},
    ])

    df_base = pd.DataFrame([
        # Station A
        {"station_id": 101, "valid_time": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 5.0, "wind_direction_deg": 320.0, "o3_value": 0.010},
        {"station_id": 101, "valid_time": t1.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 3.8888, "wind_direction_deg": 140.0, "o3_value": 0.012},
        {"station_id": 101, "valid_time": t2.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 4.0, "wind_direction_deg": 50.0, "o3_value": 0.015},

        # Station B (o3_value at t_minus_1 is 0.040, at t0 is 0.045)
        {"station_id": 102, "valid_time": t_minus_1.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 2.0, "wind_direction_deg": 180.0, "o3_value": 0.040},
        {"station_id": 102, "valid_time": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 2.0, "wind_direction_deg": 180.0, "o3_value": 0.045},

        # Station C (o3_value at t0 is 0.030)
        {"station_id": 103, "valid_time": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "wind_speed": 2.0, "wind_direction_deg": 180.0, "o3_value": 0.030},
    ])

    df_result = add_wind_lag_feature(df_base, df_stations, max_dist_km=100.0, angle_threshold_deg=45.0)

    res_a = df_result[df_result["station_id"] == 101].sort_values("valid_time").reset_index(drop=True)

    # Check row 0 (t0): should look up B at t_minus_1 -> 0.040
    val_t0 = res_a.loc[0, "neighbor_o3_lagged"]
    assert np.isclose(val_t0, 0.040, atol=1e-3), f"Expected 0.040 at t0, got {val_t0}"

    # Check row 1 (t1): should look up C at t0 -> 0.030
    val_t1 = res_a.loc[1, "neighbor_o3_lagged"]
    assert np.isclose(val_t1, 0.030, atol=1e-3), f"Expected 0.030 at t1, got {val_t1}"

    # Check row 2 (t2): wind direction 50 deg doesn't match B or C -> should be NaN
    val_t2 = res_a.loc[2, "neighbor_o3_lagged"]
    assert np.isnan(val_t2), f"Expected NaN at t2, got {val_t2}"

    print("[+] Multi-station wind-lag integration test passed successfully!")


if __name__ == "__main__":
    test_geometry_and_angles()
    test_wind_lag_calculation()
    print("\nAll unit tests passed successfully!")
