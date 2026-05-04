#!/usr/bin/env python3
"""
EV Race Strategy Optimizer: ASC 2024 Full Base Route + Hybrid 67 MPC

Goal:
    Drive the farthest distance in 50 hours using at most 50 kWh.

This version uses ONLY the ASC 2024 full base route GPX file:

    data/asc_24/0_FullBaseRoute.gpx

Main features:
    1. Loads the full base route GPX file.
    2. Computes route distance from latitude/longitude.
    3. Computes grade from GPX <ele> tags when available.
       If elevation is missing, grade defaults to zero and a warning is printed.
    4. Runs:
        - best constant-speed sweep
        - adaptive constant-power baseline
        - grade-aware MPC
    5. Includes --tune-mpc to search MPC weights/parameters and keep the best
       configuration that beats both baselines, when possible.

Examples:
    Run with a local GPX file and skip plot generation for a quick test:
        python ev_race_50h_mpc_hybrid67_updated_coeffs.py --gpx 0_FullBaseRoute.gpx --out-prefix hybrid67_updated --no-plots

    Run with a local GPX file and generate plots:
        python ev_race_50h_mpc_hybrid67_updated_coeffs.py --gpx 0_FullBaseRoute.gpx --out-prefix hybrid67_updated

    Download ASC full base route and run:
        python ev_race_50h_mpc_hybrid67_updated_coeffs.py --download-asc24 --out-prefix hybrid67_updated

Outputs:
    <out-prefix>_summary.csv
    <out-prefix>_constant_speed_sweep.csv
    <out-prefix>_route_profile.csv
    <out-prefix>_mpc_history.csv
    <out-prefix>_distance.png, _energy.png, _speed.png, _power.png, _grade.png, _speed_sweep.png unless --no-plots is set

Dependencies:
    numpy
    pandas
    matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# ASC 2024 full base route only
# -----------------------------

ASC24_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "uw-midsun/strategy_msxvi/main/data/asc_24"
)

ASC24_FULL_BASE_ROUTE_FILENAME = "0_FullBaseRoute.gpx"


def download_asc24_full_base_route(dest_dir: Path, overwrite: bool = False) -> Path:
    """
    Download only the ASC 2024 full base route GPX file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / ASC24_FULL_BASE_ROUTE_FILENAME

    if out_path.exists() and not overwrite:
        print(f"Using existing full base route: {out_path}")
        return out_path

    url = f"{ASC24_BASE_URL}/{ASC24_FULL_BASE_ROUTE_FILENAME}"
    print(f"Downloading ASC full base route: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()

    if len(data) < 1000:
        raise RuntimeError(
            f"Downloaded file looks too small ({len(data)} bytes). "
            "Check the GitHub URL or your internet connection."
        )

    out_path.write_bytes(data)
    print(f"  wrote {out_path} ({len(data):,} bytes)")
    return out_path


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class VehicleParams:
    mass_kg: float = 405.0
    drag_coefficient: float = 0.31
    frontal_area_m2: float = 1.228
    air_density_kg_m3: float = 1.225
    rolling_resistance_coeff: float = 0.010

    drivetrain_efficiency: float = 0.88

    # Physically standard I^2R-style copper-loss exponent for roughly constant voltage.
    motor_copper_loss_coeff_w_per_kw_exp: float = 55.0
    motor_copper_loss_exponent: float = 2.0

    regen_efficiency: float = 0.35
    max_regen_power_w: float = 500.0
    aux_power_w: float = 0.0


@dataclass
class RaceParams:
    total_time_s: float = 50.0 * 3600.0
    initial_energy_kwh: float = 50.0
    dt_s: float = 60.0
    min_speed_mps: float = 0.5
    max_speed_mps: float = 18.0


@dataclass
class MPCParams:
    control_interval_s: float = 60.0
    horizon_s: float = 600.0

    # Objective weights
    final_violation_weight: float = 5000.0
    path_violation_weight: float = 10000.0
    # Penalize being below the ideal energy schedule at the end of each
    # prediction horizon. This is what makes the controller a true receding-
    # horizon MPC instead of a fixed grade-aware policy.
    terminal_pace_weight: float = 1000.0

    leftover_energy_weight: float = 0.005
    smooth_weight: float = 0.0002

    # Initial speed law:
    #     speed = base_speed_mps - grade_gain * grade
    initial_base_speed_mps: float = 12.0
    initial_grade_gain: float = 150.0

    # Candidate search neighborhood
    base_offsets_mps: Tuple[float, ...] = (-0.25, 0.0, 0.25)
    gain_offsets: Tuple[float, ...] = (-15.0, 0.0, 15.0)

    # Always include these anchors in the candidate grid.
    base_anchors_mps: Tuple[float, ...] = (10.5, 11.0, 11.5, 12.0, 12.5)
    gain_anchors: Tuple[float, ...] = (0.0, 80.0, 120.0, 160.0, 200.0)

    max_grade_gain: float = 250.0

    # Power-aware MPC candidates. These wrap the adaptive constant-power rule
    # with a grade-dependent multiplier:
    #     target_power = adaptive_power * power_bias * exp(-power_grade_gain * grade)
    # Negative power_grade_gain spends more power on uphills and less on downhills.
    use_power_aware_candidates: bool = True
    initial_power_bias: float = 1.02
    initial_power_grade_gain: float = -12.0
    power_bias_offsets: Tuple[float, ...] = (0.0,)
    power_grade_gain_offsets: Tuple[float, ...] = (0.0,)
    power_bias_anchors: Tuple[float, ...] = (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06)
    power_grade_gain_anchors: Tuple[float, ...] = (-22.0, -18.0, -16.0, -14.0, -12.0, -10.0)


@dataclass
class SimResult:
    name: str
    time_s: np.ndarray
    distance_m: np.ndarray
    energy_kwh: np.ndarray
    speed_mps: np.ndarray
    power_w: np.ndarray
    grade: np.ndarray
    notes: Dict[str, float]


# -----------------------------
# Route representation/loading
# -----------------------------

class Route:
    def __init__(
        self,
        distance_m: np.ndarray,
        grade: np.ndarray,
        name: str = "route",
        elevation_m: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
    ):
        distance_m = np.asarray(distance_m, dtype=float)
        grade = np.asarray(grade, dtype=float)

        if len(distance_m) != len(grade):
            raise ValueError("distance_m and grade must have same length.")
        if len(distance_m) < 2:
            raise ValueError("Route must contain at least two points.")

        order = np.argsort(distance_m)
        distance_m = distance_m[order]
        grade = grade[order]

        _, unique_idx = np.unique(distance_m, return_index=True)
        self.distance_m = distance_m[unique_idx]
        self.grade = grade[unique_idx]
        self.name = name

        if self.distance_m[0] > 0:
            self.distance_m = np.insert(self.distance_m, 0, 0.0)
            self.grade = np.insert(self.grade, 0, self.grade[0])

        self.elevation_m = None
        if elevation_m is not None:
            elevation_m = np.asarray(elevation_m, dtype=float)[order][unique_idx]
            if len(elevation_m) == len(self.distance_m):
                self.elevation_m = elevation_m

        self.lat = None
        self.lon = None
        if lat is not None and lon is not None:
            lat = np.asarray(lat, dtype=float)[order][unique_idx]
            lon = np.asarray(lon, dtype=float)[order][unique_idx]
            if len(lat) == len(self.distance_m) and len(lon) == len(self.distance_m):
                self.lat = lat
                self.lon = lon

    @property
    def length_m(self) -> float:
        return float(self.distance_m[-1])

    def grade_at(self, position_m: float) -> float:
        if position_m >= self.length_m:
            return float(self.grade[-1])
        return float(np.interp(position_m, self.distance_m, self.grade))

    def to_profile_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "distance_m": self.distance_m,
            "grade": self.grade,
        })
        if self.elevation_m is not None:
            df["elevation_m"] = self.elevation_m
        if self.lat is not None and self.lon is not None:
            df["lat"] = self.lat
            df["lon"] = self.lon
        return df


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    return float(2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def parse_gpx_points(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = ET.parse(path)
    root = tree.getroot()

    lat_values: List[float] = []
    lon_values: List[float] = []
    ele_values: List[float] = []

    for elem in root.iter():
        if not elem.tag.lower().endswith("trkpt"):
            continue

        lat_raw = elem.attrib.get("lat")
        lon_raw = elem.attrib.get("lon")
        if lat_raw is None or lon_raw is None:
            continue

        lat_values.append(float(lat_raw))
        lon_values.append(float(lon_raw))

        ele = np.nan
        for child in elem:
            if child.tag.lower().endswith("ele") and child.text is not None:
                try:
                    ele = float(child.text)
                except ValueError:
                    ele = np.nan
                break
        ele_values.append(ele)

    if len(lat_values) < 2:
        raise ValueError(f"GPX file has fewer than two track points: {path}")

    return (
        np.asarray(lat_values, dtype=float),
        np.asarray(lon_values, dtype=float),
        np.asarray(ele_values, dtype=float),
    )


def load_gpx_route(
    path: Path,
    route_spacing_m: float = 100.0,
    grade_smoothing_m: float = 500.0,
    max_abs_grade: float = 0.15,
) -> Route:
    lat_arr, lon_arr, ele_arr = parse_gpx_points(path)

    segment_distances = [0.0]
    for i in range(1, len(lat_arr)):
        segment_distances.append(haversine_m(lat_arr[i - 1], lon_arr[i - 1], lat_arr[i], lon_arr[i]))
    cumulative_distance = np.cumsum(segment_distances)

    keep = np.r_[True, np.diff(cumulative_distance) > 1e-6]
    cumulative_distance = cumulative_distance[keep]
    lat_arr = lat_arr[keep]
    lon_arr = lon_arr[keep]
    ele_arr = ele_arr[keep]

    total_length = float(cumulative_distance[-1])
    if total_length <= 0:
        raise ValueError("GPX route has zero length.")

    target_distance = np.arange(0.0, total_length + route_spacing_m, route_spacing_m)
    target_distance[-1] = total_length

    lat_resampled = np.interp(target_distance, cumulative_distance, lat_arr)
    lon_resampled = np.interp(target_distance, cumulative_distance, lon_arr)

    has_any_elevation = np.any(np.isfinite(ele_arr))
    if has_any_elevation:
        valid = np.isfinite(ele_arr)
        ele_filled = np.interp(cumulative_distance, cumulative_distance[valid], ele_arr[valid])
        ele_resampled = np.interp(target_distance, cumulative_distance, ele_filled)

        window = max(1, int(round(grade_smoothing_m / max(route_spacing_m, 1e-9))))
        if window > 1:
            kernel = np.ones(window, dtype=float) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded = np.pad(ele_resampled, (pad_left, pad_right), mode="edge")
            ele_smooth = np.convolve(padded, kernel, mode="valid")
        else:
            ele_smooth = ele_resampled

        dd = np.gradient(target_distance)
        de = np.gradient(ele_smooth)
        grade = np.divide(de, dd, out=np.zeros_like(de), where=np.abs(dd) > 1e-9)
        grade = np.clip(grade, -max_abs_grade, max_abs_grade)
        elevation_out = ele_resampled
    else:
        print()
        print("WARNING: No <ele> tags found in the full base route GPX.")
        print("         Distance is real, but grade is set to zero.")
        print("         Terrain-aware MPC will have little/no terrain advantage without elevation.")
        print()
        grade = np.zeros_like(target_distance)
        elevation_out = None

    return Route(
        target_distance,
        grade,
        name=path.stem,
        elevation_m=elevation_out,
        lat=lat_resampled,
        lon=lon_resampled,
    )


def make_synthetic_route(length_m: float = 100_000.0, n: int = 3000) -> Route:
    distance = np.linspace(0.0, length_m, n)
    grade = (
        0.018 * np.sin(2 * np.pi * distance / 9000.0)
        + 0.010 * np.sin(2 * np.pi * distance / 2700.0 + 0.8)
        + 0.006 * np.sin(2 * np.pi * distance / 15000.0 + 2.1)
    )
    grade = np.clip(grade, -0.06, 0.06)
    return Route(distance, grade, name="synthetic_rolling_route")


# -----------------------------
# Physics model
# -----------------------------

def motor_copper_loss_w(mechanical_power_w: float, vehicle: VehicleParams) -> float:
    mechanical_power_kw = max(float(mechanical_power_w), 0.0) / 1000.0
    return float(
        vehicle.motor_copper_loss_coeff_w_per_kw_exp
        * mechanical_power_kw ** vehicle.motor_copper_loss_exponent
    )


def road_load_power_w(speed_mps: float, grade: float, vehicle: VehicleParams) -> float:
    v = max(float(speed_mps), 0.0)
    theta = np.arctan(float(grade))

    f_drag = 0.5 * vehicle.air_density_kg_m3 * vehicle.drag_coefficient * vehicle.frontal_area_m2 * v**2
    f_roll = vehicle.mass_kg * 9.81 * vehicle.rolling_resistance_coeff * np.cos(theta)
    f_grade = vehicle.mass_kg * 9.81 * np.sin(theta)

    p_wheel = (f_drag + f_roll + f_grade) * v

    if p_wheel >= 0:
        base_electrical_power = p_wheel / max(vehicle.drivetrain_efficiency, 1e-6)
        nonlinear_motor_loss = motor_copper_loss_w(p_wheel, vehicle)
        p_batt = base_electrical_power + nonlinear_motor_loss
    else:
        p_batt = p_wheel * vehicle.regen_efficiency
        p_batt = max(p_batt, -vehicle.max_regen_power_w)

    return float(p_batt + vehicle.aux_power_w)


def step_constant_speed(
    position_m: float,
    energy_kwh: float,
    speed_mps: float,
    dt_s: float,
    route: Route,
    vehicle: VehicleParams,
) -> Tuple[float, float, float, float]:
    grade = route.grade_at(position_m)
    power_w = road_load_power_w(speed_mps, grade, vehicle)
    energy_next = energy_kwh - power_w * dt_s / 3.6e6
    position_next = position_m + speed_mps * dt_s
    return position_next, energy_next, power_w, grade


_POWER_SPEED_CACHE: Dict[Tuple[float, float, float, float, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}


def _power_speed_curve(
    grade: float,
    vehicle: VehicleParams,
    min_speed_mps: float,
    max_speed_mps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cached vectorized power-vs-speed curve for rounded grade.

    Rich MPC calls the inverse power model many times. Rounding grade to 0.001
    keeps the terrain dependence while collapsing thousands of nearly identical
    cache entries. The vectorized model is much faster than calling the scalar
    road-load function for each speed grid point.
    """
    rounded_grade = round(float(grade), 4)
    key = (
        rounded_grade,
        float(vehicle.drag_coefficient),
        float(vehicle.rolling_resistance_coeff),
        float(vehicle.motor_copper_loss_coeff_w_per_kw_exp),
        float(vehicle.motor_copper_loss_exponent),
        float(min_speed_mps),
        float(max_speed_mps),
    )
    cached = _POWER_SPEED_CACHE.get(key)
    if cached is not None:
        return cached

    speeds = np.linspace(min_speed_mps, max_speed_mps, 700)
    theta = np.arctan(rounded_grade)
    f_drag = 0.5 * vehicle.air_density_kg_m3 * vehicle.drag_coefficient * vehicle.frontal_area_m2 * speeds**2
    f_roll = vehicle.mass_kg * 9.81 * vehicle.rolling_resistance_coeff * np.cos(theta)
    f_grade = vehicle.mass_kg * 9.81 * np.sin(theta)
    p_wheel = (f_drag + f_roll + f_grade) * speeds

    positive = p_wheel >= 0.0
    powers = np.empty_like(speeds)
    base = p_wheel / max(vehicle.drivetrain_efficiency, 1e-6)
    mech_kw = np.maximum(p_wheel, 0.0) / 1000.0
    loss = vehicle.motor_copper_loss_coeff_w_per_kw_exp * mech_kw ** vehicle.motor_copper_loss_exponent
    powers[positive] = base[positive] + loss[positive]
    regen = p_wheel * vehicle.regen_efficiency
    powers[~positive] = np.maximum(regen[~positive], -vehicle.max_regen_power_w)
    powers = powers + vehicle.aux_power_w

    finite = np.isfinite(powers)
    speeds = speeds[finite]
    powers = powers[finite]
    _POWER_SPEED_CACHE[key] = (speeds, powers)
    return speeds, powers


def speed_for_power(
    target_power_w: float,
    grade: float,
    vehicle: VehicleParams,
    min_speed_mps: float,
    max_speed_mps: float,
) -> float:
    speeds, powers = _power_speed_curve(grade, vehicle, min_speed_mps, max_speed_mps)

    if len(speeds) == 0:
        return float(min_speed_mps)

    if np.max(powers) < target_power_w:
        return float(speeds[-1])

    if np.min(powers) > target_power_w:
        return float(speeds[0])

    feasible_idx = np.where(powers <= target_power_w)[0]
    if len(feasible_idx) > 0:
        idx = int(feasible_idx[-1])
        if idx + 1 < len(speeds):
            p0 = float(powers[idx])
            p1 = float(powers[idx + 1])
            v0 = float(speeds[idx])
            v1 = float(speeds[idx + 1])
            if p1 > p0 + 1e-9:
                alpha = np.clip((target_power_w - p0) / (p1 - p0), 0.0, 1.0)
                return float(v0 + alpha * (v1 - v0))
        return float(speeds[idx])

    idx = int(np.argmin(np.abs(powers - target_power_w)))
    return float(speeds[idx])

# -----------------------------
# Policies and simulation
# -----------------------------

def simulate_policy(
    name: str,
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
    policy: Callable[[float, float, float], float],
) -> SimResult:
    t_values: List[float] = []
    x_values: List[float] = []
    e_values: List[float] = []
    v_values: List[float] = []
    p_values: List[float] = []
    g_values: List[float] = []

    t = 0.0
    x = 0.0
    e = race.initial_energy_kwh

    while t < race.total_time_s and e > 0.0 and x < route.length_m:
        v_cmd = float(policy(t, x, e))
        v_cmd = float(np.clip(v_cmd, race.min_speed_mps, race.max_speed_mps))

        dt = min(race.dt_s, race.total_time_s - t)
        x_next, e_next, p_w, grade = step_constant_speed(x, e, v_cmd, dt, route, vehicle)

        if e_next < 0.0 and p_w > 0:
            dt_empty = e * 3.6e6 / p_w
            dt_empty = np.clip(dt_empty, 0.0, dt)
            x_next = x + v_cmd * dt_empty
            e_next = 0.0
            dt = dt_empty

        t_values.append(t)
        x_values.append(x)
        e_values.append(e)
        v_values.append(v_cmd)
        p_values.append(p_w)
        g_values.append(grade)

        x, e = x_next, e_next
        t += dt

        if dt <= 1e-9:
            break

    t_values.append(t)
    x_values.append(x)
    e_values.append(max(e, 0.0))
    v_values.append(v_values[-1] if v_values else 0.0)
    p_values.append(p_values[-1] if p_values else 0.0)
    g_values.append(route.grade_at(x))

    return SimResult(
        name=name,
        time_s=np.asarray(t_values),
        distance_m=np.asarray(x_values),
        energy_kwh=np.asarray(e_values),
        speed_mps=np.asarray(v_values),
        power_w=np.asarray(p_values),
        grade=np.asarray(g_values),
        notes={
            "final_distance_km": float(x / 1000.0),
            "final_energy_kwh": float(max(e, 0.0)),
            "finish_time_s": float(t),
        },
    )


def constant_speed_policy(speed_mps: float) -> Callable[[float, float, float], float]:
    return lambda t, x, e: speed_mps


def constant_power_policy(
    target_power_w: float,
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
) -> Callable[[float, float, float], float]:
    """
    Adaptive constant-power baseline:
        target power = energy remaining / time remaining
    """
    def policy(t: float, x: float, e: float) -> float:
        time_remaining_s = max(race.total_time_s - t, 1e-9)
        adaptive_power_w = max(0.0, e * 3.6e6 / time_remaining_s)
        grade = route.grade_at(x)
        return speed_for_power(
            adaptive_power_w,
            grade,
            vehicle,
            race.min_speed_mps,
            race.max_speed_mps,
        )

    return policy


class GradeAwareParametricMPC:
    """
    Terrain-aware MPC using direct candidate search.

    Speed law:
        speed = base_speed - grade_gain * grade

    Positive grade slows the car down. Negative grade speeds it up.
    """

    def __init__(self, route: Route, vehicle: VehicleParams, race: RaceParams, mpc: MPCParams):
        self.route = route
        self.vehicle = vehicle
        self.race = race
        self.mpc = mpc
        self.last_params = np.array([mpc.initial_base_speed_mps, mpc.initial_grade_gain], dtype=float)
        self.last_power_params = np.array([mpc.initial_power_bias, mpc.initial_power_grade_gain], dtype=float)
        self.last_command_speed_mps = mpc.initial_base_speed_mps
        self.history: List[Dict[str, float]] = []

    def speed_law(self, base_speed_mps: float, grade_gain: float, grade: float) -> float:
        v = base_speed_mps - grade_gain * grade
        return float(np.clip(v, self.race.min_speed_mps, self.race.max_speed_mps))

    def rollout(self, t0: float, x0: float, e0: float, params: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Predict finite-horizon behavior for a speed-law candidate."""
        base_speed_mps = float(params[0])
        grade_gain = float(params[1])

        t = float(t0)
        x = float(x0)
        e = float(e0)
        min_e = float(e0)
        first_speed: Optional[float] = None
        horizon_end_t = min(self.race.total_time_s, t0 + self.mpc.horizon_s)

        while t < horizon_end_t and x < self.route.length_m:
            dt = min(self.race.dt_s, horizon_end_t - t)
            grade = self.route.grade_at(x)
            v = self.speed_law(base_speed_mps, grade_gain, grade)
            if first_speed is None:
                first_speed = v
            p_w = road_load_power_w(v, grade, self.vehicle)

            e -= p_w * dt / 3.6e6
            min_e = min(min_e, e)
            x += v * dt
            t += dt

        return x - x0, e, min_e, float(first_speed if first_speed is not None else self.race.min_speed_mps), t

    def power_aware_speed(self, bias: float, power_grade_gain: float, t: float, x: float, e: float) -> float:
        """
        Convert an adaptive power target into a speed command.

        adaptive_power = remaining_energy / remaining_time
        target_power = adaptive_power * bias * exp(-power_grade_gain * grade)

        A negative power_grade_gain spends more power on uphills and less on
        downhills. This improved the uploaded full base route because the early
        profile benefits from carrying speed through costly terrain.
        """
        time_remaining_s = max(self.race.total_time_s - t, 1e-9)
        adaptive_power_w = max(0.0, e * 3.6e6 / time_remaining_s)
        grade = self.route.grade_at(x)
        target_power_w = adaptive_power_w * float(bias) * float(np.exp(-power_grade_gain * grade))
        return speed_for_power(
            target_power_w,
            grade,
            self.vehicle,
            self.race.min_speed_mps,
            self.race.max_speed_mps,
        )

    def rollout_power_aware(self, t0: float, x0: float, e0: float, params: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Predict finite-horizon behavior for a power-shaping candidate."""
        bias = float(params[0])
        power_grade_gain = float(params[1])

        t = float(t0)
        x = float(x0)
        e = float(e0)
        min_e = float(e0)
        first_speed: Optional[float] = None
        horizon_end_t = min(self.race.total_time_s, t0 + self.mpc.horizon_s)

        while t < horizon_end_t and x < self.route.length_m:
            dt = min(self.race.dt_s, horizon_end_t - t)
            grade = self.route.grade_at(x)
            v = self.power_aware_speed(bias, power_grade_gain, t, x, e)
            if first_speed is None:
                first_speed = v
            p_w = road_load_power_w(v, grade, self.vehicle)

            e -= p_w * dt / 3.6e6
            min_e = min(min_e, e)
            x += v * dt
            t += dt

        return x - x0, e, min_e, float(first_speed if first_speed is not None else self.race.min_speed_mps), t

    def _pace_energy_target_kwh(self, t: float) -> float:
        """Linear energy schedule used as a terminal horizon constraint."""
        remaining_fraction = max(self.race.total_time_s - t, 0.0) / max(self.race.total_time_s, 1e-9)
        return self.race.initial_energy_kwh * remaining_fraction

    def objective_from_rollout(
        self,
        distance_gain_m: float,
        e_end: float,
        min_e: float,
        current_speed: float,
        horizon_end_t: float,
    ) -> float:
        distance_km = distance_gain_m / 1000.0
        smooth_penalty = (current_speed - self.last_command_speed_mps) ** 2
        final_violation = max(0.0, -e_end)
        path_violation = max(0.0, -min_e)

        # This is the receding-horizon terminal energy constraint. Candidates
        # that spend too much energy inside the finite prediction horizon are
        # penalized before the battery is actually empty.
        required_energy = self._pace_energy_target_kwh(horizon_end_t)
        terminal_pace_deficit = max(0.0, required_energy - e_end)
        leftover_energy = max(0.0, e_end - required_energy)

        return float(
            -distance_km
            + self.mpc.final_violation_weight * final_violation**2
            + self.mpc.path_violation_weight * path_violation**2
            + self.mpc.terminal_pace_weight * terminal_pace_deficit**2
            + self.mpc.leftover_energy_weight * leftover_energy
            + self.mpc.smooth_weight * smooth_penalty
        )

    def objective_power_aware(self, params: np.ndarray, t0: float, x0: float, e0: float) -> float:
        distance_gain_m, e_end, min_e, first_speed, horizon_end_t = self.rollout_power_aware(t0, x0, e0, params)
        return self.objective_from_rollout(distance_gain_m, e_end, min_e, first_speed, horizon_end_t)

    def objective(self, params: np.ndarray, t0: float, x0: float, e0: float) -> float:
        distance_gain_m, e_end, min_e, first_speed, horizon_end_t = self.rollout(t0, x0, e0, params)
        return self.objective_from_rollout(distance_gain_m, e_end, min_e, first_speed, horizon_end_t)

    def _unique_sorted(self, values: Sequence[float], lo: float, hi: float) -> List[float]:
        clipped = [float(np.clip(v, lo, hi)) for v in values]
        return sorted(set(round(v, 6) for v in clipped))

    def command(self, t: float, x: float, e: float) -> float:
        """Choose the next command by finite-horizon candidate optimization.

        At every control update this evaluates a local grid of candidate power
        shaping parameters over ``horizon_s`` seconds, applies a terminal energy
        pacing constraint, and returns only the first speed command. That makes
        this a genuine receding-horizon MPC implementation rather than a fixed
        open-loop policy.
        """
        bias_candidates = self._unique_sorted(
            list(self.mpc.power_bias_anchors)
            + [self.last_power_params[0] + d for d in self.mpc.power_bias_offsets],
            lo=0.70,
            hi=1.30,
        )
        grade_gain_candidates = self._unique_sorted(
            list(self.mpc.power_grade_gain_anchors)
            + [self.last_power_params[1] + d for d in self.mpc.power_grade_gain_offsets],
            lo=-50.0,
            hi=50.0,
        )

        best_score = float("inf")
        best_params = self.last_power_params.copy()
        best_speed = self.power_aware_speed(best_params[0], best_params[1], t, x, e)
        best_pred_e = float("nan")
        best_pred_min_e = float("nan")
        best_pred_distance = float("nan")
        best_kind = "power_aware_mpc"

        for bias in bias_candidates:
            for power_grade_gain in grade_gain_candidates:
                params = np.array([bias, power_grade_gain], dtype=float)
                distance_gain_m, e_end, min_e, first_speed, horizon_end_t = self.rollout_power_aware(t, x, e, params)
                score = self.objective_from_rollout(distance_gain_m, e_end, min_e, first_speed, horizon_end_t)
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_speed = first_speed
                    best_pred_e = e_end
                    best_pred_min_e = min_e
                    best_pred_distance = distance_gain_m
                    best_kind = "power_aware_mpc"

        # Hybrid extension: also evaluate direct speed-law candidates.
        # These can win in sections where a simple speed = base - gain*grade
        # policy has better finite-horizon behavior than power shaping.
        base_candidates = self._unique_sorted(
            list(self.mpc.base_anchors_mps)
            + [self.last_params[0] + d for d in self.mpc.base_offsets_mps],
            lo=self.race.min_speed_mps,
            hi=self.race.max_speed_mps,
        )
        speed_gain_candidates = self._unique_sorted(
            list(self.mpc.gain_anchors)
            + [self.last_params[1] + d for d in self.mpc.gain_offsets],
            lo=0.0,
            hi=self.mpc.max_grade_gain,
        )
        best_speed_params = self.last_params.copy()
        for base_speed in base_candidates:
            for speed_grade_gain in speed_gain_candidates:
                params = np.array([base_speed, speed_grade_gain], dtype=float)
                distance_gain_m, e_end, min_e, first_speed, horizon_end_t = self.rollout(t, x, e, params)
                score = self.objective_from_rollout(distance_gain_m, e_end, min_e, first_speed, horizon_end_t)
                if score < best_score:
                    best_score = score
                    best_speed_params = params
                    best_speed = first_speed
                    best_pred_e = e_end
                    best_pred_min_e = min_e
                    best_pred_distance = distance_gain_m
                    best_kind = "speed_law_mpc"

        self.last_params = best_speed_params

        self.last_power_params = best_params
        self.last_command_speed_mps = best_speed

        grade = self.route.grade_at(x)
        p_w = road_load_power_w(best_speed, grade, self.vehicle)
        self.history.append({
            "time_s": float(t),
            "position_m": float(x),
            "energy_kwh": float(e),
            "candidate_kind": best_kind,
            "power_bias": float(best_params[0]),
            "power_grade_gain": float(best_params[1]),
            "speed_base_mps": float(self.last_params[0]),
            "speed_grade_gain": float(self.last_params[1]),
            "command_speed_mps": float(best_speed),
            "instant_power_w": float(p_w),
            "objective": float(best_score),
            "predicted_horizon_distance_m": float(best_pred_distance),
            "predicted_horizon_end_energy_kwh": float(best_pred_e),
            "predicted_horizon_min_energy_kwh": float(best_pred_min_e),
            "terminal_energy_target_kwh": float(self._pace_energy_target_kwh(min(self.race.total_time_s, t + self.mpc.horizon_s))),
        })

        return best_speed

def grade_aware_mpc_policy(
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
    mpc: MPCParams,
) -> Callable[[float, float, float], float]:
    controller = GradeAwareParametricMPC(route, vehicle, race, mpc)

    last_update_t = -np.inf
    current_command = mpc.initial_base_speed_mps

    def policy(t: float, x: float, e: float) -> float:
        nonlocal last_update_t, current_command

        if t - last_update_t >= mpc.control_interval_s - 1e-9:
            current_command = controller.command(t, x, e)
            last_update_t = t

        return current_command

    policy.controller = controller  # type: ignore[attr-defined]
    return policy


# -----------------------------
# Evaluation and tuning
# -----------------------------

def run_constant_speed_sweep(
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
    speeds_mps: np.ndarray,
) -> Tuple[SimResult, pd.DataFrame]:
    rows = []
    best_result: Optional[SimResult] = None

    for v in speeds_mps:
        result = simulate_policy(
            name=f"Constant speed {v:.1f} m/s",
            route=route,
            vehicle=vehicle,
            race=race,
            policy=constant_speed_policy(float(v)),
        )
        rows.append({
            "speed_mps": float(v),
            "speed_kph": float(v * 3.6),
            "distance_km": result.notes["final_distance_km"],
            "final_energy_kwh": result.notes["final_energy_kwh"],
            "finish_time_s": result.notes["finish_time_s"],
        })

        if best_result is None or result.notes["final_distance_km"] > best_result.notes["final_distance_km"]:
            best_result = result

    assert best_result is not None
    return best_result, pd.DataFrame(rows)


def summarize(results: List[SimResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "strategy": r.name,
            "distance_km": r.notes["final_distance_km"],
            "final_energy_kwh": r.notes["final_energy_kwh"],
            "finish_time_min": r.notes["finish_time_s"] / 60.0,
            "avg_speed_kph": (
                r.notes["final_distance_km"] / max(r.notes["finish_time_s"] / 3600.0, 1e-9)
            ),
        }
        for r in results
    ]).sort_values("distance_km", ascending=False)


def evaluate_strategies(
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
    mpc: MPCParams,
    speed_grid: np.ndarray,
) -> Tuple[List[SimResult], pd.DataFrame]:
    best_speed_result, speed_sweep = run_constant_speed_sweep(route, vehicle, race, speed_grid)
    best_speed_result.name = "Best constant speed"

    constant_power_result = simulate_policy(
        name="Adaptive constant power",
        route=route,
        vehicle=vehicle,
        race=race,
        policy=constant_power_policy(1000.0, route, vehicle, race),
    )

    mpc_policy = grade_aware_mpc_policy(route, vehicle, race, mpc)
    mpc_result = simulate_policy(
        name="Grade-aware MPC",
        route=route,
        vehicle=vehicle,
        race=race,
        policy=mpc_policy,
    )

    # Attach receding-horizon diagnostics so main() can export them without
    # changing the public SimResult fields used elsewhere.
    mpc_result.controller_history = pd.DataFrame(mpc_policy.controller.history)  # type: ignore[attr-defined]

    return [best_speed_result, constant_power_result, mpc_result], speed_sweep


def tune_mpc(
    route: Route,
    vehicle: VehicleParams,
    race: RaceParams,
    speed_grid: np.ndarray,
    preset: str = "quick",
) -> Tuple[MPCParams, pd.DataFrame]:
    """
    Try multiple MPC parameter sets and return the best.

    The score is primarily MPC distance. A small penalty is applied if MPC fails
    to beat the best non-MPC baseline.
    """
    best_speed_result, _ = run_constant_speed_sweep(route, vehicle, race, speed_grid)
    constant_power_result = simulate_policy(
        name="Adaptive constant power",
        route=route,
        vehicle=vehicle,
        race=race,
        policy=constant_power_policy(1000.0, route, vehicle, race),
    )
    baseline_best_km = max(
        best_speed_result.notes["final_distance_km"],
        constant_power_result.notes["final_distance_km"],
    )

    if preset == "quick":
        # Kept intentionally small so it can run on the full GPX route.
        intervals = [10.0]
        base_values = [12.0]
        gain_values = [150.0]
        final_weights = [5000.0]
        path_weights = [10000.0]
        leftover_weights = [0.0]
        smooth_weights = [0.0]
        power_bias_values = [0.98, 1.00, 1.02, 1.04]
        power_grade_values = [-16.0, -14.0, -12.0, -10.0, -8.0]
    elif preset == "wide":
        intervals = [5.0, 10.0, 15.0]
        base_values = [12.0]
        gain_values = [150.0]
        final_weights = [5000.0]
        path_weights = [10000.0]
        leftover_weights = [0.0]
        smooth_weights = [0.0]
        power_bias_values = [0.98, 1.0, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08]
        power_grade_values = [-12.0, -10.0, -8.0, -7.0, -6.5, -6.0, -5.5, -5.0, 0.0]
    else:
        raise ValueError("preset must be 'quick' or 'wide'")

    trials: List[Dict[str, float]] = []
    best_params: Optional[MPCParams] = None
    best_score = -float("inf")

    total = (
        len(intervals) * len(base_values) * len(gain_values)
        * len(final_weights) * len(path_weights)
        * len(leftover_weights) * len(smooth_weights)
        * len(power_bias_values) * len(power_grade_values)
    )
    print(f"Tuning MPC with preset={preset}. Trials: {total}")
    print(f"Best non-MPC baseline: {baseline_best_km:.3f} km")

    trial_idx = 0
    for interval in intervals:
        for base in base_values:
            for gain in gain_values:
                for fw in final_weights:
                    for pw in path_weights:
                        for lw in leftover_weights:
                            for sw in smooth_weights:
                                for power_bias in power_bias_values:
                                    for power_grade_gain in power_grade_values:
                                        trial_idx += 1

                                        params = MPCParams(
                                            control_interval_s=interval,
                                            horizon_s=600.0,
                                            final_violation_weight=fw,
                                            path_violation_weight=pw,
                                            leftover_energy_weight=lw,
                                            smooth_weight=sw,
                                            initial_base_speed_mps=base,
                                            initial_grade_gain=gain,
                                            initial_power_bias=power_bias,
                                            initial_power_grade_gain=power_grade_gain,
                                            power_bias_anchors=(power_bias,),
                                            power_grade_gain_anchors=(power_grade_gain,),
                                            # keep speed-candidate grid small during tuning
                                            base_anchors_mps=(base,),
                                            gain_anchors=(gain,),
                                        )

                                        result = simulate_policy(
                                            name="Grade-aware MPC",
                                            route=route,
                                            vehicle=vehicle,
                                            race=race,
                                            policy=grade_aware_mpc_policy(route, vehicle, race, params),
                                        )

                                        mpc_km = result.notes["final_distance_km"]
                                        margin_km = mpc_km - baseline_best_km

                                        score = mpc_km + 10.0 * min(0.0, margin_km)

                                        row = {
                                            "trial": trial_idx,
                                            "score": score,
                                            "mpc_distance_km": mpc_km,
                                            "margin_vs_best_non_mpc_km": margin_km,
                                            "final_energy_kwh": result.notes["final_energy_kwh"],
                                            "finish_time_s": result.notes["finish_time_s"],
                                            **asdict(params),
                                        }
                                        trials.append(row)

                                        if score > best_score:
                                            best_score = score
                                            best_params = params
                                            print(
                                                f"  new best trial {trial_idx}/{total}: "
                                                f"MPC={mpc_km:.3f} km, margin={margin_km:.3f} km, "
                                                f"interval={interval}, base={base}, gain={gain}, "
                                                f"power_bias={power_bias}, power_grade={power_grade_gain}"
                                            )

    assert best_params is not None
    trials_df = pd.DataFrame(trials).sort_values("score", ascending=False)
    return best_params, trials_df


# -----------------------------
# Plotting
# -----------------------------

def plot_results(results: List[SimResult], speed_sweep: pd.DataFrame, out_prefix: str = "ev_race"):
    plt.figure()
    for r in results:
        plt.plot(r.time_s / 60.0, r.distance_m / 1000.0, label=r.name)
    plt.xlabel("Time [min]")
    plt.ylabel("Distance [km]")
    plt.title("Distance traveled")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_distance.png", dpi=160)

    plt.figure()
    for r in results:
        plt.plot(r.time_s / 60.0, r.energy_kwh, label=r.name)
    plt.xlabel("Time [min]")
    plt.ylabel("Energy remaining [kWh]")
    plt.title("Battery energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_energy.png", dpi=160)

    plt.figure()
    for r in results:
        plt.plot(r.time_s / 60.0, r.speed_mps * 3.6, label=r.name)
    plt.xlabel("Time [min]")
    plt.ylabel("Speed [km/h]")
    plt.title("Commanded speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_speed.png", dpi=160)

    plt.figure()
    for r in results:
        plt.plot(r.time_s / 60.0, r.power_w / 1000.0, label=r.name)
    plt.xlabel("Time [min]")
    plt.ylabel("Battery power [kW]")
    plt.title("Battery power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_power.png", dpi=160)

    longest = max(results, key=lambda r: r.notes["final_distance_km"])
    plt.figure()
    plt.plot(longest.distance_m / 1000.0, longest.grade * 100.0)
    plt.xlabel("Distance [km]")
    plt.ylabel("Grade [%]")
    plt.title("Terrain encountered by best strategy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_grade.png", dpi=160)

    plt.figure()
    plt.plot(speed_sweep["speed_kph"], speed_sweep["distance_km"], marker="o")
    plt.xlabel("Constant speed [km/h]")
    plt.ylabel("Distance achieved [km]")
    plt.title("Constant-speed benchmark sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_speed_sweep.png", dpi=160)


# -----------------------------
# Main
# -----------------------------

def resolve_route_from_args(args: argparse.Namespace) -> Route:
    if args.gpx:
        gpx_path = Path(args.gpx)
    else:
        gpx_dir = Path(args.gpx_dir)
        if args.download_asc24:
            gpx_path = download_asc24_full_base_route(gpx_dir, overwrite=args.overwrite_download)
        else:
            gpx_path = gpx_dir / ASC24_FULL_BASE_ROUTE_FILENAME

    if gpx_path.exists():
        return load_gpx_route(
            gpx_path,
            route_spacing_m=args.gpx_spacing_m,
            grade_smoothing_m=args.gpx_grade_smoothing_m,
        )

    if args.allow_synthetic_fallback:
        print()
        print(f"WARNING: GPX file not found: {gpx_path}")
        print("         Using synthetic route because --allow-synthetic-fallback was set.")
        print()
        return make_synthetic_route()

    raise FileNotFoundError(
        f"Could not find {gpx_path}. Use --download-asc24 or pass --gpx."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpx", type=str, default=None, help="Path to 0_FullBaseRoute.gpx.")
    parser.add_argument("--gpx-dir", type=str, default="data/asc_24", help="Directory containing 0_FullBaseRoute.gpx.")
    parser.add_argument("--download-asc24", action="store_true", help="Download 0_FullBaseRoute.gpx from GitHub.")
    parser.add_argument("--overwrite-download", action="store_true", help="Redownload 0_FullBaseRoute.gpx if it exists.")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", help="Use synthetic route if GPX is missing.")

    parser.add_argument("--gpx-spacing-m", type=float, default=100.0, help="Route spacing for GPX resampling.")
    parser.add_argument("--gpx-grade-smoothing-m", type=float, default=500.0, help="Elevation smoothing length before grade calculation.")

    parser.add_argument("--tune-mpc", action="store_true", help="Tune MPC parameters on the loaded full base route.")
    parser.add_argument("--tune-preset", choices=["quick", "wide"], default="quick", help="MPC tuning search size.")
    parser.add_argument("--out-prefix", type=str, default="hybrid67_updated", help="Output file prefix.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    vehicle = VehicleParams()
    race = RaceParams()
    route = resolve_route_from_args(args)

    print("Route source: {}".format(route.name))
    print("Route length: {:.1f} km".format(route.length_m / 1000.0))
    print("Route points: {:,}".format(len(route.distance_m)))
    print("Grade range: {:.2f}% to {:.2f}%".format(np.min(route.grade) * 100.0, np.max(route.grade) * 100.0))
    print("Energy budget: {:.3f} kWh over {:.1f} min".format(
        race.initial_energy_kwh,
        race.total_time_s / 60.0,
    ))
    print()

    speed_grid = np.linspace(3.0, 18.0, 31)

    if args.tune_mpc:
        best_mpc, tuning_df = tune_mpc(
            route=route,
            vehicle=vehicle,
            race=race,
            speed_grid=speed_grid,
            preset=args.tune_preset,
        )
        mpc_params = best_mpc
        tuning_path = f"{args.out_prefix}_mpc_tuning_trials.csv"
        tuning_df.to_csv(tuning_path, index=False)

        best_path = f"{args.out_prefix}_best_mpc_params.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(asdict(best_mpc), f, indent=2)

        print()
        print("Best MPC parameters:")
        print(json.dumps(asdict(best_mpc), indent=2))
        print()
        print(f"Wrote tuning trials: {tuning_path}")
        print(f"Wrote best params:   {best_path}")
        print()
    else:
        mpc_params = MPCParams()

    results, speed_sweep = evaluate_strategies(route, vehicle, race, mpc_params, speed_grid)
    summary = summarize(results)

    print("Strategy comparison:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    best_non_mpc = max(
        r.notes["final_distance_km"]
        for r in results
        if r.name != "Grade-aware MPC"
    )
    mpc_distance = next(r.notes["final_distance_km"] for r in results if r.name == "Grade-aware MPC")
    print(f"MPC margin vs best non-MPC method: {mpc_distance - best_non_mpc:.3f} km")
    print()

    speed_sweep_path = f"{args.out_prefix}_constant_speed_sweep.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    route_profile_path = f"{args.out_prefix}_route_profile.csv"
    mpc_history_path = f"{args.out_prefix}_mpc_history.csv"

    speed_sweep.to_csv(speed_sweep_path, index=False)
    summary.to_csv(summary_path, index=False)
    route.to_profile_dataframe().to_csv(route_profile_path, index=False)

    mpc_result_for_history = next((r for r in results if r.name == "Grade-aware MPC"), None)
    if mpc_result_for_history is not None and hasattr(mpc_result_for_history, "controller_history"):
        mpc_result_for_history.controller_history.to_csv(mpc_history_path, index=False)  # type: ignore[attr-defined]

    if not args.no_plots:
        plot_results(results, speed_sweep, args.out_prefix)

    print("Wrote:")
    print(f"  {summary_path}")
    print(f"  {speed_sweep_path}")
    print(f"  {route_profile_path}")
    if mpc_result_for_history is not None and hasattr(mpc_result_for_history, "controller_history"):
        print(f"  {mpc_history_path}")
    if not args.no_plots:
        print(f"  {args.out_prefix}_distance.png")
        print(f"  {args.out_prefix}_energy.png")
        print(f"  {args.out_prefix}_speed.png")
        print(f"  {args.out_prefix}_power.png")
        print(f"  {args.out_prefix}_grade.png")
        print(f"  {args.out_prefix}_speed_sweep.png")


if __name__ == "__main__":
    main()
