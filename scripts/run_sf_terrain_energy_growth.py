from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageDraw


SF_BBOX = {
    "west": -122.515,
    "east": -122.355,
    "south": 37.705,
    "north": 37.812,
}

ROLE_COLORS = {
    "empty": (226, 232, 240),
    "residential": (105, 180, 166),
    "commercial": (214, 158, 80),
}


@dataclass
class Cell:
    row: int
    col: int
    lon: float
    lat: float
    building_count: int
    land: bool
    elevation_m: float = 0.0
    slope: float = 0.0
    aspect_southness: float = 0.0
    solar_factor: float = 1.0
    productivity: float = 1.0
    money: float = 10_000.0
    occupied: bool = False
    use_type: str = "residential"
    floors: int = 1
    pv_kw: float = 0.0
    battery_kwh: float = 0.0
    battery_soc: float = 0.0
    daily_demand: float = 0.0
    daily_served: float = 0.0
    daily_surplus: float = 0.0
    lifetime_demand: float = 0.0
    lifetime_served: float = 0.0
    lifetime_generated: float = 0.0
    lifetime_imported: float = 0.0
    lifetime_exported: float = 0.0
    lifetime_transit: float = 0.0
    lifetime_transit_fee: float = 0.0
    cables: set[tuple[int, int]] = field(default_factory=set)
    role: str = "residential"

    @property
    def service_ratio(self) -> float:
        if self.lifetime_demand <= 0:
            return 1.0
        return self.lifetime_served / self.lifetime_demand

    @property
    def cable_degree(self) -> int:
        return len(self.cables)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return project_root().parents[0] / "data"


def request_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "CSSS2026-SF-energy-growth/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_two_profiles(profile_csv: Path, seed: int, steps: int) -> tuple[list[float], list[float], dict[str, str]]:
    def resize(values: list[float]) -> list[float]:
        if len(values) >= steps:
            return values[:steps]
        return [values[i % len(values)] for i in range(steps)]

    def calibrate(values: list[float], target_annual_kwh: float) -> list[float]:
        mean = sum(values) / max(1, len(values))
        target_mean = target_annual_kwh / 8760.0
        if mean <= 0:
            return values
        scale = target_mean / mean
        return [value * scale for value in values]

    def synthetic_values(kind: str) -> list[float]:
        values = []
        for i in range(steps):
            hour = i % 24
            weekday = (i // 24) % 7 < 5
            morning = math.exp(-((hour - 7.5) / 2.2) ** 2)
            evening = math.exp(-((hour - 19.0) / 3.0) ** 2)
            workday = (1.0 / (1.0 + math.exp(-(hour - 8.0)))) * (1.0 / (1.0 + math.exp(hour - 18.0)))
            if kind == "residential":
                values.append(0.22 + 0.26 * morning + 0.54 * evening + (0.06 if not weekday else 0.0))
            else:
                values.append(0.18 + (0.95 if weekday else 0.42) * workday)
        return values

    if not profile_csv.exists():
        res_out = calibrate(synthetic_values("residential"), 6_200.0)
        com_out = calibrate(synthetic_values("commercial"), 82_000.0)
        return res_out, com_out, {
            "residential_profile_id": "synthetic_residential_daily_shape",
            "commercial_profile_id": "synthetic_commercial_workday_shape",
            "residential_profile_calibration": "synthetic daily shape, scaled to 6,200 kWh/year",
            "commercial_profile_calibration": "synthetic workday shape, scaled to 82,000 kWh/year",
            "profile_source": f"synthetic fallback; missing {profile_csv}",
        }

    rng = random.Random(seed)
    chosen: dict[str, tuple[str, list[float]]] = {}
    seen = {"residential": 0, "commercial": 0}
    with profile_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        time_cols = [name for name in reader.fieldnames or [] if name.startswith("t")]
        for row in reader:
            kind = (row.get("profile_type") or "").lower()
            if kind not in seen:
                continue
            seen[kind] += 1
            if kind not in chosen or rng.random() < 1.0 / seen[kind]:
                values = [max(0.0, safe_float(row.get(col), 0.0)) for col in time_cols[:steps]]
                if values:
                    chosen[kind] = (row.get("profile_id", kind), values)
    if "residential" not in chosen or "commercial" not in chosen:
        raise RuntimeError(f"Could not sample both residential and commercial profiles from {profile_csv}")

    res_id, res = chosen["residential"]
    com_id, com = chosen["commercial"]
    res_out = calibrate(resize(res), 6_200.0)
    com_out = calibrate(resize(com), 82_000.0)
    return res_out, com_out, {
        "residential_profile_id": res_id,
        "commercial_profile_id": com_id,
        "residential_profile_calibration": "shape sampled from data, scaled to 6,200 kWh/year",
        "commercial_profile_calibration": "shape sampled from data, scaled to 82,000 kWh/year",
        "profile_source": str(profile_csv),
    }


def load_building_density(metadata_csv: Path, grid_size: int) -> list[list[int]]:
    counts = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lon = safe_float(row.get("centroid_lon"), float("nan"))
            lat = safe_float(row.get("centroid_lat"), float("nan"))
            if not (SF_BBOX["west"] <= lon <= SF_BBOX["east"] and SF_BBOX["south"] <= lat <= SF_BBOX["north"]):
                continue
            col = int((lon - SF_BBOX["west"]) / (SF_BBOX["east"] - SF_BBOX["west"]) * grid_size)
            row_idx = int((SF_BBOX["north"] - lat) / (SF_BBOX["north"] - SF_BBOX["south"]) * grid_size)
            if 0 <= row_idx < grid_size and 0 <= col < grid_size:
                counts[row_idx][col] += 1
    return counts


def synthetic_sf_elevation(lon: float, lat: float) -> float:
    hills = [
        (-122.4467, 37.7544, 280.0, 0.015, 0.013),  # Twin Peaks
        (-122.4570, 37.7600, 230.0, 0.016, 0.012),  # Mount Sutro
        (-122.4180, 37.7930, 110.0, 0.010, 0.010),  # Nob/Russian Hill
        (-122.4330, 37.7920, 95.0, 0.012, 0.010),   # Pacific Heights
        (-122.4030, 37.7480, 85.0, 0.018, 0.013),   # Potrero/Bernal
        (-122.4780, 37.7170, 120.0, 0.020, 0.012),  # Lake Merced / Merced Heights
    ]
    base = 6.0 + 16.0 * ((lat - SF_BBOX["south"]) / (SF_BBOX["north"] - SF_BBOX["south"]))
    elev = base
    for hx, hy, height, sx, sy in hills:
        elev += height * math.exp(-(((lon - hx) / sx) ** 2 + ((lat - hy) / sy) ** 2))
    return elev


def fetch_usgs_elevation(lon: float, lat: float) -> float | None:
    url = (
        "https://epqs.nationalmap.gov/v1/json"
        f"?x={lon:.7f}&y={lat:.7f}&units=Meters&wkid=4326&includeDate=false"
    )
    try:
        data = request_json(url, timeout=8.0)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    value = data.get("value")
    if isinstance(value, str):
        value = safe_float(value, float("nan"))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def load_or_build_terrain(
    cells: list[Cell],
    grid_size: int,
    cache_dir: Path,
    fetch_elevation: bool,
) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sf_usgs_epqs_elevation_{grid_size}.json"
    cached: dict[str, float] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = {key: safe_float(value, float("nan")) for key, value in raw.items()}

    source = "synthetic_sf_hills"
    changed = False
    for cell in cells:
        key = f"{cell.row},{cell.col}"
        value = cached.get(key, float("nan"))
        if not math.isfinite(value) and fetch_elevation:
            fetched_value = fetch_usgs_elevation(cell.lon, cell.lat)
            if fetched_value is not None:
                value = fetched_value
                cached[key] = value
                changed = True
                source = "USGS EPQS"
        if not math.isfinite(value):
            value = synthetic_sf_elevation(cell.lon, cell.lat)
        cell.elevation_m = max(0.0, value)

    if changed:
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    by_pos = {(c.row, c.col): c for c in cells}
    for cell in cells:
        west = by_pos.get((cell.row, cell.col - 1), cell)
        east = by_pos.get((cell.row, cell.col + 1), cell)
        north = by_pos.get((cell.row - 1, cell.col), cell)
        south = by_pos.get((cell.row + 1, cell.col), cell)
        dzdx = east.elevation_m - west.elevation_m
        dzdy = south.elevation_m - north.elevation_m
        cell.slope = math.hypot(dzdx, dzdy)
        cell.aspect_southness = clamp(dzdy / max(1.0, abs(dzdx) + abs(dzdy)), -1.0, 1.0)

    slopes = [c.slope for c in cells if c.land]
    slope_95 = sorted(slopes)[int(0.95 * (len(slopes) - 1))] if slopes else 1.0
    elevs = [c.elevation_m for c in cells if c.land]
    elev_min, elev_max = (min(elevs), max(elevs)) if elevs else (0.0, 1.0)
    for cell in cells:
        eastness = (cell.lon - SF_BBOX["west"]) / (SF_BBOX["east"] - SF_BBOX["west"])
        elev_norm = (cell.elevation_m - elev_min) / max(1.0, elev_max - elev_min)
        slope_norm = clamp(cell.slope / max(1.0, slope_95), 0.0, 1.0)
        fog_break = 0.58 + 0.52 * eastness + 0.08 * elev_norm
        aspect_bonus = 1.0 + 0.07 * cell.aspect_southness
        cell.solar_factor = clamp(fog_break * aspect_bonus, 0.45, 1.24)
        cell.productivity = clamp(0.72 + 0.34 * (1.0 - slope_norm) + 0.04 * eastness, 0.55, 1.10)

    return {
        "elevation_source": source if fetch_elevation else "synthetic_sf_hills",
        "elevation_cache": str(cache_path),
        "solar_microclimate": "west-to-east fog gradient plus south-facing aspect adjustment",
    }


def load_or_fetch_2025_solar(cache_dir: Path, fetch_climate: bool, steps: int) -> tuple[list[float], dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "nasa_power_sf_2025_hourly.json"
    raw: dict[str, Any] | None = None
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        "?parameters=ALLSKY_SFC_SW_DWN,T2M"
        "&community=SB&longitude=-122.4194&latitude=37.7749"
        "&start=20250101&end=20251231&format=JSON&time-standard=LST"
    )
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    elif fetch_climate:
        try:
            raw = request_json(url, timeout=60.0)
            cache_path.write_text(json.dumps(raw), encoding="utf-8")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            raw = None

    if raw:
        params = raw.get("properties", {}).get("parameter", {})
        values = list((params.get("ALLSKY_SFC_SW_DWN") or {}).values())
        solar = [max(0.0, safe_float(v, 0.0)) for v in values]
        if solar:
            max_value = max(solar)
            if max_value > 10.0:
                solar = [v / 1000.0 for v in solar]
            cf = [clamp(v * 0.82, 0.0, 1.0) for v in solar]
            if len(cf) >= steps:
                return cf[:steps], {
                    "climate_source": "NASA POWER hourly ALLSKY_SFC_SW_DWN, 2025, San Francisco point",
                    "climate_cache": str(cache_path),
                    "climate_url": url,
                }
            return [cf[i % len(cf)] for i in range(steps)], {
                "climate_source": "NASA POWER hourly ALLSKY_SFC_SW_DWN, repeated to requested steps",
                "climate_cache": str(cache_path),
                "climate_url": url,
            }

    cf = []
    for hour in range(steps):
        day = hour // 24
        hod = hour % 24
        season = 0.72 + 0.28 * math.sin(2 * math.pi * (day - 80) / 365)
        daylight = max(0.0, math.sin(math.pi * (hod - 6) / 13.5))
        cf.append(clamp(0.74 * season * daylight, 0.0, 0.86))
    return cf, {
        "climate_source": "synthetic seasonal clear-sky fallback",
        "climate_cache": str(cache_path),
        "climate_url": url,
    }


def build_cells(metadata_csv: Path, grid_size: int, seed: int) -> list[Cell]:
    counts = load_building_density(metadata_csv, grid_size)
    flat_counts = [v for row in counts for v in row if v > 0]
    min_count = 1 if not flat_counts else max(1, int(median(flat_counts) * 0.10))
    cells: list[Cell] = []
    for row in range(grid_size):
        for col in range(grid_size):
            lon = SF_BBOX["west"] + (col + 0.5) / grid_size * (SF_BBOX["east"] - SF_BBOX["west"])
            lat = SF_BBOX["north"] - (row + 0.5) / grid_size * (SF_BBOX["north"] - SF_BBOX["south"])
            building_count = counts[row][col]
            land = building_count >= min_count
            cell = Cell(
                row=row,
                col=col,
                lon=lon,
                lat=lat,
                building_count=building_count,
                land=land,
                money=0.0,
                occupied=False,
            )
            cell.role = "empty"
            cells.append(cell)
    return cells


def neighbors(row: int, col: int, grid_size: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if 0 <= rr < grid_size and 0 <= cc < grid_size:
                out.append((rr, cc))
    return out


def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted([a, b]))  # type: ignore[return-value]


def missing_edges_on_path(
    path: list[tuple[int, int]],
    cables: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    missing = []
    for left, right in zip(path, path[1:]):
        edge = edge_key(left, right)
        if edge not in cables:
            missing.append(edge)
    return missing


def find_surplus_path(
    requester: Cell,
    hour_surplus: dict[tuple[int, int], float],
    by_pos: dict[tuple[int, int], Cell],
    grid_size: int,
    cables: set[tuple[tuple[int, int], tuple[int, int]]],
    cable_cost: float,
) -> list[tuple[int, int]] | None:
    start = (requester.row, requester.col)
    heap: list[tuple[float, int, int, tuple[int, int], list[tuple[int, int]]]] = [(0.0, 0, 0, start, [start])]
    best_state = {(start, 0): 0.0}
    while heap:
        build_cost, missing_count, hops, pos, path = heapq.heappop(heap)
        if build_cost > requester.money:
            continue
        if pos != start and hour_surplus.get(pos, 0.0) > 1e-9:
            return path
        for npos in neighbors(pos[0], pos[1], grid_size):
            neighbor = by_pos.get(npos)
            if neighbor is None or not neighbor.land or not neighbor.occupied:
                continue
            edge = edge_key(pos, npos)
            edge_missing = edge not in cables
            next_missing_count = missing_count + (1 if edge_missing else 0)
            if next_missing_count > 1:
                continue
            next_cost = build_cost + (cable_cost if edge_missing else 0.0)
            if next_cost > requester.money:
                continue
            state = (npos, next_missing_count)
            if next_cost >= best_state.get(state, float("inf")):
                continue
            best_state[state] = next_cost
            heapq.heappush(heap, (next_cost, next_missing_count, hops + 1, npos, path + [npos]))
    return None


def build_surplus_reachability(
    hour_surplus: dict[tuple[int, int], float],
    cables: set[tuple[tuple[int, int], tuple[int, int]]],
    by_pos: dict[tuple[int, int], Cell],
) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[tuple[int, int], tuple[int, int]]]:
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for left, right in cables:
        if left not in by_pos or right not in by_pos:
            continue
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    root: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[int, int]] = []
    for pos, surplus in hour_surplus.items():
        if surplus <= 1e-9 or pos not in by_pos:
            continue
        parent[pos] = None
        root[pos] = pos
        queue.append(pos)
    cursor = 0
    while cursor < len(queue):
        pos = queue[cursor]
        cursor += 1
        for nxt in adjacency.get(pos, []):
            if nxt in parent:
                continue
            parent[nxt] = pos
            root[nxt] = root[pos]
            queue.append(nxt)
    return parent, root


def path_to_surplus_root(
    start: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path = [start]
    pos = start
    while parent.get(pos) is not None:
        pos = parent[pos]  # type: ignore[assignment]
        path.append(pos)
    return path


def agreement_component_stats(
    cells: list[Cell],
    cables: set[tuple[tuple[int, int], tuple[int, int]]],
) -> dict[str, Any]:
    occupied = {(cell.row, cell.col): cell for cell in cells if cell.land and cell.occupied}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {pos: [] for pos in occupied}
    for left, right in cables:
        if left in adjacency and right in adjacency:
            adjacency[left].append(right)
            adjacency[right].append(left)
    seen: set[tuple[int, int]] = set()
    sizes = []
    components: list[list[tuple[int, int]]] = []
    for pos in adjacency:
        if pos in seen:
            continue
        stack = [pos]
        seen.add(pos)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) > 1:
            sizes.append(len(comp))
            components.append(comp)
    largest = max(sizes) if sizes else 0
    mean_size = sum(sizes) / len(sizes) if sizes else 0.0
    return {
        "agreement_components": len(sizes),
        "largest_agreement_component": largest,
        "mean_agreement_component_size": mean_size,
        "agreement_depth_index": math.log2(largest) if largest > 1 else 0.0,
        "agreement_components_raw": components,
    }


def initialize_residential_startup(cell: Cell, rng: random.Random, costs: dict[str, float]) -> None:
    cell.occupied = True
    cell.use_type = "residential"
    cell.role = "residential"
    cell.floors = 1
    cell.pv_kw = 0.0
    cell.battery_kwh = 0.0
    cell.battery_soc = 0.0
    cell.daily_demand = 0.0
    cell.daily_served = 0.0
    cell.daily_surplus = 0.0
    cell.lifetime_demand = 0.0
    cell.lifetime_served = 0.0
    cell.lifetime_generated = 0.0
    cell.lifetime_imported = 0.0
    cell.lifetime_exported = 0.0
    cell.lifetime_transit = 0.0
    cell.lifetime_transit_fee = 0.0
    cell.cables.clear()
    mean = costs["startup_funding_mean"]
    spread = costs["startup_funding_spread"]
    cell.money = rng.uniform(mean - spread, mean + spread)


def add_residential_startups(
    cells: list[Cell],
    rng: random.Random,
    costs: dict[str, float],
) -> int:
    candidates = [cell for cell in cells if cell.land and not cell.occupied]
    if not candidates:
        return 0
    buildable = [cell for cell in cells if cell.land]
    solar_min = min(cell.solar_factor for cell in buildable)
    solar_max = max(cell.solar_factor for cell in buildable)
    prod_min = min(cell.productivity for cell in buildable)
    prod_max = max(cell.productivity for cell in buildable)
    solar_weight = costs["residential_site_solar_weight"]
    productivity_weight = costs["residential_site_productivity_weight"]

    def score(cell: Cell) -> float:
        solar_norm = (cell.solar_factor - solar_min) / max(1e-9, solar_max - solar_min)
        prod_norm = (cell.productivity - prod_min) / max(1e-9, prod_max - prod_min)
        return solar_weight * solar_norm + productivity_weight * prod_norm + rng.uniform(-0.035, 0.035)

    entrants = min(len(candidates), int(costs["new_residential_startups_per_day"]))
    chosen = sorted(candidates, key=score, reverse=True)[:entrants]
    for cell in chosen:
        initialize_residential_startup(cell, rng, costs)
    return len(chosen)


def decide_investments(
    cells: list[Cell],
    grid_size: int,
    rng: random.Random,
    costs: dict[str, float],
    day: int,
) -> dict[str, int]:
    actions = {"pv": 0, "battery": 0, "floor": 0, "commercial": 0}
    occupied_cells = [cell for cell in cells if cell.land and cell.occupied]
    commercial_cells = [cell for cell in occupied_cells if cell.use_type == "commercial"]
    commercial_target_count = math.ceil(costs["commercial_target_share"] * len(occupied_cells))
    commercial_conversion_slots = max(0, commercial_target_count - len(commercial_cells))
    commercial_share = len(commercial_cells) / max(1, len(occupied_cells))
    commercial_pressure = clamp(
        (costs["commercial_target_share"] - commercial_share) / max(1e-9, costs["commercial_target_share"]),
        0.0,
        1.0,
    )
    prod_min = min((cell.productivity for cell in occupied_cells), default=0.8)
    prod_max = max((cell.productivity for cell in occupied_cells), default=1.1)
    for cell in cells:
        if not cell.land or not cell.occupied:
            continue
        daily_service = cell.daily_served / cell.daily_demand if cell.daily_demand > 0 else 1.0
        reserve = 1_000.0 + 250.0 * cell.floors
        slope_cost = 1.0 + clamp(cell.slope / 80.0, 0.0, 0.75)
        upgrade_cost = costs["commercial_upgrade_cost"] * slope_cost
        residential_effect = 1.0 + costs["residential_productivity_scale"] * (cell.productivity - 1.0)
        commercial_effect = 1.0 + costs["commercial_productivity_scale"] * (cell.productivity - 1.0)
        current_income = 125.0 * cell.floors * residential_effect * clamp(daily_service, 0.15, 1.05)
        commercial_income = 520.0 * max(cell.floors, 2) * commercial_effect * clamp(daily_service, 0.15, 1.05)
        payback_days = upgrade_cost / max(1.0, commercial_income - current_income)
        prod_norm = clamp((cell.productivity - prod_min) / max(1e-9, prod_max - prod_min), 0.0, 1.0)
        commercial_readiness = (
            0.42 * prod_norm
            + 0.26 * clamp(daily_service, 0.0, 1.0)
            + 0.20 * clamp(cell.money / max(1.0, reserve + upgrade_cost + 2_000.0), 0.0, 1.0)
            + 0.12 * clamp(cell.cable_degree / 4.0, 0.0, 1.0)
        )
        conversion_chance = clamp(
            costs["commercial_base_conversion_chance"]
            + costs["commercial_pressure_conversion_chance"] * commercial_pressure
            + 0.22 * commercial_readiness,
            0.0,
            0.88,
        )
        if (
            cell.use_type == "residential"
            and actions["commercial"] < commercial_conversion_slots
            and daily_service > costs["commercial_min_service"]
            and cell.money > reserve + upgrade_cost
            and payback_days < costs["commercial_payback_days"]
            and rng.random() < conversion_chance
        ):
            cell.money -= upgrade_cost
            cell.use_type = "commercial"
            cell.floors = max(cell.floors, 2)
            actions["commercial"] += 1

        battery_step = costs["battery_unit_kwh"]
        battery_cost = costs["battery_unit_cost"]
        if (
            cell.pv_kw >= 2.0
            and cell.battery_kwh < battery_step
            and cell.money > reserve + battery_cost
            and (cell.daily_surplus > 0.25 or cell.solar_factor > 0.66 or daily_service < 0.82)
        ):
            cell.money -= battery_cost
            cell.battery_kwh += battery_step
            actions["battery"] += 1

        pv_step_kw = 2.0
        roof_limit_kw = 4.0 + 2.8 * cell.floors + (4.0 if cell.use_type == "commercial" else 0.0)
        pv_cost = costs["pv_capex_per_kw"] * pv_step_kw
        wants_first_pv = cell.pv_kw <= 0.0 and (cell.solar_factor > 0.52 or day < 10)
        wants_more_pv = (
            cell.pv_kw > 0.0
            and (cell.battery_kwh >= battery_step or daily_service < 0.64)
            and (cell.solar_factor > 0.62 or daily_service < 0.56)
        )
        if (
            cell.pv_kw + pv_step_kw <= roof_limit_kw
            and cell.money > reserve + pv_cost
            and (wants_first_pv or wants_more_pv)
        ):
            cell.money -= pv_cost
            cell.pv_kw += pv_step_kw
            actions["pv"] += 1

        floor_cost = costs["floor_cost"] * slope_cost
        if cell.use_type == "residential" and cell.floors < 4 and cell.money > reserve + floor_cost:
            cell.money -= floor_cost
            cell.floors += 1
            actions["floor"] += 1

        cell.daily_demand = 0.0
        cell.daily_served = 0.0
        cell.daily_surplus = 0.0
    return actions


def sync_building_functions(cells: list[Cell]) -> None:
    for cell in cells:
        if not cell.land or not cell.occupied:
            cell.role = "empty"
        else:
            cell.role = cell.use_type


def simulate(
    cells: list[Cell],
    grid_size: int,
    res_profile: list[float],
    com_profile: list[float],
    solar_cf: list[float],
    seed: int,
    steps: int,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, float], list[tuple[int, Image.Image]]]:
    rng = random.Random(seed + 101)
    costs = {
        "startup_funding_mean": 10_000.0,
        "startup_funding_spread": 1_500.0,
        "residential_site_solar_weight": 0.55,
        "residential_site_productivity_weight": 0.45,
        "residential_productivity_scale": 0.45,
        "commercial_productivity_scale": 1.25,
        "commercial_target_share": 0.10,
        "commercial_min_service": 0.24,
        "commercial_payback_days": 260.0,
        "commercial_base_conversion_chance": 0.03,
        "commercial_pressure_conversion_chance": 0.62,
        "pv_capex_per_kw": 2_410.0,
        "battery_unit_kwh": 10.0,
        "battery_unit_cost": 4_660.0,
        "cable_cost": 650.0,
        "floor_cost": 6_000.0,
        "commercial_upgrade_cost": 3_600.0,
        "pv_om_per_kw_day": 0.10,
        "battery_om_per_kwh_day": 0.015,
        "cable_om_per_link_day": 0.04,
        "transmission_efficiency_per_edge": 0.965,
        "transit_fee_per_kwh": 0.012,
    }
    buildable_count = sum(1 for c in cells if c.land)
    total_days = max(1, math.ceil(steps / 24))
    costs["new_residential_startups_per_day"] = max(1, round(buildable_count / total_days))
    cables: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    metrics: list[dict[str, Any]] = []
    frames: list[tuple[int, Image.Image]] = [
        (0, draw_map(cells, grid_size, {}, title="SF terrain energy growth, day 0 empty grid"))
    ]
    frame_days = set([0, 7, 14, 30, 60, 90, 150, 240, 364])
    frame_days.update(range(0, max(1, steps // 24), 30))
    surplus_map: dict[tuple[int, int], float] = {}
    day_demand_total = 0.0
    day_served_total = 0.0
    day_generated_total = 0.0
    day_shared_total = 0.0
    day_transit_fee_total = 0.0
    day_requests = 0
    day_successes = 0
    day_cable_builds = 0
    day_new_residential = 0

    for hour in range(steps):
        if hour % 24 == 0:
            day_new_residential += add_residential_startups(cells, rng, costs)
            sync_building_functions(cells)
        by_pos = {(c.row, c.col): c for c in cells if c.land and c.occupied}
        day = hour // 24
        hour_surplus: dict[tuple[int, int], float] = {}
        deficits: list[tuple[Cell, float]] = []
        demand_total = served_total = generated_total = 0.0

        for cell in cells:
            if not cell.land or not cell.occupied:
                continue
            profile = com_profile if cell.use_type == "commercial" else res_profile
            raw_demand = profile[hour % len(profile)]
            load_scale = cell.floors * (0.82 if cell.use_type == "residential" else 0.12)
            demand = raw_demand * load_scale
            generation = cell.pv_kw * solar_cf[hour % len(solar_cf)] * cell.solar_factor
            generated_total += generation
            served = min(demand, generation)
            balance = generation - demand
            if balance >= 0:
                charge_room = max(0.0, cell.battery_kwh - cell.battery_soc)
                charge = min(charge_room, balance * 0.92)
                cell.battery_soc += charge
                surplus = max(0.0, balance - charge / 0.92)
                hour_surplus[(cell.row, cell.col)] = surplus
                cell.daily_surplus += surplus
            else:
                deficit = -balance
                discharge = min(cell.battery_soc, deficit / 0.92)
                cell.battery_soc -= discharge
                served += discharge * 0.92
                deficit = max(0.0, demand - served)
                if deficit > 1e-9:
                    deficits.append((cell, deficit))
            cell.daily_demand += demand
            cell.daily_served += served
            cell.lifetime_demand += demand
            cell.lifetime_served += served
            cell.lifetime_generated += generation
            demand_total += demand
            served_total += served

        rng.shuffle(deficits)
        cable_builds = 0
        shared_total = 0.0
        transit_fee_total = 0.0
        requests = len(deficits)
        successes = 0
        reachable_parent, reachable_root = build_surplus_reachability(hour_surplus, cables, by_pos)
        for requester, deficit in deficits:
            if deficit <= 0:
                continue
            requester_pos = (requester.row, requester.col)
            path = None
            if requester_pos in reachable_root:
                path = path_to_surplus_root(requester_pos, reachable_parent)
            else:
                neigh = neighbors(requester.row, requester.col, grid_size)
                rng.shuffle(neigh)
                for npos in neigh:
                    if npos not in reachable_root:
                        continue
                    edge = edge_key(requester_pos, npos)
                    if edge in cables:
                        path = [requester_pos] + path_to_surplus_root(npos, reachable_parent)
                        break
                    if requester.money >= costs["cable_cost"]:
                        path = [requester_pos] + path_to_surplus_root(npos, reachable_parent)
                        break
            if not path or len(path) < 2:
                continue
            donor_pos = path[-1]
            donor = by_pos[donor_pos]
            available = hour_surplus.get(donor_pos, 0.0)
            if available <= 1e-9:
                continue
            missing_edges = missing_edges_on_path(path, cables)
            build_cost = len(missing_edges) * costs["cable_cost"]
            hops = len(path) - 1
            efficiency = costs["transmission_efficiency_per_edge"] ** hops
            transfer = min(deficit / max(1e-9, efficiency), available)
            delivered = transfer * efficiency
            transit_nodes = path[1:-1]
            transit_fee = delivered * costs["transit_fee_per_kwh"] * len(transit_nodes)
            if requester.money < build_cost + transit_fee:
                continue
            requester.money -= build_cost + transit_fee
            if transit_nodes and transit_fee > 0:
                fee_each = transit_fee / len(transit_nodes)
                for pos in transit_nodes:
                    node = by_pos[pos]
                    node.money += fee_each
                    node.lifetime_transit += delivered
                    node.lifetime_transit_fee += fee_each
            for edge in missing_edges:
                left, right = edge
                cables.add(edge)
                by_pos[left].cables.add(right)
                by_pos[right].cables.add(left)
                cable_builds += 1
            hour_surplus[donor_pos] -= transfer
            requester.daily_served += delivered
            requester.lifetime_served += delivered
            requester.lifetime_imported += delivered
            donor.lifetime_exported += transfer
            served_total += delivered
            shared_total += delivered
            transit_fee_total += transit_fee
            successes += 1

        day_demand_total += demand_total
        day_served_total += served_total
        day_generated_total += generated_total
        day_shared_total += shared_total
        day_transit_fee_total += transit_fee_total
        day_requests += requests
        day_successes += successes
        day_cable_builds += cable_builds

        if hour % 24 == 23:
            for cell in cells:
                if not cell.land or not cell.occupied:
                    continue
                service = cell.daily_served / cell.daily_demand if cell.daily_demand > 0 else 1.0
                income_base = 125.0 if cell.use_type == "residential" else 520.0
                productivity_scale = (
                    costs["commercial_productivity_scale"]
                    if cell.use_type == "commercial"
                    else costs["residential_productivity_scale"]
                )
                productivity_effect = 1.0 + productivity_scale * (cell.productivity - 1.0)
                income = income_base * cell.floors * productivity_effect * clamp(service, 0.15, 1.05)
                maintenance = (
                    cell.pv_kw * costs["pv_om_per_kw_day"]
                    + cell.battery_kwh * costs["battery_om_per_kwh_day"]
                    + cell.cable_degree * costs["cable_om_per_link_day"]
                )
                unmet_penalty = max(0.0, cell.daily_demand - cell.daily_served) * 0.18
                cell.money += income - maintenance - unmet_penalty
            daily_actions = decide_investments(cells, grid_size, rng, costs, day)
            sync_building_functions(cells)
            land_cells = [c for c in cells if c.land and c.occupied]
            component_stats = agreement_component_stats(cells, cables)
            commercial_cells = sum(1 for c in land_cells if c.use_type == "commercial")
            metrics.append({
                "day": day + 1,
                "served_fraction": day_served_total / day_demand_total if day_demand_total > 0 else 1.0,
                "buildable_cells": buildable_count,
                "occupied_cells": len(land_cells),
                "empty_buildable_cells": buildable_count - len(land_cells),
                "new_residential": day_new_residential,
                "mean_service_lifetime": sum(c.service_ratio for c in land_cells) / max(1, len(land_cells)),
                "total_money": sum(c.money for c in land_cells),
                "mean_money": sum(c.money for c in land_cells) / max(1, len(land_cells)),
                "pv_kw": sum(c.pv_kw for c in land_cells),
                "battery_kwh": sum(c.battery_kwh for c in land_cells),
                "residential_cells": len(land_cells) - commercial_cells,
                "commercial_cells": commercial_cells,
                "commercial_share": commercial_cells / max(1, len(land_cells)),
                "cables": len(cables),
                "cable_builds": day_cable_builds,
                "shared_kwh": day_shared_total,
                "transit_fees": day_transit_fee_total,
                "requests": day_requests,
                "successes": day_successes,
                "agreement_components": component_stats["agreement_components"],
                "largest_agreement_component": component_stats["largest_agreement_component"],
                "mean_agreement_component_size": component_stats["mean_agreement_component_size"],
                "agreement_depth_index": component_stats["agreement_depth_index"],
                **{f"invest_{k}": v for k, v in daily_actions.items()},
            })
            if day in frame_days:
                frames.append((day + 1, draw_map(cells, grid_size, metrics[-1], title=f"SF terrain energy growth, day {day + 1}")))
            day_demand_total = 0.0
            day_served_total = 0.0
            day_generated_total = 0.0
            day_shared_total = 0.0
            day_transit_fee_total = 0.0
            day_requests = 0
            day_successes = 0
            day_cable_builds = 0
            day_new_residential = 0
        surplus_map = hour_surplus

    if not frames:
        sync_building_functions(cells)
        frames.append((0, draw_map(cells, grid_size, metrics[-1] if metrics else {}, title="SF terrain energy growth")))
    return metrics, costs, frames


def blend(color: tuple[int, int, int], alpha: float, bg: tuple[int, int, int] = (248, 250, 252)) -> tuple[int, int, int]:
    return tuple(round(bg[i] * (1 - alpha) + color[i] * alpha) for i in range(3))


def draw_map(cells: list[Cell], grid_size: int, metric: dict[str, Any], title: str) -> Image.Image:
    cell_px = max(13, min(24, 620 // grid_size))
    margin_l = 56
    margin_t = 72
    grid_w = grid_size * cell_px
    width = grid_w + 430
    height = margin_t + grid_w + 54
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), title, fill=(15, 23, 42))
    draw.text((30, 44), "terrain tint = elevation, cell color = building function only, markers = PV/battery assets", fill=(71, 85, 105))

    elevs = [c.elevation_m for c in cells if c.land]
    emin, emax = (min(elevs), max(elevs)) if elevs else (0.0, 1.0)
    money_vals = [max(0.0, c.money) for c in cells if c.land and c.occupied]
    m95 = sorted(money_vals)[int(0.95 * (len(money_vals) - 1))] if money_vals else 1.0

    for cell in cells:
        x = margin_l + cell.col * cell_px
        y = margin_t + cell.row * cell_px
        if not cell.land:
            draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=(236, 241, 246))
            continue
        elev_norm = (cell.elevation_m - emin) / max(1.0, emax - emin)
        terrain = (
            round(218 - 54 * elev_norm),
            round(230 - 42 * elev_norm),
            round(204 - 72 * elev_norm),
        )
        if not cell.occupied:
            fill = blend(ROLE_COLORS["empty"], 0.62, terrain)
            draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=fill)
            continue
        role_color = ROLE_COLORS[cell.use_type]
        money_alpha = clamp(0.42 + 0.42 * cell.money / max(1.0, m95), 0.28, 0.86)
        fill = blend(role_color, money_alpha, terrain)
        draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=fill)
        if cell.pv_kw > 0:
            draw.rectangle([x + 2, y + 2, x + 5, y + 5], fill=(255, 238, 88))
        if cell.battery_kwh > 0:
            draw.rectangle([x + cell_px - 6, y + 2, x + cell_px - 3, y + 5], fill=(67, 160, 221))
        if cell.use_type == "commercial":
            draw.rectangle([x + 2, y + cell_px - 5, x + 5, y + cell_px - 2], fill=(120, 70, 25))

    drawn_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    by_pos = {(c.row, c.col): c for c in cells if c.land and c.occupied}
    for cell in cells:
        if not cell.land or not cell.occupied:
            continue
        p0 = (margin_l + cell.col * cell_px + cell_px // 2, margin_t + cell.row * cell_px + cell_px // 2)
        for npos in cell.cables:
            edge = edge_key((cell.row, cell.col), npos)
            if edge in drawn_edges:
                continue
            nb = by_pos.get(npos)
            if nb is None:
                continue
            p1 = (margin_l + nb.col * cell_px + cell_px // 2, margin_t + nb.row * cell_px + cell_px // 2)
            draw.line([p0, p1], fill=(34, 45, 64), width=1)
            drawn_edges.add(edge)

    lx = margin_l + grid_w + 34
    ly = margin_t
    draw.text((lx, ly), "Building functions", fill=(15, 23, 42))
    ly += 24
    draw.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=ROLE_COLORS["empty"])
    draw.text((lx + 24, ly), "empty buildable terrain", fill=(15, 23, 42))
    ly += 23
    for role in ["residential", "commercial"]:
        draw.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=ROLE_COLORS[role])
        draw.text((lx + 24, ly), role, fill=(15, 23, 42))
        ly += 23
    ly += 6
    draw.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=(255, 238, 88))
    draw.text((lx + 24, ly), "PV asset marker", fill=(15, 23, 42))
    ly += 23
    draw.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=(67, 160, 221))
    draw.text((lx + 24, ly), "battery asset marker", fill=(15, 23, 42))
    ly += 23
    ly += 12
    draw.text((lx, ly), f"served today: {metric.get('served_fraction', 0):.2f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"mean lifetime service: {metric.get('mean_service_lifetime', 0):.2f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"PV: {metric.get('pv_kw', 0):.0f} kW", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"battery: {metric.get('battery_kwh', 0):.0f} kWh", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"cables: {metric.get('cables', 0):.0f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"commercial cells: {metric.get('commercial_cells', 0):.0f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"residential cells: {metric.get('residential_cells', 0):.0f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"agreement comps: {metric.get('agreement_components', 0):.0f}", fill=(15, 23, 42))
    ly += 23
    draw.text((lx, ly), f"largest comp: {metric.get('largest_agreement_component', 0):.0f}", fill=(15, 23, 42))
    ly += 34
    draw.text((lx, ly), "Costs used", fill=(15, 23, 42))
    ly += 22
    draw.text((lx, ly), "PV $2.41/Wdc; battery $466/kWh x 10 kWh", fill=(71, 85, 105))
    ly += 20
    draw.text((lx, ly), "Cable $650/link; floor $6,000; commercial $3.6k", fill=(71, 85, 105))
    return image


def draw_metrics(metrics: list[dict[str, Any]], path: Path) -> None:
    width, height = 1120, 870
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), "SF terrain energy growth metrics", fill=(15, 23, 42))

    panels = [
        ("service / sharing", [("served_fraction", (22, 163, 74)), ("mean_service_lifetime", (37, 99, 235))], 0.0, 1.05),
        ("infrastructure", [("pv_kw", (234, 179, 8)), ("battery_kwh", (14, 165, 233)), ("cables", (15, 23, 42))], None, None),
        ("agreement hierarchy", [("agreement_components", (99, 102, 241)), ("largest_agreement_component", (14, 116, 144)), ("agreement_depth_index", (147, 51, 234))], None, None),
        ("building functions only", [("residential_cells", (105, 180, 166)), ("commercial_cells", (214, 158, 80))], None, None),
    ]
    for idx, (title, series, ymin, ymax) in enumerate(panels):
        x0, y0 = 72, 78 + idx * 190
        w, h = 960, 135
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(203, 213, 225))
        draw.text((x0, y0 - 22), title, fill=(15, 23, 42))
        max_day = max(1, metrics[-1]["day"]) if metrics else 1
        if ymin is None or ymax is None:
            vals = [m[key] for key, _ in series for m in metrics]
            ymin = 0.0
            ymax = max(1.0, max(vals) if vals else 1.0)
        for tick in range(5):
            y = y0 + h - round(h * tick / 4)
            draw.line([x0, y, x0 + w, y], fill=(226, 232, 240))
            draw.text((x0 - 54, y - 6), f"{ymin + (ymax - ymin) * tick / 4:.1f}", fill=(71, 85, 105))
        for key, color in series:
            pts = []
            for m in metrics:
                x = x0 + round(w * m["day"] / max_day)
                y = y0 + h - round(h * clamp((m[key] - ymin) / max(1e-9, ymax - ymin), 0, 1))
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=3)
        lx = x0 + 14
        for key, color in series:
            draw.rectangle([lx, y0 + 12, lx + 16, y0 + 24], fill=color)
            draw.text((lx + 22, y0 + 8), key, fill=(15, 23, 42))
            lx += 230
    image.save(path)


def write_agreement_components_png(cells: list[Cell], grid_size: int, path: Path) -> None:
    cell_px = max(16, min(26, 700 // grid_size))
    margin_l = 58
    margin_t = 76
    grid_w = grid_size * cell_px
    width = grid_w + 420
    height = margin_t + grid_w + 60
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((30, 22), "Emergent agreement components", fill=(15, 23, 42))
    draw.text((30, 44), "Components are connected sets of paid adjacent transmission agreements; no block grid is preassigned.", fill=(71, 85, 105))

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    by_pos = {(c.row, c.col): c for c in cells}
    for cell in cells:
        for npos in cell.cables:
            edges.add(edge_key((cell.row, cell.col), npos))
    stats = agreement_component_stats(cells, edges)
    comps = stats["agreement_components_raw"]
    comp_by_pos: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(sorted(comps, key=len, reverse=True)):
        for pos in comp:
            comp_by_pos[pos] = idx
    palette = [
        (99, 102, 241),
        (14, 116, 144),
        (245, 158, 11),
        (147, 51, 234),
        (22, 163, 74),
        (220, 38, 38),
        (2, 132, 199),
        (180, 83, 9),
    ]

    for cell in cells:
        x = margin_l + cell.col * cell_px
        y = margin_t + cell.row * cell_px
        if not cell.land:
            draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=(236, 241, 246))
            continue
        if not cell.occupied:
            draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=(226, 232, 240), outline=(248, 250, 252))
            continue
        comp_idx = comp_by_pos.get((cell.row, cell.col), -1)
        if comp_idx < 0:
            fill = (226, 232, 240)
        else:
            base = palette[comp_idx % len(palette)]
            fill = blend(base, 0.62)
        draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], fill=fill, outline=(248, 250, 252))
        if cell.use_type == "commercial":
            draw.rectangle([x + 3, y + cell_px - 6, x + 7, y + cell_px - 3], fill=(120, 70, 25))

    for left, right in edges:
        if left not in by_pos or right not in by_pos:
            continue
        x0 = margin_l + left[1] * cell_px + cell_px // 2
        y0 = margin_t + left[0] * cell_px + cell_px // 2
        x1 = margin_l + right[1] * cell_px + cell_px // 2
        y1 = margin_t + right[0] * cell_px + cell_px // 2
        draw.line([x0, y0, x1, y1], fill=(15, 23, 42), width=1)

    lx = margin_l + grid_w + 34
    ly = margin_t
    draw.text((lx, ly), "Agreement graph", fill=(15, 23, 42))
    ly += 28
    rows = [
        ("components", stats["agreement_components"]),
        ("largest component", stats["largest_agreement_component"]),
        ("mean component size", f"{stats['mean_agreement_component_size']:.1f}"),
        ("depth index log2(largest)", f"{stats['agreement_depth_index']:.2f}"),
        ("agreement edges", len(edges)),
    ]
    for label, value in rows:
        draw.text((lx, ly), f"{label}: {value}", fill=(15, 23, 42))
        ly += 24
    ly += 16
    draw.text((lx, ly), "Largest components", fill=(15, 23, 42))
    ly += 26
    for idx, comp in enumerate(sorted(comps, key=len, reverse=True)[:8]):
        color = palette[idx % len(palette)]
        draw.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=color)
        draw.text((lx + 24, ly), f"component {idx + 1}: {len(comp)} buildings", fill=(15, 23, 42))
        ly += 23
    image.save(path)


def write_hierarchy_canopy_png(cells: list[Cell], grid_size: int, path: Path) -> None:
    width, height = 1280, 1040
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((42, 34), "2D hierarchy canopy", fill=(15, 23, 42))
    draw.text(
        (42, 58),
        "Base plane is the SF building grid. Elevated canopies are connected components of paid adjacent transmission agreements; height shows component strength.",
        fill=(71, 85, 105),
    )

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    by_pos = {(c.row, c.col): c for c in cells if c.land and c.occupied}
    for cell in cells:
        if not cell.land or not cell.occupied:
            continue
        for npos in cell.cables:
            edges.add(edge_key((cell.row, cell.col), npos))
    stats = agreement_component_stats(cells, edges)
    comps = sorted(stats["agreement_components_raw"], key=len, reverse=True)
    comp_by_pos: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for pos in comp:
            comp_by_pos[pos] = idx

    tile_w = 22
    tile_h = 12
    origin_x = 505
    origin_y = 355

    def iso(row: int, col: int, z: float = 0.0) -> tuple[float, float]:
        return (
            origin_x + (col - row) * tile_w * 0.5,
            origin_y + (col + row) * tile_h * 0.5 - z,
        )

    def diamond(row: int, col: int, z: float = 0.0) -> list[tuple[float, float]]:
        cx, cy = iso(row, col, z)
        return [
            (cx, cy - tile_h * 0.5),
            (cx + tile_w * 0.5, cy),
            (cx, cy + tile_h * 0.5),
            (cx - tile_w * 0.5, cy),
        ]

    palette = [
        (30, 153, 213),
        (34, 163, 84),
        (148, 65, 220),
        (215, 137, 23),
        (99, 102, 241),
        (107, 114, 128),
    ]
    largest = max((len(comp) for comp in comps), default=1)

    for cell in cells:
        if not cell.land:
            continue
        pos = (cell.row, cell.col)
        if not cell.occupied:
            fill = (221, 230, 223)
        elif cell.use_type == "commercial":
            fill = (214, 158, 80)
        else:
            comp_idx = comp_by_pos.get(pos, -1)
            if comp_idx >= 0:
                fill = blend(palette[comp_idx % len(palette)], 0.28, ROLE_COLORS["residential"])
            else:
                fill = ROLE_COLORS["residential"]
        pts = diamond(cell.row, cell.col, 0.0)
        draw.polygon(pts, fill=fill, outline=(255, 255, 255))

    for left, right in edges:
        if left not in by_pos or right not in by_pos:
            continue
        draw.line([iso(left[0], left[1]), iso(right[0], right[1])], fill=(45, 55, 72), width=1)

    top_components = comps[:6]
    for idx, comp in enumerate(top_components):
        color = palette[idx % len(palette)]
        z = 55 + 180 * (len(comp) / largest)
        comp_set = set(comp)
        for row, col in comp:
            if (row + col) % 4 == 0:
                base = iso(row, col, 0.0)
                top = iso(row, col, z)
                draw.line([base, top], fill=blend(color, 0.45), width=1)
        for row, col in sorted(comp, key=lambda p: p[0] + p[1]):
            pts = diamond(row, col, z)
            draw.polygon(pts, fill=blend(color, 0.62), outline=color)
            east = (row, col + 1)
            south = (row + 1, col)
            if east not in comp_set:
                p1 = iso(row, col, z)
                p2 = iso(row, col + 1, z)
                draw.line([p1, p2], fill=color, width=2)
            if south not in comp_set:
                p1 = iso(row, col, z)
                p2 = iso(row + 1, col, z)
                draw.line([p1, p2], fill=color, width=2)
        center_row = sum(row for row, _ in comp) / len(comp)
        center_col = sum(col for _, col in comp) / len(comp)
        lx, ly = iso(round(center_row), round(center_col), z + 18)
        label = f"C{idx + 1}: {len(comp)}"
        tw = 7 * len(label) + 12
        draw.rectangle([lx - tw / 2, ly - 12, lx + tw / 2, ly + 8], fill=(248, 250, 252), outline=color)
        draw.text((lx - tw / 2 + 6, ly - 9), label, fill=(15, 23, 42))

    lx, ly = 42, height - 132
    draw.text((lx, ly), "Canopy colors", fill=(15, 23, 42))
    ly += 28
    for idx, comp in enumerate(top_components):
        color = palette[idx % len(palette)]
        x = lx + (idx % 3) * 220
        y = ly + (idx // 3) * 30
        draw.rectangle([x, y, x + 18, y + 18], fill=color)
        draw.text((x + 26, y + 2), f"component {idx + 1}: {len(comp)} buildings", fill=(71, 85, 105))
    draw.text(
        (42, height - 34),
        "Vertical lines connect buildings to their emergent agreement component. No canopy is used by the simulation itself.",
        fill=(71, 85, 105),
    )
    image.save(path)


def write_outputs(
    out_dir: Path,
    cells: list[Cell],
    grid_size: int,
    metrics: list[dict[str, Any]],
    frames: list[tuple[int, Image.Image]],
    costs: dict[str, float],
    sources: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_images = [image for _, image in frames]
    frame_images[0].save(
        out_dir / "sf_energy_growth_evolution.gif",
        save_all=True,
        append_images=frame_images[1:],
        duration=180,
        loop=0,
        optimize=False,
    )
    final_image = frames[-1][1]
    if metrics and frames[-1][0] != metrics[-1]["day"]:
        final_image = draw_map(cells, grid_size, metrics[-1], title=f"SF terrain energy growth, day {metrics[-1]['day']}")
    final_image.save(out_dir / "sf_energy_growth_final.png")
    draw_metrics(metrics, out_dir / "sf_energy_growth_metrics.png")
    write_agreement_components_png(cells, grid_size, out_dir / "sf_energy_growth_agreement_components.png")
    write_hierarchy_canopy_png(cells, grid_size, out_dir / "sf_energy_growth_hierarchy_canopy.png")
    timeline_dir = out_dir / "timeline_frames"
    timeline_dir.mkdir(exist_ok=True)
    for old_frame in timeline_dir.glob("day_*.png"):
        old_frame.unlink()
    timeline_frames = []
    for day, image in frames:
        filename = f"day_{day:03d}.png"
        image.save(timeline_dir / filename)
        timeline_frames.append({"day": day, "src": f"timeline_frames/{filename}"})
    timeline = {
        "metrics": metrics,
        "frames": timeline_frames,
        "summary": {
            "avg_served_fraction": sum(m["served_fraction"] for m in metrics) / max(1, len(metrics)),
            "max_served_fraction": max((m["served_fraction"] for m in metrics), default=0.0),
            "max_sharing_day": max(metrics, key=lambda m: m["shared_kwh"])["day"] if metrics else 0,
            "max_commercial_day": max(metrics, key=lambda m: m["commercial_cells"])["day"] if metrics else 0,
            "max_largest_component_day": max(metrics, key=lambda m: m["largest_agreement_component"])["day"] if metrics else 0,
        },
    }
    (out_dir / "timeline_data.js").write_text(
        "window.SF_TIMELINE = " + json.dumps(timeline, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    with (out_dir / "sf_energy_growth_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[-1].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    (out_dir / "sf_energy_growth_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "sf_energy_growth_cells.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "row", "col", "lon", "lat", "building_count", "land", "occupied", "elevation_m", "slope",
            "solar_factor", "productivity", "money", "use_type", "floors", "pv_kw",
            "battery_kwh", "service_ratio", "lifetime_imported", "lifetime_exported",
            "lifetime_transit", "lifetime_transit_fee", "cable_degree", "function",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in cells:
            row = {}
            for field in fields:
                if field == "service_ratio":
                    row[field] = c.service_ratio
                elif field == "function":
                    row[field] = c.use_type if c.land and c.occupied else "empty"
                else:
                    row[field] = getattr(c, field)
            writer.writerow(row)
    occupied_count = sum(1 for c in cells if c.land and c.occupied)
    summary = {
        "grid_size": grid_size,
        "buildable_cells": sum(1 for c in cells if c.land),
        "occupied_cells": occupied_count,
        "empty_buildable_cells": sum(1 for c in cells if c.land) - occupied_count,
        "final": metrics[-1],
        "costs": costs,
        "sources": sources,
        "hierarchy_note": "Blocks are not predefined. Agreement components are connected components of paid adjacent transmission agreements.",
    }
    (out_dir / "sf_energy_growth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SF Terrain Energy Growth</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #eef2f7; color: #0f172a; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #475569; line-height: 1.5; }}
    .panel {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; margin-top: 18px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #cbd5e1; background: #f8fafc; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>SF terrain energy growth</h1>
    <p>One-rule version: the grid starts empty, residential startups choose buildable cells by solar/productivity opportunity, and every building has only one function, residential or commercial. PV, battery, and cable are purchased assets. If a building has surplus, it can serve a requester through adjacent paid transmission agreements. Agreement components are read afterward as emergent blocks; no block grid or block treasury is predefined.</p>
    <div class="panel"><strong>Final:</strong> service={metrics[-1]['mean_service_lifetime']:.3f}, PV={metrics[-1]['pv_kw']:.0f} kW, battery={metrics[-1]['battery_kwh']:.0f} kWh, agreements={metrics[-1]['cables']}, components={metrics[-1]['agreement_components']}, largest component={metrics[-1]['largest_agreement_component']}.</div>
    <div class="panel"><h2>Evolution</h2><img src="sf_energy_growth_evolution.gif" alt="evolution"></div>
    <div class="panel"><h2>Final Map</h2><img src="sf_energy_growth_final.png" alt="final map"></div>
    <div class="panel"><h2>Metrics</h2><img src="sf_energy_growth_metrics.png" alt="metrics"></div>
    <div class="panel"><h2>Agreement Components</h2><img src="sf_energy_growth_agreement_components.png" alt="agreement components"></div>
    <div class="panel"><h2>Hierarchy Canopy</h2><img src="sf_energy_growth_hierarchy_canopy.png" alt="hierarchy canopy"></div>
    <div class="panel">
      <p>Data files: <a href="sf_energy_growth_cells.csv">cells CSV</a>, <a href="sf_energy_growth_metrics.csv">metrics CSV</a>, <a href="sf_energy_growth_summary.json">summary JSON</a>.</p>
      <p>Sources: {sources.get('climate_source', '')}; {sources.get('elevation_source', '')}; local load profiles from <code>{sources.get('profile_source', '')}</code>.</p>
    </div>
  </main>
</body>
</html>
"""
    (out_dir / "sf_energy_growth_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="First-pass SF terrain energy-growth ABM.")
    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--steps", type=int, default=8760)
    parser.add_argument("--seed", type=int, default=20250624)
    parser.add_argument("--fetch-elevation", action="store_true")
    parser.add_argument("--fetch-climate", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=project_root() / "outputs" / "sf_terrain_energy_growth")
    args = parser.parse_args()

    profile_csv = data_root() / "energy_profiles_clean" / "energy_profiles_hourly_used.csv"
    metadata_csv = data_root() / "energy_profiles_clean" / "building_energy_metadata.csv"
    cache_dir = data_root() / "sf_terrain_energy_growth_cache"

    cells = build_cells(metadata_csv, args.grid_size, args.seed)
    terrain_sources = load_or_build_terrain(cells, args.grid_size, cache_dir, args.fetch_elevation)
    solar_cf, climate_sources = load_or_fetch_2025_solar(cache_dir, args.fetch_climate, args.steps)
    res_profile, com_profile, profile_sources = load_two_profiles(profile_csv, args.seed, args.steps)
    metrics, costs, frames = simulate(
        cells,
        args.grid_size,
        res_profile,
        com_profile,
        solar_cf,
        args.seed,
        args.steps,
        args.out_dir,
    )
    sources = {**terrain_sources, **climate_sources, **profile_sources}
    write_outputs(args.out_dir, cells, args.grid_size, metrics, frames, costs, sources)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "final": metrics[-1],
        "sources": sources,
    }, indent=2))


if __name__ == "__main__":
    main()
