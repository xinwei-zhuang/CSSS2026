from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


LAND_USES = [
    {
        "key": "R",
        "name": "residential",
        "color": (134, 190, 181),
        "demand_kind": "residential",
        "demand_scale": 0.88,
        "roof_kind": "residential",
        "storage_cap": 1.70,
    },
    {
        "key": "C",
        "name": "commercial",
        "color": (203, 153, 91),
        "demand_kind": "commercial",
        "demand_scale": 1.18,
        "roof_kind": "commercial",
        "storage_cap": 1.65,
    },
    {
        "key": "I",
        "name": "industrial",
        "color": (156, 169, 205),
        "demand_kind": "industrial",
        "demand_scale": 1.35,
        "roof_kind": "industrial",
        "storage_cap": 2.45,
    },
]


NORMS = [
    {
        "key": "SELF",
        "name": "selfish",
        "description": "Does not share.",
        "color": (120, 113, 108),
        "kind": "selfish",
        "assessment": (1, 1, 1, 1),
    },
    {
        "key": "GEN",
        "name": "generous",
        "description": "Shares whenever possible.",
        "color": (37, 99, 235),
        "kind": "generous",
        "assessment": (1, 0, 1, 0),
    },
]


def norm_index(key: str) -> int:
    target = key.upper()
    for idx, norm in enumerate(NORMS):
        if norm["key"] == target:
            return idx
    raise ValueError(f"Unknown norm key: {key}")


@dataclass
class Cell:
    landuse: int
    norm: int
    block_id: int
    alive: bool = True
    resilient: bool = True
    reputation: float = 0.62
    health: float = 1.0
    storage: float = 0.0
    storage_cap: float = 1.0
    demand: float = 0.0
    served: float = 0.0
    deficit: float = 0.0
    surplus: float = 0.0
    payoff: float = 0.0
    stress_memory: float = 0.0
    cumulative_demand: float = 0.0
    cumulative_deficit: float = 0.0
    demand_curve: list[float] | None = None
    solar_curve: list[float] | None = None


@dataclass
class DataBundle:
    demand_curves: dict[str, list[float]]
    solar_curves: dict[str, list[float]]
    outage_severity: list[float]
    outage_solar_factor: list[float]
    outage_grid_support: list[float]
    roof_area_m2: dict[str, float]
    cell_specs: list[dict[str, Any]] | None
    sources: dict[str, str]


@dataclass
class SimulationResult:
    frames: list[Image.Image]
    contact_frames: list[Image.Image]
    metrics: list[dict[str, Any]]
    final_cells: list[Cell]
    data: DataBundle
    config: dict[str, Any]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_mean(values: list[float]) -> list[float]:
    mean = sum(values) / max(1, len(values))
    if mean <= 0:
        return [1.0 for _ in values]
    return [max(0.02, value / mean) for value in values]


def hourly_to_daily(values: list[float]) -> list[float]:
    totals = [0.0] * 24
    counts = [0] * 24
    for idx, value in enumerate(values):
        hour = idx % 24
        totals[hour] += value
        counts[hour] += 1
    return normalize_mean([totals[i] / max(1, counts[i]) for i in range(24)])


def fallback_demand(kind: str) -> list[float]:
    if kind == "commercial":
        raw = [
            0.42,
            0.39,
            0.38,
            0.38,
            0.42,
            0.55,
            0.78,
            1.02,
            1.18,
            1.26,
            1.30,
            1.28,
            1.22,
            1.18,
            1.16,
            1.12,
            1.02,
            0.86,
            0.70,
            0.58,
            0.51,
            0.47,
            0.44,
            0.42,
        ]
    elif kind == "industrial":
        raw = [
            0.72,
            0.70,
            0.68,
            0.67,
            0.70,
            0.78,
            0.94,
            1.08,
            1.16,
            1.22,
            1.24,
            1.22,
            1.18,
            1.16,
            1.14,
            1.12,
            1.08,
            1.02,
            0.94,
            0.86,
            0.80,
            0.76,
            0.74,
            0.72,
        ]
    else:
        raw = [
            0.70,
            0.62,
            0.58,
            0.56,
            0.60,
            0.76,
            1.02,
            1.12,
            0.98,
            0.86,
            0.82,
            0.86,
            0.90,
            0.92,
            0.96,
            1.04,
            1.18,
            1.34,
            1.42,
            1.34,
            1.18,
            1.02,
            0.88,
            0.76,
        ]
    return normalize_mean(raw)


def load_demand_curves(path: str, sample_size: int, full_series: bool = False) -> tuple[dict[str, list[float]], dict[str, int]]:
    curves = {"residential": [], "commercial": []}
    csv_path = Path(path)
    if csv_path.exists():
        target_each = max(1, sample_size // 2)
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            time_cols = [name for name in reader.fieldnames or [] if name.startswith("t")]
            for row in reader:
                profile_type = (row.get("profile_type") or "").lower()
                if profile_type not in curves or len(curves[profile_type]) >= target_each:
                    continue
                values = [safe_float(row.get(col), 0.0) for col in time_cols]
                curves[profile_type].append(normalize_mean(values) if full_series else hourly_to_daily(values))
                if all(len(values_) >= target_each for values_ in curves.values()):
                    break
    out = {}
    counts = {}
    target_len = 0
    for values in curves.values():
        if values:
            target_len = max(target_len, min(len(curve) for curve in values))
    if target_len <= 0:
        target_len = 8760 if full_series else 24
    for kind in curves:
        counts[kind] = len(curves[kind])
        if curves[kind]:
            out[kind] = normalize_mean(
                [
                    sum(curve[hour] for curve in curves[kind]) / len(curves[kind])
                    for hour in range(target_len)
                ]
            )
        else:
            daily = fallback_demand(kind)
            out[kind] = [daily[i % 24] for i in range(target_len)]
    industrial_daily = fallback_demand("industrial")
    out["industrial"] = [industrial_daily[i % 24] for i in range(target_len)]
    counts["industrial"] = 0
    return out, counts


def estimate_roof_area(path: str, sample_size: int, usable_fraction: float) -> dict[str, float]:
    fallback = {"residential": 125.0, "commercial": 620.0, "industrial": 980.0}
    csv_path = Path(path)
    if not csv_path.exists():
        return fallback
    sums = {"residential": 0.0, "commercial": 0.0}
    counts = {"residential": 0, "commercial": 0}
    target_each = max(1, sample_size // 2)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = (row.get("profile_type") or "").lower()
            if kind not in sums or counts[kind] >= target_each:
                continue
            sqft = safe_float(row.get("bldgsqft"), 0.0)
            floors = max(1.0, safe_float(row.get("floor"), 1.0))
            if sqft <= 0:
                continue
            area = sqft / floors * 0.092903 * usable_fraction
            if area <= 0:
                continue
            sums[kind] += area
            counts[kind] += 1
            if counts["residential"] >= target_each and counts["commercial"] >= target_each:
                break
    out = {
        kind: sums[kind] / counts[kind] if counts[kind] else fallback[kind]
        for kind in sums
    }
    out["industrial"] = max(fallback["industrial"], out["commercial"] * 1.35)
    return out


def landuse_index(profile_type: str, landuse_final: str) -> int:
    text = f"{profile_type} {landuse_final}".lower()
    if "industrial" in text or "pdr" in text or "mips" in text:
        return 2
    if "commercial" in text or "retail" in text or "office" in text or "hotel" in text or "mixed" in text:
        return 1
    return 0


def roof_area_from_row(row: dict[str, str], usable_fraction: float) -> float:
    sqft = safe_float(row.get("bldgsqft"), 0.0)
    floors = max(1.0, safe_float(row.get("floor"), 1.0))
    if sqft <= 0:
        return 125.0
    return max(25.0, sqft / floors * 0.092903 * usable_fraction)


def load_profile_curves(path: str, profile_ids: set[str]) -> dict[str, list[float]]:
    csv_path = Path(path)
    if not csv_path.exists() or not profile_ids:
        return {}
    curves: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        time_cols = [name for name in reader.fieldnames or [] if name.startswith("t")]
        for row in reader:
            profile_id = row.get("profile_id", "")
            if profile_id not in profile_ids:
                continue
            curves[profile_id] = [safe_float(row.get(col), 0.0) for col in time_cols]
            if len(curves) >= len(profile_ids):
                break
    return curves


def select_real_building_patch(
    metadata_csv: str,
    demand_csv: str,
    grid_size: int,
    usable_fraction: float,
    corner_fraction: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        return [], {"real_patch_source": f"missing:{metadata_csv}"}

    rows: list[dict[str, Any]] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lon = safe_float(row.get("centroid_lon"), float("nan"))
            lat = safe_float(row.get("centroid_lat"), float("nan"))
            if not math.isfinite(lon) or not math.isfinite(lat):
                continue
            if not (-123.0 < lon < -122.0 and 37.0 < lat < 38.5):
                continue
            rows.append(
                {
                    "row": row,
                    "lon": lon,
                    "lat": lat,
                    "profile_id": row.get("profile_id", ""),
                    "landuse": landuse_index(row.get("profile_type", ""), row.get("landuse_final", "")),
                    "roof_area_m2": roof_area_from_row(row, usable_fraction),
                }
            )
    if not rows:
        return [], {"real_patch_source": str(metadata_path), "real_patch_count": "0"}

    lon_min = min(item["lon"] for item in rows)
    lon_max = max(item["lon"] for item in rows)
    lat_min = min(item["lat"] for item in rows)
    lat_max = max(item["lat"] for item in rows)
    target_count = grid_size * grid_size

    selected_candidates: list[dict[str, Any]] = []
    used_fraction = corner_fraction
    for fraction in [corner_fraction, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        lon0 = lon_max - (lon_max - lon_min) * fraction
        lat0 = lat_max - (lat_max - lat_min) * fraction
        selected_candidates = [
            item for item in rows
            if lon0 <= item["lon"] <= lon_max and lat0 <= item["lat"] <= lat_max
        ]
        used_fraction = fraction
        if len(selected_candidates) >= target_count:
            break
    if len(selected_candidates) < target_count:
        selected_candidates = sorted(rows, key=lambda item: (-(item["lon"] - lon_min), -(item["lat"] - lat_min)))[:target_count]

    patch_lon_min = min(item["lon"] for item in selected_candidates)
    patch_lon_max = max(item["lon"] for item in selected_candidates)
    patch_lat_min = min(item["lat"] for item in selected_candidates)
    patch_lat_max = max(item["lat"] for item in selected_candidates)
    profile_ids = {item["profile_id"] for item in selected_candidates if item["profile_id"]}
    profile_curves = load_profile_curves(demand_csv, profile_ids)

    chosen: list[dict[str, Any]] = []
    unused = selected_candidates[:]
    for row in range(grid_size):
        for col in range(grid_size):
            x = patch_lon_min + (patch_lon_max - patch_lon_min) * ((col + 0.5) / grid_size)
            y = patch_lat_max - (patch_lat_max - patch_lat_min) * ((row + 0.5) / grid_size)
            best_idx = min(
                range(len(unused)),
                key=lambda idx: (unused[idx]["lon"] - x) ** 2 + (unused[idx]["lat"] - y) ** 2,
            )
            chosen.append(unused.pop(best_idx))
            if not unused:
                unused = selected_candidates[:]

    raw_curves: list[list[float]] = []
    for item in chosen:
        curve = profile_curves.get(item["profile_id"])
        if not curve:
            daily = fallback_demand(LAND_USES[item["landuse"]]["demand_kind"])
            curve = [daily[i % 24] for i in range(8760)]
        raw_curves.append(curve)
    target_len = min(len(curve) for curve in raw_curves)
    demand_reference = sum(sum(curve[:target_len]) / target_len for curve in raw_curves) / len(raw_curves)
    if demand_reference <= 0:
        demand_reference = 1.0

    specs = []
    for item, curve in zip(chosen, raw_curves):
        specs.append(
            {
                "building_id": item["row"].get("building_id", ""),
                "profile_id": item["profile_id"],
                "lon": item["lon"],
                "lat": item["lat"],
                "landuse": item["landuse"],
                "roof_area_m2": item["roof_area_m2"],
                "demand_curve": [max(0.0, value / demand_reference) for value in curve[:target_len]],
            }
        )
    sources = {
        "real_patch_source": str(metadata_path),
        "real_patch_corner": "upper_right",
        "real_patch_corner_fraction": f"{used_fraction:.2f}",
        "real_patch_candidates": str(len(selected_candidates)),
        "real_patch_cells": str(len(specs)),
        "real_patch_bbox": json.dumps({
            "lon_min": patch_lon_min,
            "lon_max": patch_lon_max,
            "lat_min": patch_lat_min,
            "lat_max": patch_lat_max,
        }),
    }
    return specs, sources


def read_epw_24h(path: str, month: int, day: int) -> tuple[list[float], list[float], str]:
    epw_path = Path(path)
    if not epw_path.exists():
        solar = [0, 0, 0, 0, 0, 0.05, 0.22, 0.48, 0.74, 0.94, 1.08, 1.14, 1.12, 1.02, 0.86, 0.66, 0.42, 0.18, 0.04, 0, 0, 0, 0, 0]
        return solar, [15.0] * 24, "synthetic"
    ghi = [[] for _ in range(24)]
    temp = [[] for _ in range(24)]
    with epw_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for _ in range(8):
            next(reader, None)
        for row in reader:
            if len(row) < 16:
                continue
            if int(safe_float(row[1], 0)) != month or int(safe_float(row[2], 0)) != day:
                continue
            hour = max(0, min(23, int(safe_float(row[3], 1)) - 1))
            temp[hour].append(safe_float(row[6], 15.0))
            ghi[hour].append(max(0.0, safe_float(row[13], 0.0)))
    return (
        [sum(v) / len(v) if v else 0.0 for v in ghi],
        [sum(v) / len(v) if v else 15.0 for v in temp],
        str(epw_path),
    )


def read_epw_series(path: str) -> tuple[list[float], list[float], str]:
    epw_path = Path(path)
    if not epw_path.exists():
        solar, temp, source = read_epw_24h(path, 6, 13)
        return [solar[i % 24] for i in range(8760)], [temp[i % 24] for i in range(8760)], source
    ghi = []
    temp = []
    with epw_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for _ in range(8):
            next(reader, None)
        for row in reader:
            if len(row) < 16:
                continue
            temp.append(safe_float(row[6], 15.0))
            ghi.append(max(0.0, safe_float(row[13], 0.0)))
    if not ghi:
        solar, temp_daily, source = read_epw_24h(path, 6, 13)
        return [solar[i % 24] for i in range(8760)], [temp_daily[i % 24] for i in range(8760)], source
    return ghi, temp, str(epw_path)


def pv_per_m2_kw(ghi: list[float], temp_air: list[float], efficiency: float) -> list[float]:
    tref = 25.0
    c1, c2, c3 = -3.75, 1.14, 0.0175
    rou = 0.0045
    sigma = 0.1
    out = []
    for rt, temp in zip(ghi, temp_air):
        if rt <= 0:
            out.append(0.0)
            continue
        temp_factor = 1 - rou * (c1 + c2 * temp + c3 * rt - tref)
        irradiance_factor = sigma * math.log10(max(rt, 1e-6))
        out.append(max(0.0, rt * efficiency * (temp_factor + irradiance_factor) / 1000.0))
    return out


def load_outage_profile(path: str | None, target_len: int) -> tuple[list[float], list[float], list[float], str]:
    if not path:
        return [0.0] * target_len, [1.0] * target_len, [0.0] * target_len, ""
    csv_path = Path(path)
    if not csv_path.exists():
        return [0.0] * target_len, [1.0] * target_len, [0.0] * target_len, f"missing:{path}"

    severities: list[float] = []
    solar_factors: list[float] = []
    grid_support: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            severity = safe_float(
                row.get("outage_severity", row.get("severity", row.get("outage_fraction", row.get("stress", "0")))),
                0.0,
            )
            severity = max(0.0, min(1.0, severity))
            solar_factor = safe_float(row.get("solar_factor", ""), 1.0 - 0.65 * severity)
            grid = safe_float(row.get("grid_support", ""), 0.0)
            severities.append(severity)
            solar_factors.append(max(0.0, min(1.0, solar_factor)))
            grid_support.append(max(0.0, min(1.0, grid)))

    if not severities:
        return [0.0] * target_len, [1.0] * target_len, [0.0] * target_len, str(csv_path)

    def resize(values: list[float], fallback: float) -> list[float]:
        if len(values) >= target_len:
            return values[:target_len]
        return [values[i % len(values)] if values else fallback for i in range(target_len)]

    return (
        resize(severities, 0.0),
        resize(solar_factors, 1.0),
        resize(grid_support, 0.0),
        str(csv_path),
    )


def load_data(config: dict[str, Any]) -> DataBundle:
    full_series = bool(config.get("use_full_year_profiles", False))
    demand_curves, profile_counts = load_demand_curves(
        config["demand_csv"],
        int(config["profile_sample_size"]),
        full_series,
    )
    roof = estimate_roof_area(
        config["metadata_csv"],
        int(config["profile_sample_size"]),
        float(config["roof_usable_fraction"]),
    )
    if bool(config.get("use_full_year_solar", full_series)):
        ghi, temp, solar_source = read_epw_series(config["solar_epw"])
    else:
        ghi, temp, solar_source = read_epw_24h(
            config["solar_epw"],
            int(config["solar_month"]),
            int(config["solar_day"]),
        )
    pv = pv_per_m2_kw(ghi, temp, float(config["pv_efficiency"]))
    raw_solar = {kind: [v * roof[kind] for v in pv] for kind in roof}
    reference = sum(sum(v) / max(1, len(v)) for v in raw_solar.values()) / len(raw_solar)
    if reference <= 0:
        reference = 1.0
    solar_curves = {
        kind: [value / reference for value in values]
        for kind, values in raw_solar.items()
    }
    cell_specs = None
    real_patch_sources: dict[str, str] = {}
    if bool(config.get("use_real_building_patch", False)):
        cell_specs, real_patch_sources = select_real_building_patch(
            config["metadata_csv"],
            config["demand_csv"],
            int(config["grid_size"]),
            float(config["roof_usable_fraction"]),
            float(config.get("real_patch_corner_fraction", 0.25)),
        )
        if cell_specs:
            avg_roof = sum(spec["roof_area_m2"] for spec in cell_specs) / len(cell_specs)
            if avg_roof <= 0:
                avg_roof = 1.0
            solar_curves_by_area = {
                kind: [value / max(roof[kind], 1e-6) for value in values]
                for kind, values in raw_solar.items()
            }
            for spec in cell_specs:
                landuse = LAND_USES[int(spec["landuse"])]["roof_kind"]
                per_m2 = solar_curves_by_area.get(landuse, solar_curves_by_area["residential"])
                spec["solar_curve"] = [
                    value * spec["roof_area_m2"] / avg_roof
                    for value in per_m2
                ]
    target_len = max(
        len(next(iter(demand_curves.values()))),
        len(next(iter(solar_curves.values()))),
    )
    if cell_specs:
        target_len = max(
            target_len,
            len(cell_specs[0]["demand_curve"]),
            len(cell_specs[0]["solar_curve"]),
        )
    outage_severity, outage_solar_factor, outage_grid_support, outage_source = load_outage_profile(
        config.get("outage_profile_csv"),
        target_len,
    )
    return DataBundle(
        demand_curves=demand_curves,
        solar_curves=solar_curves,
        outage_severity=outage_severity,
        outage_solar_factor=outage_solar_factor,
        outage_grid_support=outage_grid_support,
        roof_area_m2=roof,
        cell_specs=cell_specs,
        sources={
            "demand_csv": config["demand_csv"],
            "metadata_csv": config["metadata_csv"],
            "solar_epw": solar_source,
            "outage_profile_csv": outage_source,
            **real_patch_sources,
            "profiles": json.dumps(profile_counts),
            "demand_series_length": str(len(next(iter(demand_curves.values())))),
            "solar_series_length": str(len(next(iter(solar_curves.values())))),
            "outage_series_length": str(len(outage_severity)),
        },
    )


class LandUseNormSimulation:
    def __init__(self, config: dict[str, Any], data: DataBundle) -> None:
        self.config = config
        self.data = data
        self.size = int(config["grid_size"])
        self.rng = random.Random(int(config["seed"]))
        fixed_norm_key = config.get("fixed_norm_key")
        self.fixed_norm = norm_index(fixed_norm_key) if fixed_norm_key else None
        self.enable_norm_evolution = bool(config.get("enable_norm_evolution", self.fixed_norm is None))
        self.enable_hierarchy = bool(config.get("enable_hierarchy", True))
        self.enable_rebuild = bool(config.get("enable_rebuild", True))
        self.block_size = int(config.get("block_size", 6))
        self.block_rows = math.ceil(self.size / self.block_size)
        self.block_cols = math.ceil(self.size / self.block_size)
        self.block_count = self.block_rows * self.block_cols
        self.block_norms = [-1 for _ in range(self.block_count)]
        self.block_strength = [0.0 for _ in range(self.block_count)]
        self.block_age = [0 for _ in range(self.block_count)]
        self.cells: list[Cell] = []
        self._neighbor_cache: dict[tuple[int, int], list[int]] = {}
        self.landuse_map = self._make_landuse_map()
        self._init_cells()

    def block_id_for(self, row: int, col: int) -> int:
        return (row // self.block_size) * self.block_cols + (col // self.block_size)

    def _make_landuse_map(self) -> list[int]:
        if self.data.cell_specs:
            return [int(spec["landuse"]) for spec in self.data.cell_specs]
        anchors = [
            (0.27, 0.72, 0),
            (0.72, 0.25, 1),
            (0.58, 0.62, 2),
        ]
        landuse = []
        for row in range(self.size):
            for col in range(self.size):
                scores = []
                for ar, ac, kind in anchors:
                    rr = ar * self.size
                    cc = ac * self.size
                    dist = math.hypot(row - rr, col - cc)
                    scores.append((dist + self.rng.uniform(-4.0, 4.0), kind))
                landuse.append(min(scores)[1])
        for _ in range(3):
            new_map = landuse[:]
            for idx in range(len(landuse)):
                counts = [0] * len(LAND_USES)
                for nidx in self.neighbor_indices(idx, radius=1):
                    counts[landuse[nidx]] += 1
                counts[landuse[idx]] += 2
                new_map[idx] = max(range(len(counts)), key=lambda k: counts[k])
            landuse = new_map
        return landuse

    def _init_cells(self) -> None:
        self.cells = []
        for idx, landuse in enumerate(self.landuse_map):
            row, col = divmod(idx, self.size)
            norm = self.initial_norm_for_landuse(landuse)
            spec = LAND_USES[landuse]
            cell_spec = self.data.cell_specs[idx] if self.data.cell_specs and idx < len(self.data.cell_specs) else None
            roof_factor = 1.0
            if cell_spec:
                avg_roof = sum(float(item["roof_area_m2"]) for item in self.data.cell_specs or [cell_spec]) / max(1, len(self.data.cell_specs or [cell_spec]))
                roof_factor = float(cell_spec["roof_area_m2"]) / max(1e-6, avg_roof)
            storage_cap = (
                spec["storage_cap"]
                * float(self.config.get("storage_capacity_multiplier", 1.0))
                * (0.78 + 0.44 * roof_factor)
                * self.rng.uniform(0.90, 1.10)
            )
            cell = Cell(
                landuse=landuse,
                norm=norm,
                block_id=self.block_id_for(row, col),
                reputation=self.rng.uniform(0.48, 0.82),
                health=self.rng.uniform(0.88, 1.0),
                storage_cap=storage_cap,
                storage=storage_cap * self.rng.uniform(0.38, 0.72),
                demand_curve=cell_spec.get("demand_curve") if cell_spec else None,
                solar_curve=cell_spec.get("solar_curve") if cell_spec else None,
            )
            self.cells.append(cell)

    def initial_norm_for_landuse(self, landuse: int) -> int:
        if self.fixed_norm is not None:
            return self.fixed_norm
        return self.rng.randrange(len(NORMS))

    def neighbor_indices(self, idx: int, radius: int) -> list[int]:
        cache_key = (idx, radius)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        row, col = divmod(idx, self.size)
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr = row + dr
                cc = col + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    out.append(rr * self.size + cc)
        self._neighbor_cache[cache_key] = out
        return out

    def block_norm(self, cell: Cell) -> int:
        return self.block_norms[cell.block_id]

    def landuse_demand(self, landuse: int, step: int) -> float:
        spec = LAND_USES[landuse]
        curve = self.data.demand_curves[spec["demand_kind"]]
        return curve[step % len(curve)] * spec["demand_scale"]

    def landuse_solar(self, landuse: int, hour: int, step: int) -> float:
        spec = LAND_USES[landuse]
        curve = self.data.solar_curves[spec["roof_kind"]]
        shock = 1.0
        if int(self.config["shock_start"]) <= step < int(self.config["shock_end"]):
            shock = float(self.config["solar_shock_factor"])
        outage_factor = self.data.outage_solar_factor[step % len(self.data.outage_solar_factor)]
        return curve[step % len(curve)] * shock * outage_factor * float(self.config.get("solar_generation_multiplier", 1.0))

    def cell_demand(self, cell: Cell, step: int) -> float:
        if cell.demand_curve:
            return cell.demand_curve[step % len(cell.demand_curve)] * LAND_USES[cell.landuse]["demand_scale"]
        return self.landuse_demand(cell.landuse, step)

    def cell_solar(self, cell: Cell, hour: int, step: int) -> float:
        if cell.solar_curve:
            shock = 1.0
            if int(self.config["shock_start"]) <= step < int(self.config["shock_end"]):
                shock = float(self.config["solar_shock_factor"])
            outage_factor = self.data.outage_solar_factor[step % len(self.data.outage_solar_factor)]
            return cell.solar_curve[step % len(cell.solar_curve)] * shock * outage_factor * float(self.config.get("solar_generation_multiplier", 1.0))
        return self.landuse_solar(cell.landuse, hour, step)

    def should_share(self, donor: Cell, receiver: Cell, distance: float) -> bool:
        norm = NORMS[donor.norm]["kind"]
        block_norm = self.block_norm(donor)
        institutional_bonus = block_norm == donor.norm and self.block_strength[donor.block_id] > 0.35
        threshold_shift = 0.10 if institutional_bonus else 0.0
        if institutional_bonus and receiver.block_id == donor.block_id:
            return donor.storage > donor.storage_cap * 0.22
        if norm == "generous":
            return donor.storage > donor.storage_cap * (0.20 - threshold_shift)
        if norm == "selfish":
            return False
        return False

    def pool_link_allowed(self, a: Cell, b: Cell, distance: float) -> bool:
        if not bool(self.config.get("enable_shared_storage_pool", False)):
            return False
        norm = NORMS[a.norm]["kind"]
        if norm == "generous":
            return True
        if norm == "selfish":
            return False
        return False

    def norm_transfer_fraction(self, donor: Cell, receiver: Cell, distance: float) -> float:
        norm = NORMS[donor.norm]["kind"]
        if norm == "generous":
            return 1.00
        if norm == "selfish":
            return 0.0
        return 0.0

    def sharing_efficiency(self, distance: float) -> float:
        radius = max(1e-6, float(self.config.get("share_radius", 1)))
        minimum = float(self.config.get("sharing_min_efficiency", 0.35))
        decay = float(self.config.get("sharing_efficiency_decay", 0.65))
        exponent = float(self.config.get("sharing_loss_exponent", 1.0))
        normalized = min(1.0, max(0.0, distance / radius))
        efficiency = 1.0 - decay * (normalized ** exponent)
        return max(minimum, min(1.0, efficiency))

    def shared_storage_step(self, receivers: list[int], radius: int) -> dict[str, float]:
        cooperation_attempts = 0
        cooperation_successes = 0
        pool_count = 0
        pool_members = 0

        for ridx in receivers:
            receiver = self.cells[ridx]
            if receiver.deficit <= 0.01:
                continue
            rr, rc = divmod(ridx, self.size)
            candidates = []
            for didx in self.neighbor_indices(ridx, radius):
                donor = self.cells[didx]
                if not donor.alive or donor.surplus <= 0.01:
                    continue
                dr, dc = divmod(didx, self.size)
                dist = math.hypot(rr - dr, rc - dc)
                if self.pool_link_allowed(donor, receiver, dist):
                    candidates.append((dist, didx))
            if not candidates:
                continue
            pool_count += 1
            candidates.sort(key=lambda item: (-self.cells[item[1]].surplus * self.sharing_efficiency(item[0]), item[0]))
            max_donors = int(self.config.get("max_pool_donors_per_receiver", 24))
            pool_members += 1 + min(len(candidates), max_donors)
            for dist, didx in candidates[:max_donors]:
                if receiver.deficit <= 0.01:
                    break
                donor = self.cells[didx]
                cooperation_attempts += 1
                action = self.should_share(donor, receiver, dist)
                donor.reputation = self.assess(donor, receiver, action)
                if not action:
                    donor.payoff -= 0.006
                    continue
                efficiency = self.sharing_efficiency(dist)
                transfer_cap = donor.surplus * self.norm_transfer_fraction(donor, receiver, dist)
                transfer = min(transfer_cap, receiver.deficit / max(0.1, efficiency))
                if transfer <= 0.0:
                    continue
                received = transfer * efficiency
                donor.storage -= transfer
                donor.surplus = max(0.0, donor.storage - donor.storage_cap * 0.44)
                receiver.deficit = max(0.0, receiver.deficit - received)
                receiver.served += received
                donor.payoff += 0.052 - 0.014 * transfer
                receiver.payoff += 0.07 * received / max(receiver.demand, 1e-6)
                cooperation_successes += 1

        return {
            "cooperation_attempts": cooperation_attempts,
            "cooperation_successes": cooperation_successes,
            "pool_count": pool_count,
            "pool_members": pool_members,
        }

    def assess(self, donor: Cell, receiver: Cell, action: bool) -> float:
        receiver_good = receiver.reputation >= 0.5
        table = NORMS[donor.norm]["assessment"]
        if action and receiver_good:
            good = table[0]
        elif not action and receiver_good:
            good = table[1]
        elif action and not receiver_good:
            good = table[2]
        else:
            good = table[3]
        return min(1.0, donor.reputation + 0.08) if good else max(0.0, donor.reputation - 0.16)

    def energy_step(self, step: int) -> dict[str, float]:
        hour = step % 24
        outage_severity = self.data.outage_severity[step % len(self.data.outage_severity)]
        outage_profile_grid = self.data.outage_grid_support[step % len(self.data.outage_grid_support)]
        in_shock = int(self.config["shock_start"]) <= step < int(self.config["shock_end"])
        grid_support = (
            float(self.config["outage_grid_support"])
            if in_shock
            else float(self.config["normal_grid_support"])
        )
        if outage_severity > 0.0:
            grid_support = outage_profile_grid
        for cell in self.cells:
            cell.payoff *= 0.45
            if not cell.alive:
                cell.demand = 0.0
                cell.served = 0.0
                cell.deficit = 0.0
                cell.surplus = 0.0
                continue
            demand = self.cell_demand(cell, step)
            solar = self.cell_solar(cell, hour, step)
            available = cell.storage + solar + demand * grid_support
            served = min(demand, available)
            cell.demand = demand
            cell.served = served
            cell.storage = min(cell.storage_cap, max(0.0, available - demand))
            cell.deficit = max(0.0, demand - available)
            cell.surplus = max(0.0, cell.storage - cell.storage_cap * 0.44)

        cooperation_attempts = 0
        cooperation_successes = 0
        receivers = [
            idx
            for idx, cell in enumerate(self.cells)
            if cell.alive and cell.deficit > 0.03
        ]
        receivers.sort(
            key=lambda idx: (
                -self.cells[idx].deficit,
                -self.cells[idx].reputation,
            )
        )
        radius = int(self.config["share_radius"])
        pool_count = 0
        pool_members = 0
        if bool(self.config.get("enable_shared_storage_pool", False)):
            pool_stats = self.shared_storage_step(receivers, radius)
            cooperation_attempts = int(pool_stats["cooperation_attempts"])
            cooperation_successes = int(pool_stats["cooperation_successes"])
            pool_count = int(pool_stats["pool_count"])
            pool_members = int(pool_stats["pool_members"])
        else:
            for ridx in receivers:
                receiver = self.cells[ridx]
                candidates = []
                rr, rc = divmod(ridx, self.size)
                for didx in self.neighbor_indices(ridx, radius):
                    donor = self.cells[didx]
                    if not donor.alive or donor.surplus <= 0.03:
                        continue
                    dr, dc = divmod(didx, self.size)
                    dist = math.hypot(rr - dr, rc - dc)
                    candidates.append((dist, didx))
                candidates.sort(key=lambda item: (-self.cells[item[1]].surplus * self.sharing_efficiency(item[0]), item[0]))
                max_donors = int(self.config.get("max_direct_donors_per_receiver", 12))
                for dist, didx in candidates[:max_donors]:
                    if receiver.deficit <= 0.01:
                        break
                    donor = self.cells[didx]
                    cooperation_attempts += 1
                    action = self.should_share(donor, receiver, dist)
                    donor.reputation = self.assess(donor, receiver, action)
                    if not action:
                        donor.payoff -= 0.006
                        continue
                    same_block_institution = (
                        donor.block_id == receiver.block_id
                        and self.block_norms[donor.block_id] == donor.norm
                        and self.block_strength[donor.block_id] > 0.35
                    )
                    efficiency = 0.98 if same_block_institution else self.sharing_efficiency(dist)
                    transfer_cap = donor.surplus * self.norm_transfer_fraction(donor, receiver, dist)
                    transfer = min(transfer_cap, receiver.deficit / max(0.1, efficiency))
                    if transfer <= 0.0:
                        continue
                    received = transfer * efficiency
                    donor.storage -= transfer
                    donor.surplus = max(0.0, donor.storage - donor.storage_cap * 0.44)
                    receiver.deficit = max(0.0, receiver.deficit - received)
                    receiver.served += received
                    donor.payoff += 0.052 - 0.014 * transfer
                    if same_block_institution:
                        donor.payoff += 0.020
                    receiver.payoff += 0.07 * received / max(receiver.demand, 1e-6)
                    cooperation_successes += 1

        active_cells = [cell for cell in self.cells if cell.alive]
        total_demand = sum(cell.demand for cell in active_cells)
        total_served = sum(min(cell.served, cell.demand) for cell in active_cells)
        resilient_threshold = float(self.config.get("resilient_deficit_threshold", 0.05))

        for cell in self.cells:
            if not cell.alive:
                cell.resilient = False
                continue
            unmet = cell.deficit / max(cell.demand, 1e-6)
            cell.cumulative_demand += cell.demand
            cell.cumulative_deficit += max(0.0, cell.deficit)
            cumulative_unmet = cell.cumulative_deficit / max(cell.cumulative_demand, 1e-6)
            cell.resilient = cumulative_unmet <= resilient_threshold
            institution = self.block_norm(cell)
            aligned = institution == cell.norm and institution >= 0
            recovery_rate = float(self.config.get("health_recovery_rate", 0.044))
            loss_rate = float(self.config.get("health_loss_rate", 0.046))
            memory_retention = float(self.config.get("stress_memory_retention", 0.96))
            cell.stress_memory = max(0.0, min(1.0, memory_retention * cell.stress_memory + (1.0 - memory_retention) * unmet))
            cell.health = max(0.0, min(1.0, cell.health + recovery_rate * (1.0 - unmet) - loss_rate * unmet))
            cell.payoff += 0.08 * (1.0 - unmet) + 0.035 * cell.reputation
            if aligned:
                cell.payoff += 0.035 * self.block_strength[cell.block_id]
            if cell.health <= 0.035:
                cell.alive = False
                cell.resilient = False
                cell.reputation = 0.25

        return {
            "served_fraction": min(1.0, total_served / max(total_demand, 1e-6)),
            "cooperation_rate": cooperation_successes / max(1, cooperation_attempts),
            "cooperation_attempts": cooperation_attempts,
            "cooperation_successes": cooperation_successes,
            "pool_count": pool_count,
            "pool_members": pool_members,
            "outage_severity": outage_severity,
            "outage_solar_factor": self.data.outage_solar_factor[step % len(self.data.outage_solar_factor)],
            "grid_support": grid_support,
        }

    def rebuild_norm(self, idx: int, live_neighbors: list[int]) -> int:
        if self.fixed_norm is not None:
            return self.fixed_norm
        block_norm = self.block_norms[self.cells[idx].block_id]
        if block_norm >= 0 and self.rng.random() < float(self.config.get("hierarchy_rebuild_bias", 0.55)):
            return block_norm
        return self.cells[self.rng.choice(live_neighbors)].norm

    def evolution_step(self) -> dict[str, int]:
        imitation_rate = float(self.config["imitation_rate"])
        mutation_rate = float(self.config["mutation_rate"])
        rebuild_rate = float(self.config["rebuild_rate"])
        conformity_rate = float(self.config.get("hierarchy_conformity_rate", 0.025))
        new_norms = [cell.norm for cell in self.cells]
        mutations = 0
        imitations = 0
        rebuilds = 0
        hierarchy_adoptions = 0

        if not self.enable_norm_evolution and not self.enable_rebuild:
            if self.fixed_norm is not None:
                for cell in self.cells:
                    cell.norm = self.fixed_norm
            return {
                "norm_mutations": 0,
                "norm_imitations": 0,
                "building_rebuilds": 0,
                "hierarchy_adoptions": 0,
            }

        for idx, cell in enumerate(self.cells):
            if not cell.alive:
                live_neighbors = [n for n in self.neighbor_indices(idx, 1) if self.cells[n].alive]
                if self.enable_rebuild and live_neighbors and self.rng.random() < rebuild_rate:
                    cell.alive = True
                    cell.norm = self.rebuild_norm(idx, live_neighbors)
                    cell.health = 0.45
                    cell.storage = cell.storage_cap * 0.30
                    cell.reputation = 0.50
                    rebuilds += 1
                continue
            if not self.enable_norm_evolution:
                if self.fixed_norm is not None:
                    new_norms[idx] = self.fixed_norm
                continue
            block_norm = self.block_norm(cell)
            if (
                self.enable_hierarchy
                and block_norm >= 0
                and cell.norm != block_norm
                and self.rng.random() < conformity_rate * self.block_strength[cell.block_id]
            ):
                new_norms[idx] = block_norm
                hierarchy_adoptions += 1
                continue
            if self.rng.random() < mutation_rate:
                new_norms[idx] = self.rng.randrange(len(NORMS))
                mutations += 1
                continue
            if self.rng.random() > imitation_rate:
                continue
            neighbors = [n for n in self.neighbor_indices(idx, 1) if self.cells[n].alive]
            if not neighbors:
                continue
            model_idx = max(
                neighbors,
                key=lambda n: (
                    self.cells[n].payoff
                    + 0.30 * self.cells[n].health
                    + 0.12 * self.cells[n].reputation
                    + (0.20 * self.block_strength[self.cells[n].block_id] if self.block_norm(self.cells[n]) == self.cells[n].norm else 0.0)
                ),
            )
            model = self.cells[model_idx]
            own_score = cell.payoff + 0.30 * cell.health + 0.12 * cell.reputation
            model_score = model.payoff + 0.30 * model.health + 0.12 * model.reputation
            if model_score > own_score + self.rng.uniform(0.01, 0.12):
                new_norms[idx] = model.norm
                imitations += 1
        for idx, norm in enumerate(new_norms):
            self.cells[idx].norm = norm
        return {
            "norm_mutations": mutations,
            "norm_imitations": imitations,
            "building_rebuilds": rebuilds,
            "hierarchy_adoptions": hierarchy_adoptions,
        }

    def hierarchy_step(self) -> dict[str, int]:
        if not self.enable_hierarchy:
            self.block_norms = [-1 for _ in range(self.block_count)]
            self.block_strength = [0.0 for _ in range(self.block_count)]
            self.block_age = [0 for _ in range(self.block_count)]
            return {
                "hierarchy_births": 0,
                "hierarchy_switches": 0,
                "hierarchy_dissolutions": 0,
            }
        threshold = float(self.config.get("hierarchy_threshold", 0.58))
        service_threshold = float(self.config.get("hierarchy_service_threshold", 0.72))
        changes = 0
        births = 0
        dissolutions = 0
        for block_id in range(self.block_count):
            members = [cell for cell in self.cells if cell.alive and cell.block_id == block_id]
            if len(members) < max(3, int(self.block_size * self.block_size * 0.25)):
                if self.block_norms[block_id] >= 0:
                    self.block_strength[block_id] *= 0.82
                    if self.block_strength[block_id] < 0.18:
                        self.block_norms[block_id] = -1
                        self.block_age[block_id] = 0
                        dissolutions += 1
                continue
            counts = [0] * len(NORMS)
            service_by_norm = [0.0] * len(NORMS)
            for cell in members:
                counts[cell.norm] += 1
                service_by_norm[cell.norm] += cell.served / max(cell.demand, 1e-6)
            winner = max(range(len(NORMS)), key=lambda idx: counts[idx])
            winner_freq = counts[winner] / len(members)
            winner_service = service_by_norm[winner] / max(1, counts[winner])
            if winner_freq >= threshold and winner_service >= service_threshold:
                target_strength = min(1.0, 0.25 + winner_freq * 0.65 + max(0.0, winner_service - service_threshold) * 0.35)
                if self.block_norms[block_id] == winner:
                    self.block_strength[block_id] = min(1.0, self.block_strength[block_id] * 0.88 + target_strength * 0.18)
                    self.block_age[block_id] += 1
                else:
                    previous = self.block_norms[block_id]
                    switch_margin = 0.10 if previous >= 0 else 0.0
                    previous_freq = counts[previous] / len(members) if previous >= 0 else 0.0
                    if previous < 0 or winner_freq > previous_freq + switch_margin:
                        self.block_norms[block_id] = winner
                        self.block_strength[block_id] = target_strength
                        self.block_age[block_id] = 1
                        births += 1 if previous < 0 else 0
                        changes += 1 if previous >= 0 else 0
            else:
                self.block_strength[block_id] *= 0.92
                if self.block_strength[block_id] < 0.15 and self.block_norms[block_id] >= 0:
                    self.block_norms[block_id] = -1
                    self.block_age[block_id] = 0
                    dissolutions += 1
        return {
            "hierarchy_births": births,
            "hierarchy_switches": changes,
            "hierarchy_dissolutions": dissolutions,
        }

    def hierarchy_metrics(self) -> tuple[int, float, list[float]]:
        active = [idx for idx, norm in enumerate(self.block_norms) if norm >= 0]
        if not active:
            return 0, 0.0, [0.0] * len(NORMS)
        counts = [0] * len(NORMS)
        for idx in active:
            counts[self.block_norms[idx]] += 1
        coverage = len(active) / self.block_count
        return len(active), coverage, [count / len(active) for count in counts]

    def hierarchy_alignment(self) -> float:
        alive = [cell for cell in self.cells if cell.alive]
        if not alive:
            return 0.0
        aligned = [
            cell
            for cell in alive
            if self.block_norms[cell.block_id] >= 0 and self.block_norms[cell.block_id] == cell.norm
        ]
        return len(aligned) / len(alive)

    def metrics(self, step: int, stats: dict[str, float]) -> dict[str, Any]:
        alive = [cell for cell in self.cells if cell.alive]
        resilient = [cell for cell in alive if cell.resilient]
        norm_counts = [0] * len(NORMS)
        landuse_counts = [[0] * len(NORMS) for _ in LAND_USES]
        landuse_service = [0.0] * len(LAND_USES)
        landuse_n = [0] * len(LAND_USES)
        alive_stress = [cell.stress_memory for cell in alive]
        resilient_stress = [cell.stress_memory for cell in resilient]
        resilient_demand = sum(cell.demand for cell in resilient)
        resilient_served = sum(min(cell.served, cell.demand) for cell in resilient)
        for cell in alive:
            norm_counts[cell.norm] += 1
            landuse_counts[cell.landuse][cell.norm] += 1
            landuse_service[cell.landuse] += cell.served / max(cell.demand, 1e-6)
            landuse_n[cell.landuse] += 1
        hierarchy_count, hierarchy_coverage, hierarchy_norms = self.hierarchy_metrics()
        resilient_fraction = len(resilient) / len(self.cells)
        return {
            "step": step,
            "alive_fraction": len(alive) / len(self.cells),
            "resilient_fraction": resilient_fraction,
            "resilient_buildings_percent": 100.0 * resilient_fraction,
            "served_fraction": stats["served_fraction"],
            "resilient_service": min(1.0, resilient_served / max(resilient_demand, 1e-6)),
            "mean_stress_memory": sum(alive_stress) / max(1, len(alive_stress)),
            "resilient_stress_memory": sum(resilient_stress) / max(1, len(resilient_stress)),
            "max_stress_memory": max(alive_stress) if alive_stress else 0.0,
            "cooperation_rate": stats["cooperation_rate"],
            "cooperation_attempts": stats["cooperation_attempts"],
            "cooperation_successes": stats["cooperation_successes"],
            "pool_count": int(stats.get("pool_count", 0)),
            "pool_members": int(stats.get("pool_members", 0)),
            "outage_severity": float(stats.get("outage_severity", 0.0)),
            "outage_solar_factor": float(stats.get("outage_solar_factor", 1.0)),
            "grid_support": float(stats.get("grid_support", 0.0)),
            "norm_mutations": int(stats.get("norm_mutations", 0)),
            "norm_imitations": int(stats.get("norm_imitations", 0)),
            "building_rebuilds": int(stats.get("building_rebuilds", 0)),
            "hierarchy_adoptions": int(stats.get("hierarchy_adoptions", 0)),
            "hierarchy_births": int(stats.get("hierarchy_births", 0)),
            "hierarchy_switches": int(stats.get("hierarchy_switches", 0)),
            "hierarchy_dissolutions": int(stats.get("hierarchy_dissolutions", 0)),
            "hierarchy_count": hierarchy_count,
            "hierarchy_coverage": hierarchy_coverage,
            "hierarchy_alignment": self.hierarchy_alignment(),
            "block_norms": self.block_norms[:],
            "block_strength": self.block_strength[:],
            "block_rows": self.block_rows,
            "block_cols": self.block_cols,
            "block_size": self.block_size,
            "norm_frequencies": [
                count / max(1, len(alive))
                for count in norm_counts
            ],
            "hierarchy_norm_frequencies": hierarchy_norms,
            "norm_counts": norm_counts,
            "landuse_norm_counts": landuse_counts,
            "landuse_service": [
                landuse_service[i] / max(1, landuse_n[i])
                for i in range(len(LAND_USES))
            ],
            "model_mode": "static_fixed_norm" if self.fixed_norm is not None and not self.enable_norm_evolution else "evolving_norms",
            "fixed_norm_key": NORMS[self.fixed_norm]["key"] if self.fixed_norm is not None else "",
            "enable_hierarchy": self.enable_hierarchy,
            "enable_rebuild": self.enable_rebuild,
            "enable_shared_storage_pool": bool(self.config.get("enable_shared_storage_pool", False)),
            "share_radius": int(self.config.get("share_radius", 0)),
            "sharing_min_efficiency": float(self.config.get("sharing_min_efficiency", 0.0)),
            "sharing_efficiency_decay": float(self.config.get("sharing_efficiency_decay", 0.0)),
            "sharing_loss_exponent": float(self.config.get("sharing_loss_exponent", 1.0)),
        }

    def step(self, step: int) -> dict[str, Any]:
        stats: dict[str, float] = self.energy_step(step)
        stats.update(self.evolution_step())
        stats.update(self.hierarchy_step())
        return self.metrics(step, stats)

    def state_snapshot(self) -> list[tuple[bool, float, float]]:
        return [(cell.alive, cell.health, cell.storage) for cell in self.cells]

    def run(self) -> SimulationResult:
        frames = []
        contact = []
        metrics = []
        steps = int(self.config["steps"])
        convergence_window = int(self.config.get("convergence_window", 24))
        convergence_epsilon = float(self.config.get("convergence_epsilon", 0.004))
        convergence_required = int(self.config.get("convergence_required_stable_steps", 24))
        min_convergence_step = int(self.config.get("min_convergence_step", convergence_window * 3))
        stop_on_convergence = bool(self.config.get("stop_on_convergence", False))
        stop_on_extinction = bool(self.config.get("stop_on_extinction", False))
        snapshots: list[list[tuple[bool, float, float]]] = []
        stable_steps = 0
        render_stride = int(self.config.get("render_stride", 2))
        contact_stride = max(1, steps // 12)
        for step in range(steps):
            row = self.step(step)
            snapshot = self.state_snapshot()
            if step >= convergence_window:
                previous = snapshots[step - convergence_window]
                daily_health_delta = max(
                    abs(now[1] - old[1])
                    for now, old in zip(snapshot, previous)
                )
                daily_storage_delta = max(
                    abs(now[2] - old[2]) / max(cell.storage_cap, 1e-6)
                    for cell, now, old in zip(self.cells, snapshot, previous)
                )
                daily_alive_changes = sum(1 for now, old in zip(snapshot, previous) if now[0] != old[0])
            else:
                daily_health_delta = 1.0
                daily_storage_delta = 1.0
                daily_alive_changes = len(self.cells)
            stable = (
                step >= min_convergence_step
                and daily_alive_changes == 0
                and daily_health_delta <= convergence_epsilon
                and daily_storage_delta <= convergence_epsilon
            )
            stable_steps = stable_steps + 1 if stable else 0
            row["daily_health_delta"] = daily_health_delta
            row["daily_storage_delta"] = daily_storage_delta
            row["daily_alive_changes"] = daily_alive_changes
            row["stable_steps"] = stable_steps
            row["converged"] = stable_steps >= convergence_required
            metrics.append(row)
            snapshots.append(snapshot)
            if step % render_stride == 0 or step == steps - 1 or row["converged"]:
                image = render_state(self.cells, self.landuse_map, self.size, row)
                frames.append(image)
                if len(contact) < 12 and step % contact_stride == 0:
                    contact.append(image)
            if row["converged"] and stop_on_convergence:
                break
            if stop_on_extinction and row["alive_fraction"] <= 0.0:
                break
        return SimulationResult(frames, contact, metrics, self.cells, self.data, self.config)


def blend(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(v * amount + 248 * (1 - amount)) for v in color)


def render_panel_title(draw: ImageDraw.ImageDraw, x: int, y: int, title: str) -> None:
    draw.text((x, y), title, fill=(15, 23, 42))


def draw_text_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fill: tuple[int, int, int] = (248, 250, 252),
    text_fill: tuple[int, int, int] = (15, 23, 42),
) -> None:
    bbox = draw.textbbox((x, y), text)
    pad = 2
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=fill,
        outline=(15, 23, 42),
    )
    draw.text((x, y), text, fill=text_fill)


def render_hierarchy_overlay(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    scale: int,
    size: int,
    metric: dict[str, Any],
) -> None:
    block_norms = metric.get("block_norms", [])
    block_strength = metric.get("block_strength", [])
    block_rows = int(metric.get("block_rows", 0))
    block_cols = int(metric.get("block_cols", 0))
    block_size = int(metric.get("block_size", size))
    for br in range(block_rows):
        for bc in range(block_cols):
            bid = br * block_cols + bc
            if bid >= len(block_norms) or block_norms[bid] < 0:
                continue
            norm = block_norms[bid]
            strength = block_strength[bid] if bid < len(block_strength) else 0.0
            color = NORMS[norm]["color"]
            x = x0 + bc * block_size * scale
            y = y0 + br * block_size * scale
            x1 = min(x0 + size * scale, x + block_size * scale) - 1
            y1 = min(y0 + size * scale, y + block_size * scale) - 1
            width = 1 + int(2 * strength)
            for offset in range(width):
                draw.rectangle([x + offset, y + offset, x1 - offset, y1 - offset], outline=color)
            if block_size * scale >= 40:
                draw_text_badge(draw, x + 4, y + 4, NORMS[norm]["key"])


def draw_state_legend(draw: ImageDraw.ImageDraw, x: int, y: int, metric: dict[str, Any]) -> None:
    cursor = x
    for landuse in LAND_USES:
        draw.rectangle([cursor, y + 3, cursor + 12, y + 15], fill=landuse["color"])
        draw.text((cursor + 16, y), f"{landuse['key']} {landuse['name']}", fill=(15, 23, 42))
        cursor += 108

    cursor += 12
    fixed_key = str(metric.get("fixed_norm_key", "")).upper()
    norm_items = [norm for norm in NORMS if not fixed_key or norm["key"] == fixed_key]
    for norm in norm_items:
        draw.rectangle([cursor, y + 3, cursor + 12, y + 15], fill=norm["color"])
        draw.text((cursor + 16, y), f"{norm['key']} {norm['name']}", fill=(15, 23, 42))
        cursor += 110

    for color, label in [
        ((22, 163, 74), "served"),
        ((234, 179, 8), "partial"),
        ((220, 38, 38), "deficit"),
        ((30, 41, 59), "dead"),
    ]:
        draw.rectangle([cursor, y + 3, cursor + 12, y + 15], fill=color)
        draw.text((cursor + 16, y), label, fill=(15, 23, 42))
        cursor += 74

    marker_x = cursor + 4
    draw.rectangle([marker_x, y + 4, marker_x + 10, y + 14], outline=(15, 23, 42))
    draw.text((marker_x + 16, y), "resilient <=5% deficit", fill=(15, 23, 42))


def render_state(cells: list[Cell], landuse_map: list[int], size: int, metric: dict[str, Any]) -> Image.Image:
    scale = 8
    margin = 18
    gap = 16
    panel = size * scale
    footer = 92
    width = panel * 3 + gap * 2 + margin * 2
    height = panel + margin * 2 + footer
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    fixed_key = str(metric.get("fixed_norm_key", "SELF")).upper()
    norm_title = f"static {fixed_key} rule" if metric.get("model_mode") == "static_fixed_norm" else "individual norm + hierarchy"
    titles = ["land-use attribute", norm_title, "service + health"]
    for pidx, title in enumerate(titles):
        x0 = margin + pidx * (panel + gap)
        y0 = margin + 16
        render_panel_title(draw, x0, margin - 4, title)
        draw.rectangle([x0 - 1, y0 - 1, x0 + panel, y0 + panel], outline=(148, 163, 184))
        for idx, cell in enumerate(cells):
            row, col = divmod(idx, size)
            x = x0 + col * scale
            y = y0 + row * scale
            if pidx == 0:
                color = LAND_USES[cell.landuse]["color"]
                if not cell.alive:
                    color = blend(color, 0.35)
            elif pidx == 1:
                color = NORMS[cell.norm]["color"] if cell.alive else (203, 213, 225)
            else:
                service = cell.served / max(cell.demand, 1e-6) if cell.alive else 0.0
                if not cell.alive:
                    color = (30, 41, 59)
                elif service >= 0.98:
                    color = blend((22, 163, 74), 0.65 + 0.30 * cell.health)
                elif service >= 0.70:
                    color = blend((234, 179, 8), 0.55 + 0.25 * cell.health)
                else:
                    color = blend((220, 38, 38), 0.55 + 0.25 * cell.health)
            draw.rectangle([x, y, x + scale - 1, y + scale - 1], fill=color)
            if cell.resilient and pidx != 0:
                draw.rectangle(
                    [x + 2, y + 2, x + scale - 3, y + scale - 3],
                    outline=(255, 255, 255),
                )
        if pidx == 1:
            render_hierarchy_overlay(draw, x0, y0, scale, size, metric)

    norm_freq = metric["norm_frequencies"]
    top_norm = max(range(len(norm_freq)), key=lambda idx: norm_freq[idx])
    if metric.get("model_mode") == "static_fixed_norm":
        footer_text = (
            f"step={metric['step']:03d} alive={100.0 * metric['alive_fraction']:.1f}% "
            f"norm={metric.get('fixed_norm_key', NORMS[top_norm]['key'])} "
            f"converged={metric.get('converged', False)}"
        )
    else:
        footer_text = (
            f"step={metric['step']:03d} served={metric['served_fraction']:.2f} "
            f"resilient={metric['resilient_fraction']:.2f} coop={metric['cooperation_rate']:.2f} "
            f"hier={metric['hierarchy_coverage']:.2f} align={metric['hierarchy_alignment']:.2f} "
            f"top_norm={NORMS[top_norm]['key']} {norm_freq[top_norm]:.2f} "
            f"converged={metric.get('converged', False)}"
        )
    draw_state_legend(draw, margin, height - footer + 12, metric)
    draw.text((margin, height - footer + 40), footer_text, fill=(15, 23, 42))
    return image


def make_contact_sheet(frames: list[Image.Image], path: Path, columns: int = 3) -> None:
    thumbs = [frame.resize((360, 184), Image.Resampling.NEAREST) for frame in frames]
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 360, rows * 184), (248, 250, 252))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 360, (idx // columns) * 184))
    sheet.save(path)


def block_summaries(cells: list[Cell], size: int, metric: dict[str, Any]) -> list[dict[str, Any]]:
    block_norms = metric.get("block_norms", [])
    block_strength = metric.get("block_strength", [])
    block_rows = int(metric.get("block_rows", 0))
    block_cols = int(metric.get("block_cols", 0))
    block_size = int(metric.get("block_size", size))
    summaries = []
    for br in range(block_rows):
        for bc in range(block_cols):
            block_id = br * block_cols + bc
            members = [cell for cell in cells if cell.block_id == block_id]
            alive = [cell for cell in members if cell.alive]
            norm_counts = [0] * len(NORMS)
            landuse_counts = [0] * len(LAND_USES)
            service = 0.0
            resilient_count = 0
            for cell in alive:
                norm_counts[cell.norm] += 1
                landuse_counts[cell.landuse] += 1
                service += cell.served / max(cell.demand, 1e-6)
                if cell.resilient:
                    resilient_count += 1
            hierarchy_norm = block_norms[block_id] if block_id < len(block_norms) else -1
            hierarchy_strength = (
                block_strength[block_id]
                if hierarchy_norm >= 0 and block_id < len(block_strength)
                else 0.0
            )
            aligned = [
                cell
                for cell in alive
                if hierarchy_norm >= 0 and cell.norm == hierarchy_norm
            ]
            top_norm = max(range(len(NORMS)), key=lambda idx: norm_counts[idx]) if alive else -1
            top_landuse = max(range(len(LAND_USES)), key=lambda idx: landuse_counts[idx]) if members else -1
            summaries.append(
                {
                    "block_id": block_id,
                    "block_row": br,
                    "block_col": bc,
                    "row_start": br * block_size,
                    "col_start": bc * block_size,
                    "hierarchy_rule": NORMS[hierarchy_norm]["key"] if hierarchy_norm >= 0 else "",
                    "hierarchy_rule_name": NORMS[hierarchy_norm]["name"] if hierarchy_norm >= 0 else "",
                    "hierarchy_strength": hierarchy_strength,
                    "hierarchy_alignment": len(aligned) / max(1, len(alive)),
                    "dominant_individual_norm": NORMS[top_norm]["key"] if top_norm >= 0 else "",
                    "dominant_individual_norm_frequency": max(norm_counts) / max(1, len(alive)),
                    "dominant_landuse": LAND_USES[top_landuse]["key"] if top_landuse >= 0 else "",
                    "alive_fraction": len(alive) / max(1, len(members)),
                    "mean_service": service / max(1, len(alive)),
                    "resilient_fraction": resilient_count / max(1, len(members)),
                    "resilient_count": resilient_count,
                }
            )
    return summaries


def write_hierarchy_blocks_csv(cells: list[Cell], size: int, metric: dict[str, Any], path: Path) -> None:
    summaries = block_summaries(cells, size, metric)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def write_hierarchy_map_png(cells: list[Cell], size: int, metric: dict[str, Any], path: Path) -> None:
    summaries = block_summaries(cells, size, metric)
    block_rows = int(metric.get("block_rows", 0))
    block_cols = int(metric.get("block_cols", 0))
    cell_w = 150
    cell_h = 108
    margin = 40
    header = 58
    width = margin * 2 + block_cols * cell_w
    height = margin + header + block_rows * cell_h + 96
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 24), "Final hierarchical rule map", fill=(15, 23, 42))
    draw.text(
        (margin, 42),
        "Each block label shows the institutionalized rule, its strength, alignment, and dominant individual norm.",
        fill=(71, 85, 105),
    )
    for item in summaries:
        x = margin + int(item["block_col"]) * cell_w
        y = margin + header + int(item["block_row"]) * cell_h
        rule = item["hierarchy_rule"]
        if rule:
            norm_idx = next(idx for idx, norm in enumerate(NORMS) if norm["key"] == rule)
            color = blend(NORMS[norm_idx]["color"], 0.78)
        else:
            color = (226, 232, 240)
        draw.rectangle([x, y, x + cell_w - 10, y + cell_h - 10], fill=color, outline=(15, 23, 42))
        draw.text((x + 10, y + 10), f"B{item['block_id']:02d} {rule or 'none'}", fill=(15, 23, 42))
        draw.text((x + 10, y + 31), f"strength {item['hierarchy_strength']:.2f}", fill=(15, 23, 42))
        draw.text((x + 10, y + 50), f"align {item['hierarchy_alignment']:.2f}", fill=(15, 23, 42))
        draw.text((x + 10, y + 69), f"indiv {item['dominant_individual_norm']} {item['dominant_individual_norm_frequency']:.2f}", fill=(15, 23, 42))
        draw.text((x + 10, y + 88), f"LU {item['dominant_landuse']} alive {item['alive_fraction']:.2f}", fill=(15, 23, 42))

    legend_x = margin
    legend_y = height - 74
    draw.text((legend_x, legend_y), "Rule colors", fill=(15, 23, 42))
    legend_y += 22
    for idx, norm in enumerate(NORMS):
        x = legend_x + (idx % 4) * 190
        y = legend_y + (idx // 4) * 24
        draw.rectangle([x, y + 3, x + 16, y + 16], fill=norm["color"])
        draw.text((x + 22, y), f"{norm['key']} {norm['name']}", fill=(15, 23, 42))
    image.save(path)


def iso_point(
    row: float,
    col: float,
    z: float,
    origin_x: int,
    origin_y: int,
    tile_w: float,
    tile_h: float,
    z_scale: float,
) -> tuple[int, int]:
    x = origin_x + (col - row) * tile_w * 0.5
    y = origin_y + (col + row) * tile_h * 0.5 - z * z_scale
    return round(x), round(y)


def draw_iso_quad(
    draw: ImageDraw.ImageDraw,
    corners: list[tuple[float, float, float]],
    origin_x: int,
    origin_y: int,
    tile_w: float,
    tile_h: float,
    z_scale: float,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
) -> list[tuple[int, int]]:
    points = [
        iso_point(row, col, z, origin_x, origin_y, tile_w, tile_h, z_scale)
        for row, col, z in corners
    ]
    draw.polygon(points, fill=fill)
    if outline:
        draw.line(points + [points[0]], fill=outline, width=1)
    return points


def write_hierarchy_canopy_png(cells: list[Cell], size: int, metric: dict[str, Any], path: Path) -> None:
    block_norms = metric.get("block_norms", [])
    block_strength = metric.get("block_strength", [])
    block_rows = int(metric.get("block_rows", 0))
    block_cols = int(metric.get("block_cols", 0))
    block_size = int(metric.get("block_size", size))
    tile_w = 17.0
    tile_h = 8.5
    z_scale = 58.0
    width = 1180
    height = 820
    origin_x = width // 2
    origin_y = 360
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 26), "2D hierarchy canopy", fill=(15, 23, 42))
    draw.text(
        (36, 46),
        "Base plane is the 2D land-use grid. Elevated canopies are block-level rules; height shows rule strength.",
        fill=(71, 85, 105),
    )

    # Base city tissue.
    for diagonal in range(size * 2):
        for row in range(size):
            col = diagonal - row
            if col < 0 or col >= size:
                continue
            cell = cells[row * size + col]
            color = LAND_USES[cell.landuse]["color"]
            if not cell.alive:
                color = blend(color, 0.32)
            draw_iso_quad(
                draw,
                [
                    (row, col, 0.0),
                    (row, col + 1, 0.0),
                    (row + 1, col + 1, 0.0),
                    (row + 1, col, 0.0),
                ],
                origin_x,
                origin_y,
                tile_w,
                tile_h,
                z_scale,
                fill=color,
                outline=(226, 232, 240),
            )

    # Vertical ties from aligned agents to their institutionalized block rule.
    for diagonal in range(size * 2):
        for row in range(size):
            col = diagonal - row
            if col < 0 or col >= size:
                continue
            cell = cells[row * size + col]
            if not cell.alive:
                continue
            if cell.block_id >= len(block_norms):
                continue
            rule = block_norms[cell.block_id]
            if rule < 0 or rule != cell.norm:
                continue
            strength = block_strength[cell.block_id] if cell.block_id < len(block_strength) else 0.0
            z_top = 1.2 + 2.8 * strength
            base = iso_point(row + 0.5, col + 0.5, 0.08, origin_x, origin_y, tile_w, tile_h, z_scale)
            top = iso_point(row + 0.5, col + 0.5, z_top, origin_x, origin_y, tile_w, tile_h, z_scale)
            line_color = blend(NORMS[rule]["color"], 0.72)
            draw.line([base, top], fill=line_color, width=1)

    # Elevated hierarchy surfaces.
    slabs = []
    for br in range(block_rows):
        for bc in range(block_cols):
            block_id = br * block_cols + bc
            if block_id >= len(block_norms) or block_norms[block_id] < 0:
                continue
            rule = block_norms[block_id]
            strength = block_strength[block_id] if block_id < len(block_strength) else 0.0
            r0 = br * block_size
            c0 = bc * block_size
            r1 = min(size, r0 + block_size)
            c1 = min(size, c0 + block_size)
            z_top = 1.2 + 2.8 * strength
            slabs.append((r0 + c0, r0, c0, r1, c1, rule, strength, z_top))
    for _, r0, c0, r1, c1, rule, strength, z_top in sorted(slabs):
        color = blend(NORMS[rule]["color"], 0.62)
        corners_base = [(r0, c0, 0.0), (r0, c1, 0.0), (r1, c1, 0.0), (r1, c0, 0.0)]
        corners_top = [(r0, c0, z_top), (r0, c1, z_top), (r1, c1, z_top), (r1, c0, z_top)]
        base_points = [iso_point(*p, origin_x, origin_y, tile_w, tile_h, z_scale) for p in corners_base]
        top_points = [iso_point(*p, origin_x, origin_y, tile_w, tile_h, z_scale) for p in corners_top]
        for bp, tp in zip(base_points, top_points):
            draw.line([bp, tp], fill=(100, 116, 139), width=1)
        draw.polygon(top_points, fill=color)
        draw.line(top_points + [top_points[0]], fill=NORMS[rule]["color"], width=2 + int(2 * strength))
        cx, cy = iso_point((r0 + r1) * 0.5, (c0 + c1) * 0.5, z_top, origin_x, origin_y, tile_w, tile_h, z_scale)
        draw_text_badge(draw, cx - 13, cy - 7, NORMS[rule]["key"])

    # Legend.
    legend_x = 36
    legend_y = height - 126
    draw.text((legend_x, legend_y), "Hierarchy rule colors", fill=(15, 23, 42))
    legend_y += 22
    for idx, norm in enumerate(NORMS):
        x = legend_x + (idx % 4) * 200
        y = legend_y + (idx // 4) * 25
        draw.rectangle([x, y + 3, x + 16, y + 16], fill=norm["color"])
        draw.text((x + 22, y), f"{norm['key']} {norm['name']}", fill=(15, 23, 42))
    draw.text((36, height - 28), "Vertical lines connect aligned buildings to their block-level rule.", fill=(71, 85, 105))
    image.save(path)


def normalized_resilience_auc(metrics: list[dict[str, Any]]) -> float:
    if not metrics:
        return 0.0
    if len(metrics) == 1:
        return max(0.0, min(1.0, float(metrics[0]["alive_fraction"])))
    area = 0.0
    for left, right in zip(metrics, metrics[1:]):
        dt = max(0.0, float(right["step"]) - float(left["step"]))
        q0 = max(0.0, min(1.0, float(left["alive_fraction"])))
        q1 = max(0.0, min(1.0, float(right["alive_fraction"])))
        area += 0.5 * (q0 + q1) * dt
    duration = max(1.0, float(metrics[-1]["step"]) - float(metrics[0]["step"]))
    return max(0.0, min(1.0, area / duration))


def write_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        scalar_fields = [
            "step",
            "alive_fraction",
            "resilient_fraction",
            "resilient_buildings_percent",
            "served_fraction",
            "resilient_service",
            "cooperation_rate",
            "cooperation_attempts",
            "cooperation_successes",
            "pool_count",
            "pool_members",
            "outage_severity",
            "outage_solar_factor",
            "grid_support",
            "mean_stress_memory",
            "resilient_stress_memory",
            "max_stress_memory",
            "norm_mutations",
            "norm_imitations",
            "building_rebuilds",
            "hierarchy_adoptions",
            "hierarchy_births",
            "hierarchy_switches",
            "hierarchy_dissolutions",
            "hierarchy_count",
            "hierarchy_coverage",
            "hierarchy_alignment",
            "daily_health_delta",
            "daily_storage_delta",
            "daily_alive_changes",
            "stable_steps",
            "converged",
            "model_mode",
            "fixed_norm_key",
            "enable_hierarchy",
            "enable_rebuild",
            "enable_shared_storage_pool",
            "share_radius",
            "sharing_min_efficiency",
            "sharing_efficiency_decay",
            "sharing_loss_exponent",
        ]
        fields = scalar_fields + [f"norm_{norm['key']}" for norm in NORMS] + [f"hierarchy_{norm['key']}" for norm in NORMS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            flat = {field: row.get(field, "") for field in scalar_fields}
            for idx, norm in enumerate(NORMS):
                flat[f"norm_{norm['key']}"] = row["norm_frequencies"][idx]
                flat[f"hierarchy_{norm['key']}"] = row["hierarchy_norm_frequencies"][idx]
            writer.writerow(flat)


def draw_series_panel(
    draw: ImageDraw.ImageDraw,
    metrics: list[dict[str, Any]],
    x0: int,
    y0: int,
    width: int,
    height: int,
    series: list[tuple[str, str, tuple[int, int, int], list[float]]],
) -> None:
    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=(255, 255, 255), outline=(203, 213, 225))
    for tick in range(6):
        y = y0 + height - round(height * tick / 5)
        draw.line([x0, y, x0 + width, y], fill=(226, 232, 240))
        draw.text((x0 - 34, y - 5), f"{tick / 5:.1f}", fill=(71, 85, 105))
    if len(metrics) <= 1:
        return
    max_step = max(1, metrics[-1]["step"])
    for _, _, color, values in series:
        points = []
        for metric, value in zip(metrics, values):
            x = x0 + round(width * metric["step"] / max_step)
            y = y0 + height - round(height * max(0.0, min(1.0, value)))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    legend_x = x0 + 14
    legend_y = y0 + 12
    for key, label, color, _ in series:
        draw.rectangle([legend_x, legend_y + 3, legend_x + 16, legend_y + 13], fill=color)
        draw.text((legend_x + 22, legend_y), f"{key} {label}", fill=(15, 23, 42))
        legend_y += 18


def write_metrics_png(metrics: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (1160, 760), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    static_mode = metrics[-1].get("model_mode") == "static_fixed_norm"
    title = "Static fixed-rule energy ABM metrics" if static_mode else "Land-use conditioned norm hierarchy metrics"
    draw.text((36, 24), title, fill=(15, 23, 42))
    if static_mode:
        cumulative_resilience = [
            normalized_resilience_auc(metrics[: idx + 1])
            for idx in range(len(metrics))
        ]
        resilience_series = [
            ("alive", "alive buildings", (15, 23, 42), [row["alive_fraction"] for row in metrics]),
            ("R", "normalized resilience AUC", (37, 99, 235), cumulative_resilience),
        ]
    else:
        resilience_series = [
            ("alive", "active buildings", (15, 23, 42), [row["alive_fraction"] for row in metrics]),
            ("served", "load served", (22, 163, 74), [row["served_fraction"] for row in metrics]),
            ("resilient", "buildings <=5% cumulative deficit", (220, 38, 38), [row["resilient_fraction"] for row in metrics]),
        ]
    if not static_mode:
        resilience_series.extend([
            ("hier", "hierarchy coverage", (147, 51, 234), [row["hierarchy_coverage"] for row in metrics]),
            ("align", "norm hierarchy alignment", (14, 165, 233), [row["hierarchy_alignment"] for row in metrics]),
        ])
    draw.text((72, 70), "evaluation metrics" if static_mode else "system and hierarchy", fill=(15, 23, 42))
    draw_series_panel(draw, metrics, 72, 96, 1000, 250, resilience_series)

    if static_mode:
        draw.text((72, 396), "Final evaluation", fill=(15, 23, 42))
        draw.text((72, 428), f"Alive buildings: {100.0 * metrics[-1]['alive_fraction']:.1f}%", fill=(15, 23, 42))
        draw.text((72, 456), f"Resilience normalized to [0,1]: {normalized_resilience_auc(metrics):.3f}", fill=(15, 23, 42))
        draw.text((72, 484), "R is the area under Q(t), where Q(t) is alive-building fraction.", fill=(71, 85, 105))
    else:
        norm_series = [
            (
                norm["key"],
                norm["name"],
                norm["color"],
                [row["norm_frequencies"][idx] for row in metrics],
            )
            for idx, norm in enumerate(NORMS)
        ]
        draw.text((72, 396), "individual norm frequencies among surviving buildings", fill=(15, 23, 42))
        draw_series_panel(draw, metrics, 72, 422, 1000, 220, norm_series)
    image.save(path)


def write_html(out_dir: Path, summary: dict[str, Any]) -> None:
    static_mode = summary.get("model_mode") == "static_fixed_norm"
    fixed_key = summary.get("fixed_norm_key") or "fixed"
    page_title = f"Static {fixed_key}-Rule Energy ABM" if static_mode else "Land-Use Conditioned Norm Hierarchy"
    intro = (
        "All buildings use one fixed sharing rule. Norm mutation, imitation, hierarchy, and rebuild are disabled. "
        "Evaluation reports alive buildings (%) and normalized resilience AUC."
        if static_mode
        else (
            "Land use is a fixed cell attribute that shapes demand and generation. "
            "Buildings adapt norms quickly. Rebuild is slow. Block-level norms can emerge from local individual norm success "
            "and then bias future behavior."
        )
    )
    hierarchy_panels = "" if static_mode else """
    <div class="panel">
      <h2>Hierarchy Rule Map</h2>
      <img src="landuse_norm_hierarchy_map.png" alt="annotated hierarchy rule map">
    </div>
    <div class="panel">
      <h2>Hierarchy Canopy</h2>
      <img src="landuse_norm_hierarchy_canopy.png" alt="2D hierarchy canopy">
    </div>
"""
    convergence_summary = (
        f", converged={summary.get('converged', False)}, stable steps={summary.get('stable_steps', 0)}, "
        f"daily health delta={summary.get('daily_health_delta', 0.0):.4f}, "
        f"daily storage delta={summary.get('daily_storage_delta', 0.0):.4f}"
        if static_mode
        else ""
    )
    if static_mode:
        final_summary = (
            f"alive buildings={summary.get('alive_buildings_percent', 0.0):.1f}%, "
            f"resilience={summary.get('resilience_normalized', 0.0):.3f}{convergence_summary}."
        )
    else:
        final_summary = (
            f"served={summary['served_fraction']:.2f}, "
            f"resilient={summary['resilient_fraction']:.2f}, "
            f"hierarchy coverage={summary['hierarchy_coverage']:.2f}, "
            f"alignment={summary['hierarchy_alignment']:.2f}, "
            f"top norm={summary['top_norm']}."
        )
    norm_rows = "\n".join(
        f"<tr><td>{norm['key']}</td><td>{norm['name']}</td><td><span style='display:inline-block;width:14px;height:14px;background:rgb{norm['color']}'></span></td></tr>"
        for norm in NORMS
    )
    landuse_rows = "\n".join(
        f"<tr><td>{land['key']}</td><td>{land['name']}</td><td><span style='display:inline-block;width:14px;height:14px;background:rgb{land['color']}'></span></td></tr>"
        for land in LAND_USES
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #eef2f7; color: #0f172a; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    p {{ color: #475569; line-height: 1.5; }}
    .panel {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; margin-top: 18px; }}
    img {{ max-width: 100%; height: auto; display: block; border: 1px solid #cbd5e1; background: #f8fafc; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; text-align: left; padding: 8px; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>{page_title}</h1>
    <p>{intro}</p>
    <div class="panel">
      <strong>Final summary:</strong>
      {final_summary}
    </div>
    <div class="panel">
      <h2>Evolution</h2>
      <img src="landuse_norm_evolution.gif" alt="land-use norm evolution">
    </div>
    <div class="panel">
      <h2>Contact Sheet</h2>
      <img src="landuse_norm_contact_sheet.png" alt="land-use norm contact sheet">
    </div>
    <div class="panel">
      <h2>Metrics</h2>
      <img src="landuse_norm_metrics.png" alt="land-use norm metrics">
    </div>
    {hierarchy_panels}
    <div class="panel">
      <h2>Norms</h2>
      <table><tbody>{norm_rows}</tbody></table>
    </div>
    <div class="panel">
      <h2>Land Uses</h2>
      <table><tbody>{landuse_rows}</tbody></table>
    </div>
    <div class="panel">
      <p>Metrics: <a href="landuse_norm_metrics.png">PNG</a> / <a href="landuse_norm_metrics.csv">CSV</a> / <a href="landuse_norm_metrics.json">JSON</a></p>
      <p>Hierarchy blocks: <a href="landuse_norm_hierarchy_map.png">map PNG</a> / <a href="landuse_norm_hierarchy_canopy.png">canopy PNG</a> / <a href="landuse_norm_hierarchy_blocks.csv">CSV</a></p>
    </div>
  </main>
</body>
</html>
"""
    (out_dir / "landuse_norm_results.html").write_text(html, encoding="utf-8")


def save_result(result: SimulationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "landuse_norm_evolution.gif"
    result.frames[0].save(
        gif_path,
        save_all=True,
        append_images=result.frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )
    result.frames[-1].save(out_dir / "landuse_norm_final_snapshot.png")
    make_contact_sheet(result.contact_frames, out_dir / "landuse_norm_contact_sheet.png")
    write_metrics_png(result.metrics, out_dir / "landuse_norm_metrics.png")
    write_hierarchy_map_png(
        result.final_cells,
        int(math.sqrt(len(result.final_cells))),
        result.metrics[-1],
        out_dir / "landuse_norm_hierarchy_map.png",
    )
    write_hierarchy_canopy_png(
        result.final_cells,
        int(math.sqrt(len(result.final_cells))),
        result.metrics[-1],
        out_dir / "landuse_norm_hierarchy_canopy.png",
    )
    write_hierarchy_blocks_csv(
        result.final_cells,
        int(math.sqrt(len(result.final_cells))),
        result.metrics[-1],
        out_dir / "landuse_norm_hierarchy_blocks.csv",
    )
    (out_dir / "landuse_norm_metrics.json").write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )
    write_csv(result.metrics, out_dir / "landuse_norm_metrics.csv")
    summary = result.metrics[-1].copy()
    summary["alive_buildings_percent"] = 100.0 * summary["alive_fraction"]
    summary["resilience_normalized"] = normalized_resilience_auc(result.metrics)
    top_norm = max(range(len(NORMS)), key=lambda idx: summary["norm_frequencies"][idx])
    summary["top_norm"] = f"{NORMS[top_norm]['key']} / {NORMS[top_norm]['name']}"
    write_html(out_dir, summary)


def run_from_config(config_path: str) -> tuple[SimulationResult, Path]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    data = load_data(config)
    sim = LandUseNormSimulation(config, data)
    result = sim.run()
    out_dir = Path(config["out_dir"])
    save_result(result, out_dir)
    return result, out_dir
