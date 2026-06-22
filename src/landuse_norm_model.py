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
        "key": "ALLC",
        "name": "generous",
        "color": (37, 99, 235),
        "kind": "generous",
        "assessment": (1, 0, 1, 0),
    },
    {
        "key": "SELF",
        "name": "selfish",
        "color": (120, 113, 108),
        "kind": "selfish",
        "assessment": (1, 1, 1, 1),
    },
    {
        "key": "DISC",
        "name": "standing",
        "color": (22, 163, 74),
        "kind": "standing",
        "assessment": (1, 0, 1, 1),
    },
    {
        "key": "SJ",
        "name": "stern judging",
        "color": (147, 51, 234),
        "kind": "stern",
        "assessment": (1, 0, 0, 1),
    },
    {
        "key": "SHUN",
        "name": "shunning",
        "color": (220, 38, 38),
        "kind": "shunning",
        "assessment": (1, 0, 0, 0),
    },
    {
        "key": "CRIT",
        "name": "critical first",
        "color": (234, 88, 12),
        "kind": "critical",
        "assessment": (1, 0, 1, 1),
    },
    {
        "key": "MKT",
        "name": "market",
        "color": (14, 165, 233),
        "kind": "market",
        "assessment": (1, 0, 1, 1),
    },
    {
        "key": "LOCAL",
        "name": "neighbor loyal",
        "color": (202, 138, 4),
        "kind": "local",
        "assessment": (1, 0, 1, 1),
    },
]


@dataclass
class Cell:
    landuse: int
    norm: int
    block_id: int
    alive: bool = True
    critical: bool = False
    reputation: float = 0.62
    health: float = 1.0
    storage: float = 0.0
    storage_cap: float = 1.0
    demand: float = 0.0
    served: float = 0.0
    deficit: float = 0.0
    surplus: float = 0.0
    payoff: float = 0.0


@dataclass
class DataBundle:
    demand_curves: dict[str, list[float]]
    solar_curves: dict[str, list[float]]
    roof_area_m2: dict[str, float]
    sources: dict[str, str]


@dataclass
class SimulationResult:
    frames: list[Image.Image]
    contact_frames: list[Image.Image]
    metrics: list[dict[str, Any]]
    final_cells: list[Cell]
    data: DataBundle


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


def load_demand_curves(path: str, sample_size: int) -> tuple[dict[str, list[float]], dict[str, int]]:
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
                curves[profile_type].append(hourly_to_daily(values))
                if all(len(values_) >= target_each for values_ in curves.values()):
                    break
    out = {}
    counts = {}
    for kind in curves:
        counts[kind] = len(curves[kind])
        if curves[kind]:
            out[kind] = normalize_mean(
                [
                    sum(curve[hour] for curve in curves[kind]) / len(curves[kind])
                    for hour in range(24)
                ]
            )
        else:
            out[kind] = fallback_demand(kind)
    out["industrial"] = fallback_demand("industrial")
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


def load_data(config: dict[str, Any]) -> DataBundle:
    demand_curves, profile_counts = load_demand_curves(
        config["demand_csv"],
        int(config["profile_sample_size"]),
    )
    roof = estimate_roof_area(
        config["metadata_csv"],
        int(config["profile_sample_size"]),
        float(config["roof_usable_fraction"]),
    )
    ghi, temp, solar_source = read_epw_24h(
        config["solar_epw"],
        int(config["solar_month"]),
        int(config["solar_day"]),
    )
    pv = pv_per_m2_kw(ghi, temp, float(config["pv_efficiency"]))
    raw_solar = {kind: [v * roof[kind] for v in pv] for kind in roof}
    reference = sum(sum(v) / 24 for v in raw_solar.values()) / len(raw_solar)
    if reference <= 0:
        reference = 1.0
    solar_curves = {
        kind: [value / reference for value in values]
        for kind, values in raw_solar.items()
    }
    return DataBundle(
        demand_curves=demand_curves,
        solar_curves=solar_curves,
        roof_area_m2=roof,
        sources={
            "demand_csv": config["demand_csv"],
            "metadata_csv": config["metadata_csv"],
            "solar_epw": solar_source,
            "profiles": json.dumps(profile_counts),
        },
    )


class LandUseNormSimulation:
    def __init__(self, config: dict[str, Any], data: DataBundle) -> None:
        self.config = config
        self.data = data
        self.size = int(config["grid_size"])
        self.rng = random.Random(int(config["seed"]))
        self.block_size = int(config.get("block_size", 6))
        self.block_rows = math.ceil(self.size / self.block_size)
        self.block_cols = math.ceil(self.size / self.block_size)
        self.block_count = self.block_rows * self.block_cols
        self.block_norms = [-1 for _ in range(self.block_count)]
        self.block_strength = [0.0 for _ in range(self.block_count)]
        self.block_age = [0 for _ in range(self.block_count)]
        self.cells: list[Cell] = []
        self.landuse_map = self._make_landuse_map()
        self._init_cells()

    def block_id_for(self, row: int, col: int) -> int:
        return (row // self.block_size) * self.block_cols + (col // self.block_size)

    def _make_landuse_map(self) -> list[int]:
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
        critical_fraction = float(self.config.get("critical_fraction", 0.10))
        for idx, landuse in enumerate(self.landuse_map):
            row, col = divmod(idx, self.size)
            norm = self.initial_norm_for_landuse(landuse)
            spec = LAND_USES[landuse]
            storage_cap = spec["storage_cap"] * self.rng.uniform(0.78, 1.22)
            center_bias = math.exp(-math.hypot(row - self.size * 0.34, col - self.size * 0.34) / (self.size * 0.18))
            critical_prob = critical_fraction * (0.55 + 2.8 * center_bias)
            cell = Cell(
                landuse=landuse,
                norm=norm,
                block_id=self.block_id_for(row, col),
                reputation=self.rng.uniform(0.48, 0.82),
                health=self.rng.uniform(0.88, 1.0),
                storage_cap=storage_cap,
                storage=storage_cap * self.rng.uniform(0.38, 0.72),
                critical=self.rng.random() < critical_prob,
            )
            self.cells.append(cell)

    def initial_norm_for_landuse(self, landuse: int) -> int:
        if landuse == 0:
            options = [0, 2, 7, 2]
        elif landuse == 1:
            options = [1, 6, 6, 2]
        else:
            options = [1, 5, 6, 3]
        return self.rng.choice(options)

    def neighbor_indices(self, idx: int, radius: int) -> list[int]:
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
        return out

    def block_norm(self, cell: Cell) -> int:
        return self.block_norms[cell.block_id]

    def landuse_demand(self, landuse: int, hour: int) -> float:
        spec = LAND_USES[landuse]
        curve = self.data.demand_curves[spec["demand_kind"]]
        return curve[hour] * spec["demand_scale"]

    def landuse_solar(self, landuse: int, hour: int, step: int) -> float:
        spec = LAND_USES[landuse]
        curve = self.data.solar_curves[spec["roof_kind"]]
        shock = 1.0
        if int(self.config["shock_start"]) <= step < int(self.config["shock_end"]):
            shock = float(self.config["solar_shock_factor"])
        return curve[hour] * shock

    def should_share(self, donor: Cell, receiver: Cell, distance: float) -> bool:
        norm = NORMS[donor.norm]["kind"]
        receiver_good = receiver.reputation >= 0.5
        block_norm = self.block_norm(donor)
        institutional_bonus = block_norm == donor.norm and self.block_strength[donor.block_id] > 0.35
        threshold_shift = 0.10 if institutional_bonus else 0.0
        if institutional_bonus and receiver.block_id == donor.block_id:
            return donor.storage > donor.storage_cap * 0.22
        if norm == "generous":
            return donor.storage > donor.storage_cap * (0.20 - threshold_shift)
        if norm == "selfish":
            return receiver.critical and donor.surplus > donor.demand * (0.65 - threshold_shift)
        if norm == "standing":
            return receiver_good or receiver.critical
        if norm == "stern":
            return receiver_good or receiver.critical
        if norm == "shunning":
            return receiver_good and donor.storage > donor.storage_cap * (0.42 - threshold_shift)
        if norm == "critical":
            return receiver.critical or (receiver_good and donor.storage > donor.storage_cap * (0.35 - threshold_shift))
        if norm == "market":
            return receiver.deficit * (2.0 if receiver.critical else 1.0) > 0.30 + 0.08 * distance - threshold_shift
        if norm == "local":
            return receiver_good and distance <= (2.25 + (0.75 if institutional_bonus else 0.0))
        return False

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
        if receiver.critical and action:
            good = 1
        return min(1.0, donor.reputation + 0.08) if good else max(0.0, donor.reputation - 0.16)

    def energy_step(self, step: int) -> dict[str, float]:
        hour = step % 24
        in_shock = int(self.config["shock_start"]) <= step < int(self.config["shock_end"])
        grid_support = (
            float(self.config["outage_grid_support"])
            if in_shock
            else float(self.config["normal_grid_support"])
        )
        for cell in self.cells:
            cell.payoff *= 0.45
            if not cell.alive:
                cell.demand = 0.0
                cell.served = 0.0
                cell.deficit = 0.0
                cell.surplus = 0.0
                continue
            demand = self.landuse_demand(cell.landuse, hour)
            solar = self.landuse_solar(cell.landuse, hour, step)
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
                not self.cells[idx].critical,
                -self.cells[idx].deficit,
                -self.cells[idx].reputation,
            )
        )
        radius = int(self.config["share_radius"])
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
            candidates.sort(key=lambda item: (item[0], -self.cells[item[1]].surplus))
            for dist, didx in candidates[:6]:
                if receiver.deficit <= 0.01:
                    break
                donor = self.cells[didx]
                cooperation_attempts += 1
                action = self.should_share(donor, receiver, dist)
                donor.reputation = self.assess(donor, receiver, action)
                if not action:
                    donor.payoff -= 0.018 if receiver.critical else 0.006
                    continue
                same_block_institution = (
                    donor.block_id == receiver.block_id
                    and self.block_norms[donor.block_id] == donor.norm
                    and self.block_strength[donor.block_id] > 0.35
                )
                transfer = min(donor.surplus, receiver.deficit / max(0.1, 1.0 - 0.05 * dist))
                received = transfer * (0.98 if same_block_institution else max(0.55, 1.0 - 0.05 * dist))
                donor.storage -= transfer
                donor.surplus = max(0.0, donor.storage - donor.storage_cap * 0.44)
                receiver.deficit = max(0.0, receiver.deficit - received)
                receiver.served += received
                donor.payoff += 0.052 * (2.0 if receiver.critical else 1.0) - 0.014 * transfer
                if same_block_institution:
                    donor.payoff += 0.020
                receiver.payoff += 0.07 * received / max(receiver.demand, 1e-6)
                cooperation_successes += 1

        active_cells = [cell for cell in self.cells if cell.alive]
        total_demand = sum(cell.demand for cell in active_cells)
        total_served = sum(min(cell.served, cell.demand) for cell in active_cells)
        critical_cells = [cell for cell in active_cells if cell.critical]
        critical_demand = sum(cell.demand for cell in critical_cells)
        critical_served = sum(min(cell.served, cell.demand) for cell in critical_cells)

        for cell in self.cells:
            if not cell.alive:
                continue
            unmet = cell.deficit / max(cell.demand, 1e-6)
            institution = self.block_norm(cell)
            aligned = institution == cell.norm and institution >= 0
            recovery_rate = float(self.config.get("health_recovery_rate", 0.044))
            loss_rate = float(self.config.get("health_loss_rate", 0.046))
            cell.health = max(0.0, min(1.0, cell.health + recovery_rate * (1.0 - unmet) - loss_rate * unmet))
            cell.payoff += 0.08 * (1.0 - unmet) + 0.035 * cell.reputation
            if aligned:
                cell.payoff += 0.035 * self.block_strength[cell.block_id]
            if cell.critical:
                cell.payoff += 0.05 * (1.0 - unmet)
            if cell.health <= 0.035:
                cell.alive = False
                cell.reputation = 0.25

        return {
            "served_fraction": min(1.0, total_served / max(total_demand, 1e-6)),
            "critical_service": min(1.0, critical_served / max(critical_demand, 1e-6)),
            "cooperation_rate": cooperation_successes / max(1, cooperation_attempts),
            "cooperation_attempts": cooperation_attempts,
        }

    def rebuild_norm(self, idx: int, live_neighbors: list[int]) -> int:
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
        for idx, cell in enumerate(self.cells):
            if not cell.alive:
                live_neighbors = [n for n in self.neighbor_indices(idx, 1) if self.cells[n].alive]
                if live_neighbors and self.rng.random() < rebuild_rate:
                    cell.alive = True
                    cell.norm = self.rebuild_norm(idx, live_neighbors)
                    cell.health = 0.45
                    cell.storage = cell.storage_cap * 0.30
                    cell.reputation = 0.50
                    rebuilds += 1
                continue
            block_norm = self.block_norm(cell)
            if block_norm >= 0 and cell.norm != block_norm and self.rng.random() < conformity_rate * self.block_strength[cell.block_id]:
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
        critical = [cell for cell in self.cells if cell.critical]
        critical_alive = [cell for cell in critical if cell.alive and cell.health > 0.08]
        norm_counts = [0] * len(NORMS)
        landuse_counts = [[0] * len(NORMS) for _ in LAND_USES]
        landuse_service = [0.0] * len(LAND_USES)
        landuse_n = [0] * len(LAND_USES)
        for cell in alive:
            norm_counts[cell.norm] += 1
            landuse_counts[cell.landuse][cell.norm] += 1
            landuse_service[cell.landuse] += cell.served / max(cell.demand, 1e-6)
            landuse_n[cell.landuse] += 1
        hierarchy_count, hierarchy_coverage, hierarchy_norms = self.hierarchy_metrics()
        return {
            "step": step,
            "alive_fraction": len(alive) / len(self.cells),
            "critical_survival": len(critical_alive) / max(1, len(critical)),
            "critical_fraction": len(critical) / len(self.cells),
            "served_fraction": stats["served_fraction"],
            "critical_service": stats["critical_service"],
            "cooperation_rate": stats["cooperation_rate"],
            "cooperation_attempts": stats["cooperation_attempts"],
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
        }

    def step(self, step: int) -> dict[str, Any]:
        stats: dict[str, float] = self.energy_step(step)
        stats.update(self.evolution_step())
        stats.update(self.hierarchy_step())
        return self.metrics(step, stats)

    def run(self) -> SimulationResult:
        frames = []
        contact = []
        metrics = []
        steps = int(self.config["steps"])
        for step in range(steps):
            row = self.step(step)
            metrics.append(row)
            if step % 2 == 0 or step == steps - 1:
                image = render_state(self.cells, self.landuse_map, self.size, row)
                frames.append(image)
                if len(contact) < 12 and step % max(1, steps // 12) == 0:
                    contact.append(image)
        return SimulationResult(frames, contact, metrics, self.cells, self.data)


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


def render_state(cells: list[Cell], landuse_map: list[int], size: int, metric: dict[str, Any]) -> Image.Image:
    scale = 8
    margin = 18
    gap = 16
    panel = size * scale
    footer = 64
    width = panel * 3 + gap * 2 + margin * 2
    height = panel + margin * 2 + footer
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    titles = ["land-use attribute", "individual norm + hierarchy", "service + health"]
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
            if cell.critical and pidx != 0:
                draw.rectangle(
                    [x + 2, y + 2, x + scale - 3, y + scale - 3],
                    outline=(255, 255, 255),
                )
        if pidx == 1:
            render_hierarchy_overlay(draw, x0, y0, scale, size, metric)

    norm_freq = metric["norm_frequencies"]
    top_norm = max(range(len(norm_freq)), key=lambda idx: norm_freq[idx])
    footer_text = (
        f"step={metric['step']:03d} served={metric['served_fraction']:.2f} "
        f"critical={metric['critical_survival']:.2f} coop={metric['cooperation_rate']:.2f} "
        f"hier={metric['hierarchy_coverage']:.2f} align={metric['hierarchy_alignment']:.2f} "
        f"top_norm={NORMS[top_norm]['key']} {norm_freq[top_norm]:.2f}"
    )
    draw.text((margin, height - footer + 16), footer_text, fill=(15, 23, 42))
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
            critical_total = 0
            critical_alive = 0
            for cell in alive:
                norm_counts[cell.norm] += 1
                landuse_counts[cell.landuse] += 1
                service += cell.served / max(cell.demand, 1e-6)
                if cell.critical:
                    critical_total += 1
                    if cell.health > 0.08:
                        critical_alive += 1
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
                    "critical_survival": critical_alive / max(1, critical_total),
                    "critical_count": critical_total,
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


def write_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "step",
            "alive_fraction",
            "critical_survival",
            "critical_fraction",
            "served_fraction",
            "critical_service",
            "cooperation_rate",
            "cooperation_attempts",
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
        ] + [f"norm_{norm['key']}" for norm in NORMS] + [f"hierarchy_{norm['key']}" for norm in NORMS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in metrics:
            flat = {field: row[field] for field in fields[:18]}
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
    draw.text((36, 24), "Land-use conditioned norm hierarchy metrics", fill=(15, 23, 42))
    resilience_series = [
        ("alive", "active buildings", (15, 23, 42), [row["alive_fraction"] for row in metrics]),
        ("served", "load served", (22, 163, 74), [row["served_fraction"] for row in metrics]),
        ("critical", "critical survival", (220, 38, 38), [row["critical_survival"] for row in metrics]),
        ("hier", "hierarchy coverage", (147, 51, 234), [row["hierarchy_coverage"] for row in metrics]),
        ("align", "norm hierarchy alignment", (14, 165, 233), [row["hierarchy_alignment"] for row in metrics]),
    ]
    draw.text((72, 70), "system and hierarchy", fill=(15, 23, 42))
    draw_series_panel(draw, metrics, 72, 96, 1000, 250, resilience_series)

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
  <title>Land-Use Norm Evolution</title>
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
    <h1>Land-Use Conditioned Norm Hierarchy</h1>
    <p>
      Land use is a fixed cell attribute that shapes demand and generation.
      Buildings adapt norms quickly. Rebuild is slow. Block-level norms can
      emerge from local individual norm success and then bias future behavior.
    </p>
    <div class="panel">
      <strong>Final summary:</strong>
      served={summary['served_fraction']:.2f},
      critical survival={summary['critical_survival']:.2f},
      hierarchy coverage={summary['hierarchy_coverage']:.2f},
      alignment={summary['hierarchy_alignment']:.2f},
      top norm={summary['top_norm']}.
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
    <div class="panel">
      <h2>Hierarchy Rule Map</h2>
      <img src="landuse_norm_hierarchy_map.png" alt="annotated hierarchy rule map">
    </div>
    <div class="panel">
      <h2>Hierarchy Canopy</h2>
      <img src="landuse_norm_hierarchy_canopy.png" alt="2D hierarchy canopy">
    </div>
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
