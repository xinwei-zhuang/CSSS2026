from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import torch


def _synthetic_daily_curve(kind: str) -> list[float]:
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
    mean = sum(raw) / len(raw)
    return [v / mean for v in raw]


def _normalize_curve(values: list[float]) -> list[float]:
    mean = sum(values) / max(1, len(values))
    if mean <= 0:
        return [1.0 for _ in values]
    return [max(0.02, v / mean) for v in values]


def _hourly_to_daily(values: list[float]) -> list[float]:
    totals = [0.0] * 24
    counts = [0] * 24
    for idx, value in enumerate(values):
        hour = idx % 24
        totals[hour] += value
        counts[hour] += 1
    daily = [totals[i] / max(1, counts[i]) for i in range(24)]
    return _normalize_curve(daily)


def _average_curves(curves: list[list[float]], fallback_kind: str) -> list[float]:
    if not curves:
        return _synthetic_daily_curve(fallback_kind)
    return _normalize_curve(
        [sum(curve[hour] for curve in curves) / len(curves) for hour in range(24)]
    )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_roof_area_m2(
    metadata_csv: str,
    sample_size: int,
    usable_fraction: float,
) -> dict[str, float]:
    path = Path(metadata_csv) if metadata_csv else None
    fallback_roof = {"residential": 125.0, "commercial": 620.0}
    if not path or not path.exists():
        return fallback_roof

    roof_sums = {"residential": 0.0, "commercial": 0.0}
    counts = {"residential": 0, "commercial": 0}
    target_each = max(1, sample_size // 2)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile_type = (row.get("profile_type") or "").lower()
            if profile_type not in roof_sums or counts[profile_type] >= target_each:
                continue
            sqft = _safe_float(row.get("bldgsqft"), 0.0)
            floors = max(1.0, _safe_float(row.get("floor"), 1.0))
            if sqft <= 0:
                continue
            roof_m2 = sqft / floors * 0.092903 * usable_fraction
            if roof_m2 <= 0:
                continue
            roof_sums[profile_type] += roof_m2
            counts[profile_type] += 1
            if counts["residential"] >= target_each and counts["commercial"] >= target_each:
                break

    return {
        kind: roof_sums[kind] / counts[kind] if counts[kind] else fallback_roof[kind]
        for kind in fallback_roof
    }


def _synthetic_solar_generation() -> list[float]:
    raw = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.05,
        0.22,
        0.48,
        0.74,
        0.94,
        1.08,
        1.14,
        1.12,
        1.02,
        0.86,
        0.66,
        0.42,
        0.18,
        0.04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    mean_positive = sum(raw) / 24
    return [v / mean_positive if mean_positive else 0.0 for v in raw]


def _read_epw_24h(
    epw_path: str,
    month: int,
    day: int,
) -> tuple[list[float], list[float], str]:
    path = Path(epw_path) if epw_path else None
    if not path or not path.exists():
        return _synthetic_solar_generation(), [15.0] * 24, "synthetic"

    hourly_ghi = [[] for _ in range(24)]
    hourly_temp = [[] for _ in range(24)]
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for _ in range(8):
            next(reader, None)
        for row in reader:
            if len(row) < 16:
                continue
            row_month = int(_safe_float(row[1], 0))
            row_day = int(_safe_float(row[2], 0))
            if month > 0 and day > 0 and (row_month != month or row_day != day):
                continue
            # EPW hours are 1-24 and represent the preceding interval.
            hour = int(_safe_float(row[3], 1)) - 1
            hour = max(0, min(23, hour))
            temp_air = _safe_float(row[6], 15.0)
            ghi = max(0.0, _safe_float(row[13], 0.0))
            hourly_ghi[hour].append(ghi)
            hourly_temp[hour].append(temp_air)

    ghi = [sum(v) / len(v) if v else 0.0 for v in hourly_ghi]
    temp = [sum(v) / len(v) if v else 15.0 for v in hourly_temp]
    if not any(ghi):
        return _synthetic_solar_generation(), temp, str(path)
    return ghi, temp, str(path)


def _pv_per_m2_kw(
    ghi: list[float],
    temp_air: list[float],
    eta_ref: float,
) -> list[float]:
    # Same simplified parameters used by the reference notebook.
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
        out.append(max(0.0, rt * eta_ref * (temp_factor + irradiance_factor) / 1000.0))
    return out


def _build_solar_generation_rows(
    config,
    n_ncas: int,
) -> tuple[list[list[float]], str, dict[str, float]]:
    ghi, temp, source = _read_epw_24h(
        config.city_solar_epw,
        int(config.city_solar_month),
        int(config.city_solar_day),
    )
    pv_per_m2 = _pv_per_m2_kw(ghi, temp, float(config.city_pv_efficiency))
    roof_area = _estimate_roof_area_m2(
        config.city_building_metadata_csv,
        max(0, int(config.city_profile_sample_size)),
        max(0.0, float(config.city_roof_usable_fraction)),
    )
    type_area = [
        roof_area["residential"],
        roof_area["commercial"],
        (roof_area["residential"] + roof_area["commercial"]) * 0.5,
    ]
    # Keep training numerically gentle while preserving real climate shape and
    # roof-area differences. The model still sees dimensionless daily curves,
    # but large roofs keep their neighborhood-scale generation advantage.
    raw_rows = [
        [pv * area for pv in pv_per_m2]
        for area in type_area
    ]
    reference = sum(sum(row) / len(row) for row in raw_rows) / len(raw_rows)
    if reference <= 0:
        base = _synthetic_solar_generation()
        raw_rows = [[v for v in base] for _ in range(3)]
        reference = 1.0
    normalized_rows = [[v / reference for v in row] for row in raw_rows]

    solar_rows = []
    for idx in range(n_ncas):
        solar_rows.append(normalized_rows[idx % 3])
    return solar_rows, source, roof_area


@dataclass
class CityEnergyProfiles:
    demand: torch.Tensor
    solar_generation: torch.Tensor
    storage_support: torch.Tensor
    critical_weight: torch.Tensor
    source: str
    solar_source: str
    roof_area_m2: dict[str, float]
    residential_profiles: int
    commercial_profiles: int

    @property
    def period(self) -> int:
        return int(self.demand.shape[1])

    def demand_at(self, step: int) -> torch.Tensor:
        return self.demand[:, step % self.period]

    def solar_at(self, step: int) -> torch.Tensor:
        return self.solar_generation[:, step % self.period]

    def daylight_at(self, step: int) -> float:
        row = self.solar_at(step)
        peak = self.solar_generation.max().clamp_min(1e-6)
        return float((row.mean() / peak).clamp(0.0, 1.0).item())

    def service_ratio(
        self,
        step: int,
        solar_scale: float,
    ) -> torch.Tensor:
        demand = self.demand_at(step)
        supply = self.solar_at(step) * solar_scale + self.storage_support
        return torch.clamp(supply / demand.clamp_min(1e-6), min=0.0, max=1.8)

    @classmethod
    def from_config(cls, config, n_ncas: int, device: str, dtype: torch.dtype):
        csv_path = Path(config.city_profiles_csv) if config.city_profiles_csv else None
        sample_size = max(0, int(config.city_profile_sample_size))
        residential: list[list[float]] = []
        commercial: list[list[float]] = []

        if csv_path and csv_path.exists() and sample_size > 0:
            target_each = max(1, sample_size // 2)
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                time_cols = [name for name in reader.fieldnames or [] if name.startswith("t")]
                for row in reader:
                    profile_type = (row.get("profile_type") or "").lower()
                    if profile_type not in {"residential", "commercial"}:
                        continue
                    if profile_type == "residential" and len(residential) >= target_each:
                        continue
                    if profile_type == "commercial" and len(commercial) >= target_each:
                        continue
                    values = []
                    for col in time_cols:
                        try:
                            values.append(float(row[col]))
                        except (KeyError, TypeError, ValueError):
                            values.append(0.0)
                    daily = _hourly_to_daily(values)
                    if profile_type == "residential":
                        residential.append(daily)
                    else:
                        commercial.append(daily)
                    if len(residential) >= target_each and len(commercial) >= target_each:
                        break

        res_curve = _average_curves(residential, "residential")
        com_curve = _average_curves(commercial, "commercial")
        storage_curve = _normalize_curve(
            [(res_curve[h] + com_curve[h]) * 0.5 for h in range(24)]
        )

        base_curves = [res_curve, com_curve, storage_curve]
        demand_scales = [0.82, 1.28, 1.00]
        storage_support = [0.42, 0.34, 1.18]
        critical_weight = [0.15, 0.35, 1.00]
        solar_rows, solar_source, roof_area = _build_solar_generation_rows(
            config,
            n_ncas,
        )

        demand_rows = []
        storage_rows = []
        critical_rows = []
        for idx in range(n_ncas):
            k = idx % 3
            demand_rows.append([v * demand_scales[k] for v in base_curves[k]])
            storage_rows.append(storage_support[k])
            critical_rows.append(critical_weight[k])

        source = str(csv_path) if csv_path else "synthetic"
        return cls(
            demand=torch.tensor(demand_rows, device=device, dtype=dtype),
            solar_generation=torch.tensor(solar_rows, device=device, dtype=dtype),
            storage_support=torch.tensor(storage_rows, device=device, dtype=dtype),
            critical_weight=torch.tensor(critical_rows, device=device, dtype=dtype),
            source=source,
            solar_source=solar_source,
            roof_area_m2=roof_area,
            residential_profiles=len(residential),
            commercial_profiles=len(commercial),
        )
