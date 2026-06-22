from __future__ import annotations

import csv
from dataclasses import dataclass
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


@dataclass
class CityEnergyProfiles:
    demand: torch.Tensor
    solar_potential: torch.Tensor
    storage_support: torch.Tensor
    critical_weight: torch.Tensor
    source: str
    residential_profiles: int
    commercial_profiles: int

    @property
    def period(self) -> int:
        return int(self.demand.shape[1])

    def demand_at(self, step: int) -> torch.Tensor:
        return self.demand[:, step % self.period]

    def service_ratio(
        self,
        step: int,
        daylight: float,
        solar_scale: float,
    ) -> torch.Tensor:
        demand = self.demand_at(step)
        supply = self.solar_potential * daylight * solar_scale + self.storage_support
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
        solar_potential = [1.38, 0.62, 0.94]
        storage_support = [0.42, 0.34, 1.18]
        critical_weight = [0.15, 0.35, 1.00]

        demand_rows = []
        solar_rows = []
        storage_rows = []
        critical_rows = []
        for idx in range(n_ncas):
            k = idx % 3
            demand_rows.append([v * demand_scales[k] for v in base_curves[k]])
            solar_rows.append(solar_potential[k])
            storage_rows.append(storage_support[k])
            critical_rows.append(critical_weight[k])

        source = str(csv_path) if csv_path else "synthetic"
        return cls(
            demand=torch.tensor(demand_rows, device=device, dtype=dtype),
            solar_potential=torch.tensor(solar_rows, device=device, dtype=dtype),
            storage_support=torch.tensor(storage_rows, device=device, dtype=dtype),
            critical_weight=torch.tensor(critical_rows, device=device, dtype=dtype),
            source=source,
            residential_profiles=len(residential),
            commercial_profiles=len(commercial),
        )
