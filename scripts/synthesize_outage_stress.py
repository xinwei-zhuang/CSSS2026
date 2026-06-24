from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def safe_float(value: object, fallback: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def read_mean_load(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        time_cols = [name for name in reader.fieldnames or [] if name.startswith("t")]
        totals = [0.0] * len(time_cols)
        count = 0
        for row in reader:
            count += 1
            for idx, col in enumerate(time_cols):
                totals[idx] += safe_float(row.get(col), 0.0)
    if count <= 0:
        raise ValueError(f"No load profiles found in {path}")
    return [value / count for value in totals]


def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        return [0.0 for _ in values]
    return [(value - lo) / span for value in values]


def synthesize(load: list[float]) -> list[dict[str, float]]:
    n = len(load)
    load_norm = normalize(load)
    rows = [
        {
            "step": float(step),
            "load_pressure": load_norm[step],
            "outage_severity": 0.0,
            "solar_factor": 1.0,
            "grid_support": 0.0,
        }
        for step in range(n)
    ]

    period_hours = 24 * 28
    duration_hours = 12
    daytime_start_hour = 9
    for start in range(daytime_start_hour, n, period_hours):
        end = min(n, start + duration_hours)
        for step in range(start, end):
            rows[step]["outage_severity"] = 1.0
            rows[step]["solar_factor"] = 0.0
            rows[step]["grid_support"] = 0.0

    return rows


def write_profile_png(rows: list[dict[str, float]], path: Path) -> None:
    width, height = 1180, 420
    margin = 54
    chart_w = width - margin * 2
    chart_h = 260
    chart_y = 78
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), "Periodic no-solar stress profile", fill=(15, 23, 42))
    draw.text((36, 46), "Solar generation is set to zero for periodic daytime windows; grid support stays zero throughout.", fill=(71, 85, 105))
    draw.rectangle([margin, chart_y, margin + chart_w, chart_y + chart_h], fill=(255, 255, 255), outline=(203, 213, 225))
    for tick in range(6):
        y = chart_y + chart_h - round(chart_h * tick / 5)
        draw.line([margin, y, margin + chart_w, y], fill=(226, 232, 240))
        draw.text((margin - 40, y - 6), f"{tick / 5:.1f}", fill=(71, 85, 105))

    def series_points(key: str) -> list[tuple[int, int]]:
        n = max(1, len(rows) - 1)
        points = []
        for idx, row in enumerate(rows):
            x = margin + round(chart_w * idx / n)
            y = chart_y + chart_h - round(chart_h * max(0.0, min(1.0, row[key])))
            points.append((x, y))
        return points

    load_points = series_points("load_pressure")
    outage_points = series_points("outage_severity")
    if len(load_points) > 1:
        draw.line(load_points, fill=(148, 163, 184), width=2)
    if len(outage_points) > 1:
        draw.line(outage_points, fill=(220, 38, 38), width=3)

    legend_y = chart_y + chart_h + 28
    draw.rectangle([margin, legend_y + 4, margin + 16, legend_y + 16], fill=(220, 38, 38))
    draw.text((margin + 24, legend_y), "no-solar severity", fill=(15, 23, 42))
    draw.rectangle([margin + 190, legend_y + 4, margin + 206, legend_y + 16], fill=(148, 163, 184))
    draw.text((margin + 214, legend_y), "normalized load pressure", fill=(15, 23, 42))
    active = sum(1 for row in rows if row["outage_severity"] > 0)
    peak = max(row["outage_severity"] for row in rows)
    draw.text((36, height - 42), f"Hours={len(rows)}  no-solar hours={active}  peak severity={peak:.3f}", fill=(71, 85, 105))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize a periodic hourly no-solar stress profile.")
    parser.add_argument("--demand-csv", default="../data/energy_profiles_clean/energy_profiles_hourly_used.csv")
    parser.add_argument("--out", default="data/outage_profiles/periodic_no_solar_stress.csv")
    args = parser.parse_args()

    demand_path = Path(args.demand_csv)
    if not demand_path.is_absolute():
        demand_path = ROOT / demand_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = synthesize(read_mean_load(demand_path))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "load_pressure", "outage_severity", "solar_factor", "grid_support"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "step": int(row["step"]),
                "load_pressure": f"{row['load_pressure']:.6f}",
                "outage_severity": f"{row['outage_severity']:.6f}",
                "solar_factor": f"{row['solar_factor']:.6f}",
                "grid_support": f"{row['grid_support']:.6f}",
            })
    active = sum(1 for row in rows if row["outage_severity"] > 0)
    peak = max(row["outage_severity"] for row in rows)
    png_path = out_path.with_suffix(".png")
    write_profile_png(rows, png_path)
    print(f"wrote={out_path}")
    print(f"plot={png_path}")
    print(f"hours={len(rows)} no_solar_hours={active} peak_severity={peak:.3f}")


if __name__ == "__main__":
    main()
