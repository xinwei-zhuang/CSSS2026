from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import NORMS, LandUseNormSimulation, load_data, save_result


SUMMARY_FIELDS = [
    "norm",
    "name",
    "alive_fraction",
    "critical_survival",
    "annual_mean_served_fraction",
    "annual_mean_critical_service",
    "final_served_fraction",
    "final_critical_service",
    "mean_stress_memory",
    "critical_stress_memory",
    "max_stress_memory",
    "annual_cooperation_attempts",
    "annual_cooperation_successes",
    "final_cooperation_attempts",
    "final_cooperation_successes",
    "mean_pool_count",
    "mean_pool_members",
    "converged",
    "last_step",
]


def run_one(base_config: dict[str, Any], norm_key: str, out_root: Path) -> dict[str, Any]:
    config = dict(base_config)
    config["fixed_norm_key"] = norm_key
    config["enable_norm_evolution"] = False
    config["enable_hierarchy"] = False
    config["enable_rebuild"] = False
    config["out_dir"] = str(out_root / norm_key.lower())

    data = load_data(config)
    result = LandUseNormSimulation(config, data).run()
    save_result(result, Path(config["out_dir"]))

    final = result.metrics[-1]
    tail = result.metrics[-min(168, len(result.metrics)) :]
    annual_mean_served = sum(row["served_fraction"] for row in result.metrics) / max(1, len(result.metrics))
    annual_mean_critical = sum(row["critical_service"] for row in result.metrics) / max(1, len(result.metrics))
    annual_attempts = sum(row["cooperation_attempts"] for row in result.metrics)
    annual_successes = sum(row["cooperation_successes"] for row in result.metrics)
    norm = NORMS[[item["key"] for item in NORMS].index(norm_key)]
    return {
        "norm": norm_key,
        "name": norm["name"],
        "alive_fraction": final["alive_fraction"],
        "critical_survival": final["critical_survival"],
        "annual_mean_served_fraction": annual_mean_served,
        "annual_mean_critical_service": annual_mean_critical,
        "final_served_fraction": final["served_fraction"],
        "final_critical_service": final["critical_service"],
        "mean_stress_memory": final.get("mean_stress_memory", 0.0),
        "critical_stress_memory": final.get("critical_stress_memory", 0.0),
        "max_stress_memory": final.get("max_stress_memory", 0.0),
        "annual_cooperation_attempts": annual_attempts,
        "annual_cooperation_successes": annual_successes,
        "final_cooperation_attempts": final["cooperation_attempts"],
        "final_cooperation_successes": final["cooperation_successes"],
        "mean_pool_count": sum(row.get("pool_count", 0) for row in tail) / max(1, len(tail)),
        "mean_pool_members": sum(row.get("pool_members", 0) for row in tail) / max(1, len(tail)),
        "converged": final["converged"],
        "last_step": final["step"],
    }


def write_summary_png(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1320, 760
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 26), "Static shared-storage ABM: 8 fixed norms", fill=(15, 23, 42))
    draw.text((36, 52), "No external grid, annual load/solar series, local shared storage, no evolution/rebuild.", fill=(71, 85, 105))

    chart_x, chart_y = 54, 104
    chart_w, chart_h = 600, 360
    draw.rectangle([chart_x, chart_y, chart_x + chart_w, chart_y + chart_h], fill=(255, 255, 255), outline=(203, 213, 225))
    for tick in range(6):
        y = chart_y + chart_h - round(chart_h * tick / 5)
        draw.line([chart_x, y, chart_x + chart_w, y], fill=(226, 232, 240))
        draw.text((chart_x - 34, y - 6), f"{tick / 5:.1f}", fill=(71, 85, 105))

    bar_group = chart_w / len(rows)
    for i, row in enumerate(rows):
        x = chart_x + i * bar_group + 18
        bw = max(14, int(bar_group * 0.22))
        values = [
            ("alive_fraction", (15, 23, 42)),
            ("critical_survival", (220, 38, 38)),
            ("annual_mean_served_fraction", (22, 163, 74)),
        ]
        for j, (key, color) in enumerate(values):
            value = max(0.0, min(1.0, float(row[key])))
            h = round(chart_h * value)
            bx = int(x + j * (bw + 4))
            draw.rectangle([bx, chart_y + chart_h - h, bx + bw, chart_y + chart_h], fill=color)
        draw.text((int(x), chart_y + chart_h + 10), row["norm"], fill=(15, 23, 42))

    legend_x, legend_y = chart_x + 18, chart_y + chart_h + 42
    for label, color in [
        ("alive buildings", (15, 23, 42)),
        ("critical survival", (220, 38, 38)),
        ("served load", (22, 163, 74)),
    ]:
        draw.rectangle([legend_x, legend_y + 4, legend_x + 15, legend_y + 16], fill=color)
        draw.text((legend_x + 22, legend_y), label, fill=(71, 85, 105))
        legend_x += 170

    table_x, table_y = 704, 104
    headers = ["norm", "alive", "critical", "stress", "pool", "success"]
    col_w = [70, 82, 82, 82, 82, 94]
    y = table_y
    x = table_x
    for header, w in zip(headers, col_w):
        draw.text((x, y), header, fill=(15, 23, 42))
        x += w
    y += 26
    for row in rows:
        vals = [
            row["norm"],
            f"{row['alive_fraction']:.3f}",
            f"{row['critical_survival']:.3f}",
            f"{row['mean_stress_memory']:.3f}",
            f"{row['mean_pool_members']:.1f}",
            str(int(row["annual_cooperation_successes"])),
        ]
        x = table_x
        for val, w in zip(vals, col_w):
            draw.text((x, y), val, fill=(51, 65, 85))
            x += w
        y += 24

    draw.text((36, height - 44), "Resilience is reported as critical-load survival plus service and stress-memory diagnostics.", fill=(71, 85, 105))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static fixed-norm ABM baselines for the eight rule types.")
    parser.add_argument("--config", default="configs/static-shared-pool-annual-no-grid.json")
    parser.add_argument("--out-dir", default="results/static_shared_pool_annual_sweep")
    parser.add_argument("--norms", default=",".join(norm["key"] for norm in NORMS))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    requested = [item.strip().upper() for item in args.norms.split(",") if item.strip()]
    valid = {norm["key"] for norm in NORMS}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"Unknown norms: {', '.join(unknown)}")

    rows = [run_one(base_config, norm_key, out_root) for norm_key in requested]
    csv_path = out_root / "static_shared_pool_annual_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "static_shared_pool_annual_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary_png(rows, out_root / "static_shared_pool_annual_summary.png")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
