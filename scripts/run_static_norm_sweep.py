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
    "alive_buildings_percent",
    "resilience_normalized",
]


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
    norm = NORMS[[item["key"] for item in NORMS].index(norm_key)]
    return {
        "norm": norm_key,
        "name": norm["name"],
        "alive_buildings_percent": 100.0 * final["alive_fraction"],
        "resilience_normalized": normalized_resilience_auc(result.metrics),
    }


def write_summary_png(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 980, 620
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 26), "Static shared-storage ABM: fixed norm comparison", fill=(15, 23, 42))
    draw.text((36, 52), "Evaluation metrics: alive buildings (%) and normalized resilience AUC.", fill=(71, 85, 105))

    chart_x, chart_y = 54, 104
    chart_w, chart_h = 520, 340
    draw.rectangle([chart_x, chart_y, chart_x + chart_w, chart_y + chart_h], fill=(255, 255, 255), outline=(203, 213, 225))
    for tick in range(6):
        y = chart_y + chart_h - round(chart_h * tick / 5)
        draw.line([chart_x, y, chart_x + chart_w, y], fill=(226, 232, 240))
        draw.text((chart_x - 40, y - 6), f"{tick / 5:.1f}", fill=(71, 85, 105))

    bar_group = chart_w / len(rows)
    for i, row in enumerate(rows):
        x = chart_x + i * bar_group + 44
        bw = max(28, int(bar_group * 0.20))
        values = [
            ("alive_buildings_percent", (15, 23, 42), 0.01),
            ("resilience_normalized", (37, 99, 235), 1.0),
        ]
        for j, (key, color, scale) in enumerate(values):
            value = max(0.0, min(1.0, float(row[key]) * scale))
            h = round(chart_h * value)
            bx = int(x + j * (bw + 10))
            draw.rectangle([bx, chart_y + chart_h - h, bx + bw, chart_y + chart_h], fill=color)
        draw.text((int(x), chart_y + chart_h + 10), row["name"], fill=(15, 23, 42))

    legend_x, legend_y = chart_x + 18, chart_y + chart_h + 42
    for label, color in [
        ("alive buildings (%) / 100", (15, 23, 42)),
        ("resilience AUC", (37, 99, 235)),
    ]:
        draw.rectangle([legend_x, legend_y + 4, legend_x + 15, legend_y + 16], fill=color)
        draw.text((legend_x + 22, legend_y), label, fill=(71, 85, 105))
        legend_x += 250

    table_x, table_y = 632, 104
    headers = ["norm", "alive %", "resilience"]
    col_w = [88, 96, 110]
    y = table_y
    x = table_x
    for header, w in zip(headers, col_w):
        draw.text((x, y), header, fill=(15, 23, 42))
        x += w
    y += 26
    for row in rows:
        vals = [
            row["name"],
            f"{row['alive_buildings_percent']:.1f}",
            f"{row['resilience_normalized']:.3f}",
        ]
        x = table_x
        for val, w in zip(vals, col_w):
            draw.text((x, y), val, fill=(51, 65, 85))
            x += w
        y += 24

    draw.text((36, height - 44), "Resilience = area under Q(t) using alive-building fraction as Q(t), normalized by full-duration baseline.", fill=(71, 85, 105))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static fixed-norm ABM baselines for selfish and generous rules.")
    parser.add_argument("--config", default="configs/static-shared-pool-annual-no-grid.json")
    parser.add_argument("--out-dir", default="results/static_shared_pool_annual_sweep")
    parser.add_argument("--norms", default="SELF,GEN")
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
