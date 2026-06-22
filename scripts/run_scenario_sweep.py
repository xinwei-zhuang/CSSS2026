from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import (  # noqa: E402
    NORMS,
    LandUseNormSimulation,
    load_data,
    render_state,
)


SHOCK_LEVELS = [
    {
        "name": "mild",
        "solar_shock_factor": 0.62,
        "shock_end": 108,
    },
    {
        "name": "base",
        "solar_shock_factor": 0.38,
        "shock_end": 120,
    },
    {
        "name": "severe",
        "solar_shock_factor": 0.18,
        "shock_end": 144,
    },
]

SHARING_LEVELS = [
    {"name": "local", "share_radius": 1},
    {"name": "near", "share_radius": 2},
    {"name": "wide", "share_radius": 3},
]

REBUILD_LEVELS = [
    {"name": "slow", "rebuild_rate": 0.006},
    {"name": "fast", "rebuild_rate": 0.030},
]

SEEDS = [42, 123]


def norm_entropy(frequencies: list[float]) -> float:
    entropy = 0.0
    for value in frequencies:
        if value > 0:
            entropy -= value * math.log(value)
    return entropy / math.log(len(frequencies))


def classify(row: dict[str, Any]) -> str:
    if row["final_alive"] < 0.35 or row["final_critical"] < 0.35:
        return "collapse"
    if row["final_alive"] < 0.75 or row["final_critical"] < 0.75:
        return "partial recovery"
    if row["final_hierarchy_coverage"] >= 0.65 and row["final_hierarchy_alignment"] >= 0.60:
        return "hierarchical norm order"
    if row["top_norm_freq"] >= 0.28:
        return "individual norm dominance"
    if row["last72_alive_range"] <= 0.03 and row["final_critical"] >= 0.90:
        return "stable mixed recovery"
    return "recovering mixed state"


def run_one(config: dict[str, Any], data: Any) -> tuple[list[dict[str, Any]], LandUseNormSimulation]:
    sim = LandUseNormSimulation(config, data)
    metrics = []
    for step in range(int(config["steps"])):
        metrics.append(sim.step(step))
    return metrics, sim


def summarize(
    scenario_id: str,
    labels: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    final = metrics[-1]
    norm_freq = final["norm_frequencies"]
    top_idx = max(range(len(NORMS)), key=lambda idx: norm_freq[idx])
    last72 = metrics[-72:] if len(metrics) >= 72 else metrics
    during_shock = [
        row
        for row in metrics
        if labels["shock_start"] <= row["step"] < labels["shock_end"]
    ]
    row = {
        "scenario_id": scenario_id,
        **labels,
        "final_alive": final["alive_fraction"],
        "final_served": final["served_fraction"],
        "final_critical": final["critical_survival"],
        "final_cooperation": final["cooperation_rate"],
        "min_alive": min(row["alive_fraction"] for row in metrics),
        "min_served": min(row["served_fraction"] for row in metrics),
        "min_critical": min(row["critical_survival"] for row in metrics),
        "shock_min_served": min(row["served_fraction"] for row in during_shock),
        "shock_mean_cooperation": mean(row["cooperation_rate"] for row in during_shock),
        "last72_alive_mean": mean(row["alive_fraction"] for row in last72),
        "last72_alive_range": max(row["alive_fraction"] for row in last72)
        - min(row["alive_fraction"] for row in last72),
        "last72_served_range": max(row["served_fraction"] for row in last72)
        - min(row["served_fraction"] for row in last72),
        "final_hierarchy_count": final.get("hierarchy_count", 0),
        "final_hierarchy_coverage": final.get("hierarchy_coverage", 0.0),
        "final_hierarchy_alignment": final.get("hierarchy_alignment", 0.0),
        "hierarchy_births_total": sum(row.get("hierarchy_births", 0) for row in metrics),
        "hierarchy_switches_total": sum(row.get("hierarchy_switches", 0) for row in metrics),
        "hierarchy_dissolutions_total": sum(row.get("hierarchy_dissolutions", 0) for row in metrics),
        "hierarchy_adoptions_total": sum(row.get("hierarchy_adoptions", 0) for row in metrics),
        "building_rebuilds_total": sum(row.get("building_rebuilds", 0) for row in metrics),
        "top_norm": NORMS[top_idx]["key"],
        "top_norm_name": NORMS[top_idx]["name"],
        "top_norm_freq": norm_freq[top_idx],
        "norm_entropy": norm_entropy(norm_freq),
    }
    for idx, norm in enumerate(NORMS):
        row[f"norm_{norm['key']}"] = norm_freq[idx]
        row[f"hierarchy_{norm['key']}"] = final["hierarchy_norm_frequencies"][idx]
    row["outcome"] = classify(row)
    return row


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value_color(value: float, low: tuple[int, int, int], high: tuple[int, int, int]) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    return tuple(round(low[i] + (high[i] - low[i]) * value) for i in range(3))


def draw_axis(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((x0, y0 - 22), title, fill=(15, 23, 42))
    for tick in range(6):
        y = y1 - round((y1 - y0) * tick / 5)
        draw.line([x0, y, x1, y], fill=(226, 232, 240))
        draw.text((x0 - 34, y - 5), f"{tick / 5:.1f}", fill=(71, 85, 105))


def draw_series(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    x_field: str,
    y_field: str,
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    xs = [float(row[x_field]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    span = max(1e-9, xmax - xmin)
    points = []
    for row in rows:
        x = x0 + round((x1 - x0) * (float(row[x_field]) - xmin) / span)
        y = y1 - round((y1 - y0) * max(0.0, min(1.0, float(row[y_field]))))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=3)


def write_summary_png(rows: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (1280, 840), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 28), "Scenario sweep summary", fill=(15, 23, 42))

    sorted_by_critical = sorted(rows, key=lambda row: row["final_critical"], reverse=True)
    top = sorted_by_critical[:8]
    bottom = sorted_by_critical[-8:]

    x0, y0 = 44, 84
    bar_w, bar_h = 540, 24
    draw.text((x0, y0 - 28), "best final critical survival", fill=(15, 23, 42))
    for idx, row in enumerate(top):
        y = y0 + idx * 34
        width = round(bar_w * row["final_critical"])
        draw.rectangle([x0, y, x0 + bar_w, y + bar_h], fill=(226, 232, 240))
        draw.rectangle([x0, y, x0 + width, y + bar_h], fill=(22, 163, 74))
        draw.text(
            (x0 + bar_w + 10, y + 4),
            f"{row['scenario_id']} critical={row['final_critical']:.2f} alive={row['final_alive']:.2f}",
            fill=(15, 23, 42),
        )

    x0b, y0b = 44, 420
    draw.text((x0b, y0b - 28), "worst final critical survival", fill=(15, 23, 42))
    for idx, row in enumerate(bottom):
        y = y0b + idx * 34
        width = round(bar_w * row["final_critical"])
        draw.rectangle([x0b, y, x0b + bar_w, y + bar_h], fill=(226, 232, 240))
        draw.rectangle([x0b, y, x0b + width, y + bar_h], fill=(220, 38, 38))
        draw.text(
            (x0b + bar_w + 10, y + 4),
            f"{row['scenario_id']} critical={row['final_critical']:.2f} alive={row['final_alive']:.2f}",
            fill=(15, 23, 42),
        )

    scatter = (790, 94, 1210, 390)
    draw_axis(draw, scatter, "shock severity vs final critical survival")
    for row in rows:
        severity = {
            "mild": 0.15,
            "base": 0.50,
            "severe": 0.85,
        }[row["shock"]]
        x = scatter[0] + round((scatter[2] - scatter[0]) * severity)
        y = scatter[3] - round((scatter[3] - scatter[1]) * row["final_critical"])
        radius = 5 + int(row["share_radius"])
        color = value_color(row["final_alive"], (220, 38, 38), (22, 163, 74))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(15, 23, 42))
    draw.text((790, 400), "x: mild/base/severe; dot size: radius; color: final alive", fill=(71, 85, 105))

    scatter2 = (790, 520, 1210, 760)
    draw_axis(draw, scatter2, "hierarchy coverage vs final critical survival")
    for row in rows:
        x = scatter2[0] + round((scatter2[2] - scatter2[0]) * row["final_hierarchy_coverage"])
        y = scatter2[3] - round((scatter2[3] - scatter2[1]) * row["final_critical"])
        color = value_color(row["final_hierarchy_alignment"], (14, 165, 233), (234, 88, 12))
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline=(15, 23, 42))
    draw.text((790, 770), "x: hierarchy coverage; color: hierarchy alignment", fill=(71, 85, 105))

    image.save(path)


def write_outcome_map(rows: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (1280, 720), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((36, 24), "Outcome map: mean final critical survival", fill=(15, 23, 42))
    cell_w, cell_h = 170, 92
    start_x, start_y = 170, 95
    for sx, sharing in enumerate(SHARING_LEVELS):
        draw.text((start_x + sx * cell_w, start_y - 32), f"radius={sharing['share_radius']}", fill=(15, 23, 42))
    for ry, shock in enumerate(SHOCK_LEVELS):
        draw.text((36, start_y + ry * cell_h + 32), shock["name"], fill=(15, 23, 42))
        for sx, sharing in enumerate(SHARING_LEVELS):
            subset = [
                row
                for row in rows
                if row["shock"] == shock["name"] and row["sharing"] == sharing["name"]
            ]
            value = mean(row["final_critical"] for row in subset)
            alive = mean(row["final_alive"] for row in subset)
            hierarchy = mean(row["final_hierarchy_coverage"] for row in subset)
            color = value_color(value, (248, 113, 113), (34, 197, 94))
            x = start_x + sx * cell_w
            y = start_y + ry * cell_h
            draw.rectangle([x, y, x + cell_w - 10, y + cell_h - 12], fill=color, outline=(15, 23, 42))
            draw.text((x + 10, y + 12), f"critical {value:.2f}", fill=(15, 23, 42))
            draw.text((x + 10, y + 34), f"alive {alive:.2f}", fill=(15, 23, 42))
            draw.text((x + 10, y + 56), f"hierarchy {hierarchy:.2f}", fill=(15, 23, 42))

    outcome_counts: dict[str, int] = {}
    for row in rows:
        outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1
    y = 440
    draw.text((36, y), "Outcome counts", fill=(15, 23, 42))
    y += 34
    for outcome, count in sorted(outcome_counts.items()):
        draw.text((56, y), f"{outcome}: {count}", fill=(15, 23, 42))
        y += 26
    image.save(path)


def make_snapshot_sheet(selected: list[tuple[dict[str, Any], Image.Image]], path: Path) -> None:
    columns = 2
    thumb_w, thumb_h = 576, 272
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), (248, 250, 252))
    for idx, (_, image) in enumerate(selected):
        sheet.paste(image.resize((thumb_w, thumb_h), Image.Resampling.NEAREST), ((idx % columns) * thumb_w, (idx // columns) * thumb_h))
    sheet.save(path)


def select_representatives(rows: list[dict[str, Any]]) -> list[str]:
    selectors = [
        max(rows, key=lambda row: row["final_critical"]),
        min(rows, key=lambda row: row["final_critical"]),
        max(rows, key=lambda row: row["norm_entropy"]),
        max(rows, key=lambda row: row["top_norm_freq"]),
        max(rows, key=lambda row: row["last72_alive_range"]),
        min(rows, key=lambda row: row["shock_min_served"]),
    ]
    scenario_ids = []
    for row in selectors:
        if row["scenario_id"] not in scenario_ids:
            scenario_ids.append(row["scenario_id"])
    return scenario_ids[:6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/landuse-norm-final.json")
    parser.add_argument("--steps", type=int, default=336)
    parser.add_argument("--out-dir", default="results/scenario_sweep")
    args = parser.parse_args()

    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base_config["steps"] = args.steps
    base_config["out_dir"] = args.out_dir
    data = load_data(base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("scenario_sweep_*")) + list(out_dir.glob("*_snapshot.png")):
        stale.unlink()

    rows = []
    snapshots: dict[str, Image.Image] = {}
    total = len(SHOCK_LEVELS) * len(SHARING_LEVELS) * len(REBUILD_LEVELS) * len(SEEDS)
    completed = 0
    for shock in SHOCK_LEVELS:
        for sharing in SHARING_LEVELS:
            for rebuild in REBUILD_LEVELS:
                for seed in SEEDS:
                    completed += 1
                    scenario_id = f"{shock['name']}_{sharing['name']}_{rebuild['name']}_s{seed}"
                    config = deepcopy(base_config)
                    config.update(shock)
                    config.update(sharing)
                    config.update(rebuild)
                    config["seed"] = seed
                    metrics, sim = run_one(config, data)
                    labels = {
                        "shock": shock["name"],
                        "sharing": sharing["name"],
                        "rebuild": rebuild["name"],
                        "seed": seed,
                        "share_radius": sharing["share_radius"],
                        "rebuild_rate": rebuild["rebuild_rate"],
                        "solar_shock_factor": shock["solar_shock_factor"],
                        "shock_start": config["shock_start"],
                        "shock_end": shock["shock_end"],
                    }
                    row = summarize(scenario_id, labels, metrics)
                    rows.append(row)
                    snapshots[scenario_id] = render_state(sim.cells, sim.landuse_map, sim.size, metrics[-1])
                    print(f"[{completed:02d}/{total}] {scenario_id} {row['outcome']} critical={row['final_critical']:.2f} alive={row['final_alive']:.2f}")

    rows.sort(key=lambda row: (row["shock"], row["sharing"], row["rebuild"], row["seed"]))
    write_rows_csv(rows, out_dir / "scenario_sweep_results.csv")
    (out_dir / "scenario_sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_summary_png(rows, out_dir / "scenario_sweep_summary.png")
    write_outcome_map(rows, out_dir / "scenario_sweep_outcome_map.png")

    selected_ids = select_representatives(rows)
    selected = [(next(row for row in rows if row["scenario_id"] == sid), snapshots[sid]) for sid in selected_ids]
    for row, image in selected:
        image.save(out_dir / f"{row['scenario_id']}_snapshot.png")
    make_snapshot_sheet(selected, out_dir / "scenario_sweep_representative_snapshots.png")

    best = max(rows, key=lambda row: row["final_critical"])
    worst = min(rows, key=lambda row: row["final_critical"])
    print("out_dir=" + str(out_dir.resolve()))
    print(f"best={best['scenario_id']} critical={best['final_critical']:.3f} alive={best['final_alive']:.3f} outcome={best['outcome']}")
    print(f"worst={worst['scenario_id']} critical={worst['final_critical']:.3f} alive={worst['final_alive']:.3f} outcome={worst['outcome']}")


if __name__ == "__main__":
    main()
