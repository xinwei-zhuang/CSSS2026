from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import LAND_USES, LandUseNormSimulation, load_data  # noqa: E402


COLORS = {
    "demand": "#dc2626",
    "generation": "#16a34a",
    "storage": "#2563eb",
}


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    out = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        out -= p * math.log(p)
    return out / math.log(max(2, len(counts)))


def normalize_scores(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def select_interesting_window(data: Any, grid_size: int, zoom_size: int) -> tuple[int, int, dict[str, float]]:
    if not data.cell_specs:
        raise ValueError("This visualization needs use_real_building_patch=true.")
    outage = data.outage_severity
    windows: list[dict[str, float]] = []
    for row0 in range(grid_size - zoom_size + 1):
        for col0 in range(grid_size - zoom_size + 1):
            idxs = [
                (row0 + r) * grid_size + (col0 + c)
                for r in range(zoom_size)
                for c in range(zoom_size)
            ]
            total_demand = 0.0
            total_generation = 0.0
            outage_load = 0.0
            landuse_counts = [0] * len(LAND_USES)
            for idx in idxs:
                spec = data.cell_specs[idx]
                demand = spec["demand_curve"]
                generation = spec["solar_curve"]
                n = min(len(demand), len(generation), len(outage))
                total_demand += sum(demand[:n])
                total_generation += sum(generation[:n])
                outage_load += sum(demand[i] * outage[i] for i in range(n))
                landuse_counts[int(spec["landuse"])] += 1
            windows.append(
                {
                    "row0": float(row0),
                    "col0": float(col0),
                    "demand": total_demand,
                    "mismatch": abs(total_demand - total_generation),
                    "outage_load": outage_load,
                    "diversity": entropy(landuse_counts),
                }
            )
    demand_n = normalize_scores([w["demand"] for w in windows])
    mismatch_n = normalize_scores([w["mismatch"] for w in windows])
    outage_n = normalize_scores([w["outage_load"] for w in windows])
    diversity_n = normalize_scores([w["diversity"] for w in windows])
    best_i = 0
    best_score = -1.0
    for i, window in enumerate(windows):
        score = 0.34 * demand_n[i] + 0.30 * outage_n[i] + 0.22 * mismatch_n[i] + 0.14 * diversity_n[i]
        window["score"] = score
        if score > best_score:
            best_score = score
            best_i = i
    best = windows[best_i]
    return int(best["row0"]), int(best["col0"]), best


def daily_average(values: list[float]) -> list[float]:
    out = []
    for start in range(0, len(values), 24):
        chunk = values[start : start + 24]
        if chunk:
            out.append(mean(chunk))
    return out


def points(values: list[float], width: int, height: int, vmax: float) -> str:
    if not values:
        return ""
    if len(values) == 1:
        values = values * 2
    vmax = max(vmax, 1e-6)
    coords = []
    for i, value in enumerate(values):
        x = width * i / max(1, len(values) - 1)
        y = height - height * max(0.0, min(1.0, value / vmax))
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def mini_svg(demand: list[float], generation: list[float], storage: list[float]) -> str:
    width = 150
    height = 70
    vmax = max(max(demand or [0]), max(generation or [0]), max(storage or [0]), 1e-6)
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" preserveAspectRatio=\"none\">"
        "<g class=\"mini-legend\">"
        f"<text x=\"4\" y=\"9\" fill=\"{COLORS['demand']}\">D</text>"
        f"<text x=\"18\" y=\"9\" fill=\"{COLORS['generation']}\">G</text>"
        f"<text x=\"32\" y=\"9\" fill=\"{COLORS['storage']}\">S</text>"
        "</g>"
        f"<polyline class=\"line demand\" points=\"{points(demand, width, height, vmax)}\"/>"
        f"<polyline class=\"line generation\" points=\"{points(generation, width, height, vmax)}\"/>"
        f"<polyline class=\"line storage\" points=\"{points(storage, width, height, vmax)}\"/>"
        "</svg>"
    )


def run_traces(config: dict[str, Any], norm: str, row0: int, col0: int, zoom_size: int) -> tuple[Any, dict[int, dict[str, list[float]]]]:
    config = dict(config)
    config["fixed_norm_key"] = norm
    config["enable_norm_evolution"] = False
    config["enable_hierarchy"] = False
    config["enable_rebuild"] = False
    data = load_data(config)
    sim = LandUseNormSimulation(config, data)
    grid_size = int(config["grid_size"])
    idxs = [
        (row0 + r) * grid_size + (col0 + c)
        for r in range(zoom_size)
        for c in range(zoom_size)
    ]
    traces = {
        idx: {"demand": [], "generation": [], "storage": [], "health": []}
        for idx in idxs
    }
    for step in range(int(config["steps"])):
        hour = step % 24
        for idx in idxs:
            cell = sim.cells[idx]
            traces[idx]["demand"].append(sim.cell_demand(cell, step))
            traces[idx]["generation"].append(sim.cell_solar(cell, hour, step))
        sim.step(step)
        for idx in idxs:
            cell = sim.cells[idx]
            traces[idx]["storage"].append(cell.storage)
            traces[idx]["health"].append(cell.health)
    return sim, traces


def write_html(
    data: Any,
    sim: Any,
    traces: dict[int, dict[str, list[float]]],
    row0: int,
    col0: int,
    zoom_size: int,
    score: dict[str, float],
    out_path: Path,
    norm: str,
) -> None:
    grid_size = sim.size
    cards = []
    for r in range(zoom_size):
        for c in range(zoom_size):
            idx = (row0 + r) * grid_size + (col0 + c)
            cell = sim.cells[idx]
            spec = data.cell_specs[idx] if data.cell_specs else {}
            land = LAND_USES[cell.landuse]
            demand = daily_average(traces[idx]["demand"])
            generation = daily_average(traces[idx]["generation"])
            storage = daily_average(traces[idx]["storage"])
            annual_demand = sum(traces[idx]["demand"])
            annual_generation = sum(traces[idx]["generation"])
            deficit_ratio = cell.cumulative_deficit / max(cell.cumulative_demand, 1e-6)
            resilient_mark = "<em title=\"resilient: cumulative deficit <= 5%\">R</em>" if cell.resilient else ""
            title = (
                f"Building {spec.get('building_id', '')}\\n"
                f"Profile {spec.get('profile_id', '')}\\n"
                f"Lon {float(spec.get('lon', 0.0)):.6f}, Lat {float(spec.get('lat', 0.0)):.6f}\\n"
                f"Roof {float(spec.get('roof_area_m2', 0.0)):.1f} m2\\n"
                f"Annual demand {annual_demand:.1f}, generation {annual_generation:.1f}\\n"
                f"Final storage {cell.storage:.2f}, health {cell.health:.2f}, alive {cell.alive}\\n"
                f"Cumulative deficit ratio {deficit_ratio:.3f}, resilient {cell.resilient}"
            )
            cards.append(
                f"""
        <article class="cell-card" title="{html.escape(title)}" style="--land:{land['color'][0]}, {land['color'][1]}, {land['color'][2]}">
          <div class="cell-head">
            <strong>{html.escape(land['key'])}</strong>
            <span>{resilient_mark}{row0 + r:02d},{col0 + c:02d}</span>
          </div>
          {mini_svg(demand, generation, storage)}
          <dl>
            <div><dt>roof</dt><dd>{float(spec.get('roof_area_m2', 0.0)):.0f}</dd></div>
            <div><dt>d</dt><dd>{annual_demand:.0f}</dd></div>
            <div><dt>g</dt><dd>{annual_generation:.0f}</dd></div>
            <div><dt>def</dt><dd>{deficit_ratio:.2f}</dd></div>
          </dl>
        </article>
"""
            )
    bbox = json.loads(data.sources.get("real_patch_bbox", "{}") or "{}")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>16x16 Building Metadata And Energy Traces</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --panel: #ffffff;
      --bg: #e8edf3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    main {{
      width: min(1800px, 100%);
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 28px; letter-spacing: 0; }}
    p {{ color: var(--muted); line-height: 1.45; }}
    .hero, .legend {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: start;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, auto);
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .stats b {{ color: var(--ink); display: block; font-size: 15px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(16, minmax(72px, 1fr));
      gap: 6px;
    }}
    .cell-card {{
      min-height: 138px;
      background: rgba(var(--land), 0.13);
      border: 1px solid rgba(var(--land), 0.55);
      border-radius: 6px;
      padding: 6px;
      overflow: hidden;
    }}
    .cell-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 4px;
      font-size: 11px;
      line-height: 1.1;
      margin-bottom: 4px;
    }}
    .cell-head strong {{
      color: rgb(var(--land));
      font-size: 13px;
    }}
    .cell-head span {{ color: var(--muted); }}
    .cell-head em {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 13px;
      height: 13px;
      margin-right: 3px;
      border: 1px solid #0f172a;
      color: #0f172a;
      background: rgba(255,255,255,0.78);
      font-style: normal;
      font-size: 9px;
      font-weight: 800;
    }}
    svg {{
      width: 100%;
      height: 70px;
      display: block;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(148, 163, 184, 0.35);
    }}
    .line {{
      fill: none;
      stroke-width: 2.0;
      vector-effect: non-scaling-stroke;
      opacity: 0.9;
    }}
    .mini-legend text {{
      font-size: 8px;
      font-weight: 700;
      paint-order: stroke;
      stroke: rgba(255,255,255,0.88);
      stroke-width: 2px;
    }}
    .demand {{ stroke: {COLORS['demand']}; }}
    .generation {{ stroke: {COLORS['generation']}; }}
    .storage {{ stroke: {COLORS['storage']}; }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 3px;
      margin: 5px 0 0;
      font-size: 10px;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; color: var(--ink); font-weight: 600; }}
    .legend-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .swatch {{
      display: inline-block;
      width: 16px;
      height: 3px;
      vertical-align: middle;
      margin-right: 6px;
    }}
    @media (max-width: 1100px) {{
      main {{ padding: 14px; }}
      .hero {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: repeat(8, minmax(72px, 1fr)); }}
    }}
    @media (max-width: 620px) {{
      .grid {{ grid-template-columns: repeat(4, minmax(70px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>16x16 Building Metadata And Energy Traces</h1>
        <p>Zoom window selected automatically from the real SF upper-right building patch. Each cell is one real building; hover for full metadata.</p>
      </div>
      <div class="stats">
        <span><b>{norm}</b>norm</span>
        <span><b>{row0},{col0}</b>window origin</span>
        <span><b>{score.get('score', 0.0):.3f}</b>interest score</span>
        <span><b>{sim.config.get('steps', '')}</b>hours</span>
      </div>
    </section>
    <section class="legend">
      <p>Patch bbox: lon {bbox.get('lon_min', 0):.5f} .. {bbox.get('lon_max', 0):.5f}, lat {bbox.get('lat_min', 0):.5f} .. {bbox.get('lat_max', 0):.5f}</p>
      <div class="legend-row">
        <span><i class="swatch" style="background:{COLORS['demand']}"></i>demand</span>
        <span><i class="swatch" style="background:{COLORS['generation']}"></i>generation</span>
        <span><i class="swatch" style="background:{COLORS['storage']}"></i>storage</span>
        <span><b>R</b> resilient building, cumulative deficit <= 5%.</span>
        <span>Cell color shows land use: R residential, C commercial, I industrial.</span>
      </div>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 16x16 real-building metadata and demand/generation/storage traces.")
    parser.add_argument("--config", default="configs/static-shared-pool-annual-stress-no-grid.json")
    parser.add_argument("--out", default="outputs/building_traces_16x16.html")
    parser.add_argument("--norm", default="GEN")
    parser.add_argument("--zoom-size", type=int, default=16)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_data(config)
    row0, col0, score = select_interesting_window(data, int(config["grid_size"]), args.zoom_size)
    sim, traces = run_traces(config, args.norm.upper(), row0, col0, args.zoom_size)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_html(data, sim, traces, row0, col0, args.zoom_size, score, out_path, args.norm.upper())
    print(f"html={out_path}")
    print(f"window_origin={row0},{col0}")
    print(f"interest_score={score.get('score', 0.0):.3f}")


if __name__ == "__main__":
    main()
