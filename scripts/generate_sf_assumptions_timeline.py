from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


SF_BBOX = {
    "west": -122.515,
    "east": -122.355,
    "south": 37.705,
    "north": 37.812,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def synthetic_sf_elevation(lon: float, lat: float) -> float:
    hills = [
        (-122.4467, 37.7544, 280.0, 0.015, 0.013),
        (-122.4570, 37.7600, 230.0, 0.016, 0.012),
        (-122.4180, 37.7930, 110.0, 0.010, 0.010),
        (-122.4330, 37.7920, 95.0, 0.012, 0.010),
        (-122.4030, 37.7480, 85.0, 0.018, 0.013),
        (-122.4780, 37.7170, 120.0, 0.020, 0.012),
    ]
    base = 6.0 + 16.0 * ((lat - SF_BBOX["south"]) / (SF_BBOX["north"] - SF_BBOX["south"]))
    elev = base
    for hx, hy, height, sx, sy in hills:
        elev += height * math.exp(-(((lon - hx) / sx) ** 2 + ((lat - hy) / sy) ** 2))
    return max(0.0, elev)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_cells(path: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            land = row["land"].lower() == "true"
            lon = safe_float(row["lon"])
            lat = safe_float(row["lat"])
            elevation_m = safe_float(row["elevation_m"])
            cells.append(
                {
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "lon": lon,
                    "lat": lat,
                    "land": land,
                    "building_count": int(float(row["building_count"])),
                    "elevation_m": elevation_m,
                    "height_m": synthetic_sf_elevation(lon, lat),
                    "slope": safe_float(row["slope"]),
                    "solar_factor": safe_float(row["solar_factor"]),
                    "productivity": safe_float(row["productivity"]),
                }
            )
    return cells


def load_climate_day(path: Path, day_of_year: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    params = raw.get("properties", {}).get("parameter", {})
    solar = params.get("ALLSKY_SFC_SW_DWN", {})
    temp = params.get("T2M", {})
    start = dt.datetime(2025, 1, 1)
    date = start + dt.timedelta(days=day_of_year - 1)
    hours = []
    for hour in range(24):
        key = f"{date:%Y%m%d}{hour:02d}"
        ghi = max(0.0, safe_float(solar.get(key), 0.0))
        if ghi > 10.0:
            ghi_kwh = ghi / 1000.0
        else:
            ghi_kwh = ghi
        hours.append(
            {
                "hour": hour,
                "timestamp": f"{date:%Y-%m-%d} {hour:02d}:00",
                "ghi_kwh_m2": ghi_kwh,
                "temp_c": safe_float(temp.get(key), 0.0),
            }
        )
    max_ghi = max(0.001, max(h["ghi_kwh_m2"] for h in hours))
    for hour in hours:
        hour["climate_factor"] = max(0.0, min(1.0, hour["ghi_kwh_m2"] / max_ghi))
    return hours


def write_data_js(out_dir: Path, cells: list[dict[str, Any]], hours: list[dict[str, Any]], day_of_year: int) -> None:
    land_cells = [cell for cell in cells if cell["land"]]
    height_cells = [cell for cell in cells if cell["height_m"] > 0]
    data = {
        "grid_size": max(cell["row"] for cell in cells) + 1,
        "day_of_year": day_of_year,
        "date": hours[0]["timestamp"].split(" ")[0],
        "cells": cells,
        "hours": hours,
        "summary": {
            "land_cells": len(land_cells),
            "elevation_min": min(cell["elevation_m"] for cell in land_cells),
            "elevation_max": max(cell["elevation_m"] for cell in land_cells),
            "height_min": min(cell["height_m"] for cell in height_cells),
            "height_max": max(cell["height_m"] for cell in height_cells),
            "productivity_min": min(cell["productivity"] for cell in land_cells),
            "productivity_max": max(cell["productivity"] for cell in land_cells),
            "solar_factor_min": min(cell["solar_factor"] for cell in land_cells),
            "solar_factor_max": max(cell["solar_factor"] for cell in land_cells),
        },
        "assumptions": {
            "productivity": "Productivity is a terrain-derived scalar: flatter cells are easier to build on; eastness gives a small bonus.",
            "climate": "Hourly 2025 NASA POWER GHI is a citywide temporal signal. Spatial solar potential is climate_factor * local solar_factor.",
            "solar_factor": "Local solar_factor encodes west-to-east fog gradient plus a small south-facing terrain aspect adjustment.",
        },
    }
    (out_dir / "sf_assumptions_timeline_data.js").write_text(
        "window.SF_ASSUMPTIONS = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def write_html(out_dir: Path) -> None:
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SF Terrain, Productivity, And Solar Potential</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #667085;
      --line: #d7dee8;
      --paper: #ffffff;
      --soft: #f5f8fb;
      --water: #e9f0f6;
      --terrain-low: #d9ead5;
      --terrain-high: #8a7f6d;
      --solar: #f2c94c;
      --prod: #1f9d8a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #e8eef4;
      color: var(--ink);
    }

    main {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 18px;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: baseline;
      padding: 8px 2px 16px;
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.08;
      letter-spacing: 0;
    }

    h2 {
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }

    p {
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.52;
    }

    .subtitle {
      margin-top: 8px;
      max-width: 980px;
    }

    .tag {
      flex: 0 0 auto;
      border: 1px solid var(--line);
      background: var(--paper);
      padding: 7px 10px;
      border-radius: 6px;
      color: #3d4b63;
      font-size: 13px;
      white-space: nowrap;
    }

    .timeline {
      margin-top: 16px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }

    .timeline-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }

    .hour-badge {
      min-width: 112px;
      min-height: 42px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: #25344d;
      color: white;
      font-weight: 800;
      font-size: 18px;
      letter-spacing: 0;
    }

    input[type="range"] {
      width: 100%;
      accent-color: #25344d;
    }

    .climate-readout {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 920px);
      justify-content: center;
      gap: 18px;
      margin-top: 18px;
      align-items: start;
    }

    .panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .panel-head h2 {
      margin: 0;
    }

    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 32px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--soft);
      color: #3d4b63;
      font-size: 13px;
      white-space: nowrap;
      user-select: none;
    }

    .toggle input {
      margin: 0;
      accent-color: #25344d;
    }

    canvas {
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .metric {
      min-height: 82px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
    }

    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 23px;
      letter-spacing: 0;
    }

    .legend {
      display: grid;
      gap: 8px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }

    .legend-item {
      display: grid;
      grid-template-columns: 16px 1fr;
      gap: 8px;
      align-items: center;
    }

    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid rgba(23, 32, 51, 0.16);
    }

    .explain {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }

    .rule {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      margin-top: 10px;
    }

    .rule strong { display: block; margin-bottom: 4px; }
    .rule span { color: var(--muted); line-height: 1.45; display: block; }

    @media (max-width: 1120px) {
      main { padding: 14px; }
      .topbar { flex-direction: column; align-items: flex-start; }
      .timeline-row,
      .grid,
      .explain { grid-template-columns: 1fr; }
      .climate-readout { white-space: normal; }
    }

    @media (max-width: 620px) {
      .metrics-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div>
        <h1>SF terrain, productivity, and solar potential</h1>
        <p class="subtitle">This page separates the exogenous assumptions: height is shown as terrain context, slope is mapped into productivity, and 2025 hourly climate is mapped into solar potential. Drag the timeline to inspect one sampled day.</p>
      </div>
      <div class="tag" id="dateTag">sample day</div>
    </header>

    <section class="timeline">
      <div class="timeline-row">
        <div class="hour-badge" id="hourBadge">12:00</div>
        <input id="hourSlider" type="range" min="0" max="23" value="12" step="1" aria-label="hour of day">
        <div class="climate-readout" id="climateReadout">GHI 0.00 kWh/m2 - temp 0.0 C</div>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>1. Height Map: Elevation Context Only</h2>
        <canvas id="heightCanvas" width="760" height="760"></canvas>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>2. Property Value Mapping: Slope -> Productivity</h2>
          <label class="toggle"><input id="productivityToggle" type="checkbox"> productivity values</label>
        </div>
        <canvas id="productivityCanvas" width="760" height="760"></canvas>
      </div>

      <div class="panel">
        <h2>3. Climate x Local Solar Factor</h2>
        <canvas id="solarCanvas" width="760" height="760"></canvas>
      </div>

      <aside class="panel">
        <h2>Selected Hour</h2>
        <p id="hourText">Each cell's solar potential is citywide hourly GHI multiplied by its local fog/aspect factor.</p>
        <div class="metrics-grid">
          <div class="metric"><span>citywide climate factor</span><strong id="climateFactorValue">0.000</strong></div>
          <div class="metric"><span>GHI</span><strong id="ghiValue">0.00</strong></div>
          <div class="metric"><span>temperature</span><strong id="tempValue">0.0 C</strong></div>
          <div class="metric"><span>mean land solar potential</span><strong id="meanSolarValue">0.000</strong></div>
          <div class="metric"><span>max land solar potential</span><strong id="maxSolarValue">0.000</strong></div>
          <div class="metric"><span>mean productivity</span><strong id="meanProdValue">0.000</strong></div>
        </div>
        <div class="legend">
          <div class="legend-item"><span class="swatch" style="background:#d9ead5"></span>height map: lower elevation</div>
          <div class="legend-item"><span class="swatch" style="background:#8a7f6d"></span>height map: higher elevation</div>
          <div class="legend-item"><span class="swatch" style="background:#1f9d8a"></span>property value: higher productivity</div>
          <div class="legend-item"><span class="swatch" style="background:#f2c94c"></span>yellow = hourly solar potential</div>
          <div class="legend-item"><span class="swatch" style="background:#e9f0f6"></span>thin white mask = no building-density cell</div>
        </div>
      </aside>
    </section>

    <section class="explain">
      <div class="panel">
        <h2>Assumption 1: Productivity</h2>
        <div class="rule">
          <strong>Only terrain-derived</strong>
          <span>Productivity is not derived from elevation. It is derived mostly from local slope, with a small eastness term: flatter cells are easier to build on; steep terrain reduces building productivity.</span>
        </div>
        <div class="rule">
          <strong>No building function here</strong>
          <span>Buildings are only residential or commercial in the ABM. PV and batteries are purchasable assets, not additional building functions.</span>
        </div>
      </div>

      <div class="panel">
        <h2>Assumption 3: Climate</h2>
        <div class="rule">
          <strong>One temporal climate signal</strong>
          <span>The hourly curve is NASA POWER 2025 GHI at a San Francisco point. It controls when solar is possible during the day.</span>
        </div>
        <div class="rule">
          <strong>One spatial modifier</strong>
          <span>Each cell has a local solar factor: west-side fog lowers potential, east-side sun raises it, and south-facing terrain adds a small bonus.</span>
        </div>
      </div>
    </section>
  </main>

  <script src="sf_assumptions_timeline_data.js"></script>
  <script>
    const DATA = window.SF_ASSUMPTIONS;
    const cells = DATA.cells;
    const landCells = cells.filter((cell) => cell.land);
    const gridSize = DATA.grid_size;
    const slider = document.getElementById("hourSlider");
    const hourBadge = document.getElementById("hourBadge");
    const climateReadout = document.getElementById("climateReadout");
    const heightCanvas = document.getElementById("heightCanvas");
    const productivityCanvas = document.getElementById("productivityCanvas");
    const solarCanvas = document.getElementById("solarCanvas");
    const productivityToggle = document.getElementById("productivityToggle");

    document.getElementById("dateTag").textContent = `${DATA.date} · day ${DATA.day_of_year}`;

    const minElev = DATA.summary.elevation_min;
    const maxElev = DATA.summary.elevation_max;
    const minHeight = DATA.summary.height_min;
    const maxHeight = DATA.summary.height_max;
    const minProd = DATA.summary.productivity_min;
    const maxProd = DATA.summary.productivity_max;
    const maxLocalSolar = DATA.summary.solar_factor_max;
    const meanProd = landCells.reduce((sum, cell) => sum + cell.productivity, 0) / landCells.length;
    document.getElementById("meanProdValue").textContent = meanProd.toFixed(3);

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function mix(a, b, t) {
      return Math.round(a + (b - a) * t);
    }

    function color(low, high, t) {
      const l = low.match(/\\w\\w/g).map((x) => parseInt(x, 16));
      const h = high.match(/\\w\\w/g).map((x) => parseInt(x, 16));
      return `rgb(${mix(l[0], h[0], t)},${mix(l[1], h[1], t)},${mix(l[2], h[2], t)})`;
    }

    function setupCanvas(canvas) {
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(320, Math.round(rect.width));
      canvas.width = width * dpr;
      canvas.height = width * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, size: width };
    }

    function drawHeight() {
      const { ctx, size } = setupCanvas(heightCanvas);
      const pad = 22;
      const cellSize = (size - pad * 2) / gridSize;
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, size, size);
      for (const cell of cells) {
        const x = pad + cell.col * cellSize;
        const y = pad + cell.row * cellSize;
        const heightT = clamp((cell.height_m - minHeight) / Math.max(1, maxHeight - minHeight), 0, 1);
        ctx.fillStyle = color("d7ead6", "594a42", heightT);
        ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
        const contour = Math.floor(heightT * 12);
        const eastNeighbor = cells.find((other) => other.row === cell.row && other.col === cell.col + 1);
        const southNeighbor = cells.find((other) => other.row === cell.row + 1 && other.col === cell.col);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.38)";
        ctx.lineWidth = 1;
        if (eastNeighbor && Math.floor(clamp((eastNeighbor.height_m - minHeight) / Math.max(1, maxHeight - minHeight), 0, 1) * 12) !== contour) {
          ctx.beginPath();
          ctx.moveTo(x + cellSize - 0.8, y);
          ctx.lineTo(x + cellSize - 0.8, y + cellSize - 0.8);
          ctx.stroke();
        }
        if (southNeighbor && Math.floor(clamp((southNeighbor.height_m - minHeight) / Math.max(1, maxHeight - minHeight), 0, 1) * 12) !== contour) {
          ctx.beginPath();
          ctx.moveTo(x, y + cellSize - 0.8);
          ctx.lineTo(x + cellSize - 0.8, y + cellSize - 0.8);
          ctx.stroke();
        }
        if (!cell.land) {
          ctx.fillStyle = "rgba(248, 250, 252, 0.42)";
          ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
        }
      }
      ctx.fillStyle = "#172033";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("Height map only: elevation is context, not the productivity variable", pad, size - 8);
    }

    function drawProductivity() {
      const { ctx, size } = setupCanvas(productivityCanvas);
      const pad = 22;
      const cellSize = (size - pad * 2) / gridSize;
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, size, size);
      for (const cell of cells) {
        const x = pad + cell.col * cellSize;
        const y = pad + cell.row * cellSize;
        if (!cell.land) {
          ctx.fillStyle = "#e9f0f6";
          ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
          continue;
        }
        const prodT = clamp((cell.productivity - minProd) / Math.max(0.001, maxProd - minProd), 0, 1);
        const slopeT = clamp(cell.slope / 95, 0, 1);
        ctx.fillStyle = color("f0a36a", "1f9d8a", prodT);
        ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
        ctx.fillStyle = `rgba(23, 32, 51, ${0.08 + slopeT * 0.34})`;
        ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
        if (productivityToggle.checked && cellSize >= 18) {
          ctx.fillStyle = prodT > 0.55 ? "#062d27" : "#172033";
          ctx.font = `${Math.max(7, Math.floor(cellSize * 0.30))}px Inter, system-ui, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(cell.productivity.toFixed(2), x + cellSize * 0.5, y + cellSize * 0.52);
        }
      }
      ctx.fillStyle = "#172033";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("Productivity mapping: local slope lowers property value; elevation is not used directly", pad, size - 8);
    }

    function drawSolar(hourIndex) {
      const hour = DATA.hours[hourIndex];
      const { ctx, size } = setupCanvas(solarCanvas);
      const pad = 22;
      const cellSize = (size - pad * 2) / gridSize;
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, size, size);

      let sum = 0;
      let max = 0;
      for (const cell of cells) {
        const x = pad + cell.col * cellSize;
        const y = pad + cell.row * cellSize;
        if (!cell.land) {
          ctx.fillStyle = "#e9f0f6";
          ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
          continue;
        }
        const potential = hour.climate_factor * cell.solar_factor / maxLocalSolar;
        sum += potential;
        max = Math.max(max, potential);
        const t = clamp(potential, 0, 1);
        ctx.fillStyle = color("27384f", "f2c94c", t);
        ctx.fillRect(x, y, cellSize - 0.8, cellSize - 0.8);
      }
      const mean = sum / landCells.length;
      document.getElementById("meanSolarValue").textContent = mean.toFixed(3);
      document.getElementById("maxSolarValue").textContent = max.toFixed(3);
      ctx.fillStyle = "#172033";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.fillText("Hourly solar potential = climate factor x local solar factor", pad, size - 8);
    }

    function update(hourIndex) {
      const hour = DATA.hours[hourIndex];
      const hh = String(hour.hour).padStart(2, "0");
      hourBadge.textContent = `${hh}:00`;
      climateReadout.textContent = `GHI ${hour.ghi_kwh_m2.toFixed(2)} kWh/m2 - temp ${hour.temp_c.toFixed(1)} C`;
      document.getElementById("climateFactorValue").textContent = hour.climate_factor.toFixed(3);
      document.getElementById("ghiValue").textContent = hour.ghi_kwh_m2.toFixed(2);
      document.getElementById("tempValue").textContent = `${hour.temp_c.toFixed(1)} C`;
      document.getElementById("hourText").textContent = `${hour.timestamp}: citywide climate factor ${hour.climate_factor.toFixed(3)}; spatial variation comes only from the local fog/aspect solar factor.`;
      drawHeight();
      drawProductivity();
      drawSolar(hourIndex);
    }

    slider.addEventListener("input", () => update(Number(slider.value)));
    productivityToggle.addEventListener("change", () => drawProductivity());
    window.addEventListener("resize", () => update(Number(slider.value)));
    update(Number(slider.value));
  </script>
</body>
</html>
"""
    (out_dir / "sf_assumptions_terrain_climate.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a draggable SF terrain/climate assumptions viewer.")
    parser.add_argument("--day-of-year", type=int, default=209)
    parser.add_argument("--out-dir", type=Path, default=project_root() / "outputs" / "sf_terrain_energy_growth")
    parser.add_argument("--cells-csv", type=Path, default=project_root() / "outputs" / "sf_terrain_energy_growth" / "sf_energy_growth_cells.csv")
    parser.add_argument("--climate-json", type=Path, default=project_root().parents[0] / "data" / "sf_terrain_energy_growth_cache" / "nasa_power_sf_2025_hourly.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(args.cells_csv)
    hours = load_climate_day(args.climate_json, args.day_of_year)
    write_data_js(args.out_dir, cells, hours, args.day_of_year)
    write_html(args.out_dir)
    print(json.dumps({
        "html": str(args.out_dir / "sf_assumptions_terrain_climate.html"),
        "data": str(args.out_dir / "sf_assumptions_timeline_data.js"),
        "date": hours[0]["timestamp"].split(" ")[0],
        "hours": len(hours),
        "cells": len(cells),
    }, indent=2))


if __name__ == "__main__":
    main()
