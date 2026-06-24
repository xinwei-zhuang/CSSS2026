from __future__ import annotations

import base64
import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "static_shared_pool_annual_sweep"
OUT = RESULTS / "combined_results.html"
STRESS_PNG = ROOT / "data" / "outage_profiles" / "periodic_no_solar_stress.png"
REAL_PATCH_PNG = ROOT / "outputs" / "real_patch" / "real_patch_map.png"
TRACE_HTML = ROOT / "outputs" / "building_traces_16x16.html"

NORM_DIRS = ["self", "gen"]
ASSETS = [
    ("Evolution", "landuse_norm_evolution.gif", "image/gif"),
    ("Final snapshot", "landuse_norm_final_snapshot.png", "image/png"),
    ("Metrics", "landuse_norm_metrics.png", "image/png"),
    ("Contact sheet", "landuse_norm_contact_sheet.png", "image/png"),
]


def data_uri(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def read_summary() -> list[dict[str, str]]:
    summary_path = RESULTS / "static_shared_pool_annual_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def summary_table(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['norm'])}</td>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{fmt(row['alive_buildings_percent'], 1)}</td>"
            f"<td>{fmt(row['resilience_normalized'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def norm_section(norm_dir: str, summary: dict[str, str]) -> str:
    directory = RESULTS / norm_dir
    cards = []
    for label, filename, mime in ASSETS:
        path = directory / filename
        if not path.exists():
            continue
        cards.append(
            "<figure>"
            f"<img src=\"{data_uri(path, mime)}\" alt=\"{html.escape(summary['norm'])} {html.escape(label)}\">"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )

    return f"""
    <section id="{html.escape(norm_dir)}" class="norm-section">
      <div class="section-head">
        <div>
          <h2>{html.escape(summary['norm'])}: {html.escape(summary['name'])}</h2>
          <p>Alive buildings {fmt(summary['alive_buildings_percent'], 1)}% | Resilience {fmt(summary['resilience_normalized'])}</p>
        </div>
        <a href="#top">Top</a>
      </div>
      <div class="media-grid">
        {''.join(cards)}
      </div>
    </section>
    """


def main() -> None:
    rows = read_summary()
    row_by_norm = {row["norm"].lower(): row for row in rows}
    summary_png = data_uri(RESULTS / "static_shared_pool_annual_summary.png", "image/png")
    stress_panel = ""
    if STRESS_PNG.exists():
        stress_panel = f"""
    <section class="panel">
      <h2>Periodic No-Solar Stress Profile</h2>
      <p>Solar generation is set to zero for periodic daytime windows; external grid support remains zero throughout.</p>
      <img class="summary-img" src="{data_uri(STRESS_PNG, "image/png")}" alt="Periodic no-solar stress profile">
    </section>
"""
    patch_panel = ""
    if REAL_PATCH_PNG.exists():
        patch_panel = f"""
    <section class="panel">
      <h2>Real SF Patch</h2>
      <p>36 x 36 real-building sample selected from the upper-right corner of the SF building metadata extent.</p>
      <img class="summary-img" src="{data_uri(REAL_PATCH_PNG, "image/png")}" alt="Real SF upper-right building patch">
    </section>
"""
    trace_panel = ""
    if TRACE_HTML.exists():
        trace_panel = """
    <section class="panel">
      <h2>16x16 Building Trace Zoom</h2>
      <p>Each cell is one real building with metadata plus demand, generation, and storage time-series lines.</p>
      <p><a href="../../outputs/building_traces_16x16.html">Open the 16x16 building trace grid</a></p>
    </section>
"""
    sections = "\n".join(norm_section(norm_dir, row_by_norm[norm_dir]) for norm_dir in NORM_DIRS)
    nav = " ".join(
        f"<a href=\"#{norm_dir}\">{html.escape(row_by_norm[norm_dir]['norm'])}</a>"
        for norm_dir in NORM_DIRS
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Static Building-Energy ABM: Combined Results</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --soft: #f1f5f9;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #e8edf3;
    }}
    main {{
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 22px; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    a {{ color: #2563eb; text-decoration: none; }}
    .hero, .panel, .norm-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: start;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .nav a {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 9px;
      background: var(--soft);
      color: var(--ink);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 8px;
      border-bottom: 1px solid #e2e8f0;
      white-space: nowrap;
    }}
    th {{ color: #334155; background: #f8fafc; }}
    .summary-img {{
      width: 100%;
      border: 1px solid var(--line);
      background: #f8fafc;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .section-head p {{ margin: 6px 0 0; }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .comparison-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .comparison-grid .norm-section {{
      margin-bottom: 0;
    }}
    .comparison-grid .media-grid {{
      grid-template-columns: 1fr;
    }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      background: #f8fafc;
    }}
    figure img {{
      width: 100%;
      display: block;
    }}
    figcaption {{
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      background: white;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 16px; }}
      .hero {{ grid-template-columns: 1fr; }}
      .nav {{ justify-content: flex-start; }}
      .comparison-grid {{ grid-template-columns: 1fr; }}
      .media-grid {{ grid-template-columns: 1fr; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main id="top">
    <header class="hero">
      <div>
        <h1>Static Building-Energy ABM: Combined Results</h1>
        <p>Selfish and generous sharing rules under the same annual no-grid baseline. Evaluation reports only alive buildings (%) and normalized resilience AUC.</p>
      </div>
      <nav class="nav">{nav}</nav>
    </header>

    <section class="panel">
      <h2>Summary</h2>
      <p>Resilience is the area under Q(t), where Q(t) is the alive-building fraction, normalized to [0,1].</p>
      <table>
        <thead>
          <tr>
            <th>Norm</th>
            <th>Name</th>
            <th>Alive buildings (%)</th>
            <th>Resilience [0,1]</th>
          </tr>
        </thead>
        <tbody>
          {summary_table(rows)}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Summary PNG</h2>
      <img class="summary-img" src="{summary_png}" alt="Static shared-storage annual summary">
    </section>

    {stress_panel}

    {patch_panel}

    {trace_panel}

    <div class="comparison-grid">
      {sections}
    </div>
  </main>
</body>
</html>
"""
    OUT.write_text(document, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
