from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from model import CASunGroup  # noqa: E402
from world import World  # noqa: E402


TRIBE_COLORS = [
    (37, 99, 235),
    (217, 119, 6),
    (21, 128, 61),
]
TRIBE_NAMES = ["solar area", "dense load", "storage support"]


def grid_metrics(grid: torch.Tensor, n_ncas: int) -> dict[str, float | list[float]]:
    alive = grid[: n_ncas + 1].detach().float().cpu()
    winners = torch.argmax(alive, dim=0)
    total = winners.numel()
    coverage = [
        float((winners == (idx + 1)).sum().item() / total)
        for idx in range(n_ncas)
    ]
    sun = float((winners == 0).sum().item() / total)
    occupied = sum(coverage)
    mix = 0.0
    if occupied > 0:
        mix = 1.0 - sum((c / occupied) ** 2 for c in coverage)
        if n_ncas > 1:
            mix /= 1.0 - 1.0 / n_ncas
    return {
        "sun_fraction": sun,
        "occupied_fraction": occupied,
        "tribe_coverage": coverage,
        "tribe_mix": mix,
    }


def render_grid(
    grid: torch.Tensor,
    n_ncas: int,
    step: int,
    scale: int,
    metrics: dict[str, float | list[float]],
) -> Image.Image:
    alive = grid[: n_ncas + 1].detach().float().cpu()
    winners = torch.argmax(alive, dim=0)
    strength = torch.max(alive, dim=0).values.clamp(0, 1)
    height, width = winners.shape
    margin = 18
    footer = 38
    image = Image.new(
        "RGB",
        (width * scale + 2 * margin, height * scale + 2 * margin + footer),
        (248, 250, 252),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [margin - 1, margin - 1, margin + width * scale, margin + height * scale],
        outline=(15, 23, 42),
    )
    for row in range(height):
        for col in range(width):
            winner = int(winners[row, col].item())
            control = float(strength[row, col].item())
            if winner == 0:
                color = (226, 232, 240)
            else:
                base = TRIBE_COLORS[(winner - 1) % len(TRIBE_COLORS)]
                bright = 0.45 + 0.55 * control
                color = tuple(min(255, max(0, int(v * bright))) for v in base)
            x = margin + col * scale
            y = margin + row * scale
            draw.rectangle([x, y, x + scale - 1, y + scale - 1], fill=color)

    coverage = metrics["tribe_coverage"]
    coverage_text = " ".join(
        f"{TRIBE_NAMES[i]}={coverage[i]:.2f}" for i in range(min(n_ncas, 3))
    )
    footer_text = (
        f"step={step:03d} occupied={metrics['occupied_fraction']:.2f} "
        f"mix={metrics['tribe_mix']:.2f} {coverage_text}"
    )
    draw.text(
        (margin, margin + height * scale + 12),
        footer_text,
        fill=(15, 23, 42),
    )
    return image


def make_contact_sheet(frames: list[Image.Image], path: Path, columns: int = 4) -> None:
    thumbs = [frame.resize((260, 292), Image.Resampling.NEAREST) for frame in frames]
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 260, rows * 292), (248, 250, 252))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % columns) * 260, (idx // columns) * 292))
    sheet.save(path)


def write_html(
    html_path: Path,
    gif_name: str,
    sheet_name: str,
    metrics_name: str,
    checkpoint_name: str,
) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Petri City PD-NCA Final Results</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: #eef2f7;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 18px;
      color: #475569;
      line-height: 1.5;
    }}
    .result {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 18px;
      margin-top: 18px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
    }}
    code {{
      background: #e2e8f0;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <main>
    <h1>Petri City PD-NCA Final Results</h1>
    <p>
      This run uses three competing neural cellular automata with real San Francisco
      building demand profiles and area-based solar generation from EPW radiation.
      Checkpoint: <code>{checkpoint_name}</code>.
    </p>
    <div class="result">
      <h2>Evolution GIF</h2>
      <img src="{gif_name}" alt="PD-NCA city tissue evolution">
    </div>
    <div class="result">
      <h2>Contact Sheet</h2>
      <img src="{sheet_name}" alt="PD-NCA city tissue contact sheet">
    </div>
    <p class="result">
      Metrics JSON: <a href="{metrics_name}">{metrics_name}</a>
    </p>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def render(checkpoint: Path, out_dir: Path, steps: int, sample_every: int, scale: int) -> None:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {checkpoint}")

    config = Config.from_file(str(config_path))
    config.device = "cpu"
    config.mode = "frozen_eval"
    config.batch_size = 1
    config.__post_init__()

    world = World(config)
    group = CASunGroup(config)
    group.load(str(checkpoint))

    grid = world.get_seed()
    with torch.no_grad():
        _, grids, _ = group(grid, steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    sheet_frames = []
    metrics = []
    for step in range(grids.shape[0]):
        grid_step = grids[step, 0]
        step_metrics = grid_metrics(grid_step, config.n_ncas)
        metrics.append({"step": step + 1, **step_metrics})
        if step % sample_every == 0 or step == grids.shape[0] - 1:
            frame = render_grid(
                grid_step,
                config.n_ncas,
                step + 1,
                scale,
                step_metrics,
            )
            frames.append(frame)
            if len(sheet_frames) < 12:
                sheet_frames.append(frame)

    gif_path = out_dir / "city_pdnca_final.gif"
    sheet_path = out_dir / "city_pdnca_contact_sheet.png"
    snapshot_path = out_dir / "city_pdnca_final_snapshot.png"
    metrics_path = out_dir / "city_pdnca_metrics.json"
    html_path = out_dir / "city_pdnca_results.html"

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    make_contact_sheet(sheet_frames, sheet_path)
    frames[-1].save(snapshot_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_html(
        html_path,
        gif_path.name,
        sheet_path.name,
        metrics_path.name,
        checkpoint.name,
    )

    print(f"gif={gif_path.resolve()}")
    print(f"contact_sheet={sheet_path.resolve()}")
    print(f"snapshot={snapshot_path.resolve()}")
    print(f"html={html_path.resolve()}")
    print(f"metrics={metrics_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="results/final_area_solar_visualization")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--scale", type=int, default=12)
    args = parser.parse_args()

    render(
        checkpoint=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        steps=args.steps,
        sample_every=args.sample_every,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
