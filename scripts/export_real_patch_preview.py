from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import LAND_USES, load_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the selected real-building patch used by the model.")
    parser.add_argument("--config", default="configs/static-shared-pool-annual-stress-no-grid.json")
    parser.add_argument("--out-dir", default="outputs/real_patch")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_data(config)
    if not data.cell_specs:
        raise ValueError("Config did not produce real building cell specs. Set use_real_building_patch=true.")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    size = int(config["grid_size"])

    csv_path = out_dir / "real_patch_cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["grid_row", "grid_col", "building_id", "profile_id", "lon", "lat", "landuse", "roof_area_m2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, spec in enumerate(data.cell_specs):
            row, col = divmod(idx, size)
            writer.writerow({
                "grid_row": row,
                "grid_col": col,
                "building_id": spec.get("building_id", ""),
                "profile_id": spec.get("profile_id", ""),
                "lon": spec.get("lon", ""),
                "lat": spec.get("lat", ""),
                "landuse": LAND_USES[int(spec["landuse"])]["name"],
                "roof_area_m2": f"{float(spec['roof_area_m2']):.3f}",
            })

    scale = 18
    margin = 34
    image = Image.new("RGB", (size * scale + margin * 2, size * scale + margin * 2 + 54), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 14), "Real SF upper-right building patch", fill=(15, 23, 42))
    for idx, spec in enumerate(data.cell_specs):
        row, col = divmod(idx, size)
        x = margin + col * scale
        y = margin + 30 + row * scale
        color = LAND_USES[int(spec["landuse"])]["color"]
        draw.rectangle([x, y, x + scale - 1, y + scale - 1], fill=color)
    legend_y = margin + 30 + size * scale + 14
    x = margin
    for land in LAND_USES:
        draw.rectangle([x, legend_y + 3, x + 14, legend_y + 15], fill=land["color"])
        draw.text((x + 20, legend_y), land["name"], fill=(71, 85, 105))
        x += 150
    png_path = out_dir / "real_patch_map.png"
    image.save(png_path)
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    print(data.sources.get("real_patch_bbox", ""))


if __name__ == "__main__":
    main()
