from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import NORMS, run_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/landuse-norm-final.json")
    args = parser.parse_args()

    result, out_dir = run_from_config(args.config)
    final = result.metrics[-1]
    top_norm = max(range(len(NORMS)), key=lambda idx: final["norm_frequencies"][idx])
    print(f"out_dir={out_dir.resolve()}")
    print(f"steps={len(result.metrics)}")
    print(f"served_fraction={final['served_fraction']:.3f}")
    print(f"critical_survival={final['critical_survival']:.3f}")
    print(f"cooperation_rate={final['cooperation_rate']:.3f}")
    print(f"top_norm={NORMS[top_norm]['key']}:{NORMS[top_norm]['name']} {final['norm_frequencies'][top_norm]:.3f}")
    print("final_norm_frequencies=" + json.dumps({
        NORMS[idx]["key"]: round(value, 4)
        for idx, value in enumerate(final["norm_frequencies"])
    }))


if __name__ == "__main__":
    main()
