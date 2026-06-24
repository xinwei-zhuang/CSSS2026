from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landuse_norm_model import NORMS, normalized_resilience_auc, run_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/static-shared-pool-annual-no-grid.json")
    args = parser.parse_args()

    result, out_dir = run_from_config(args.config)
    final = result.metrics[-1]
    top_norm = max(range(len(NORMS)), key=lambda idx: final["norm_frequencies"][idx])
    print(f"out_dir={out_dir.resolve()}")
    print(f"steps={len(result.metrics)}")
    print(f"alive_buildings_percent={100.0 * final['alive_fraction']:.1f}")
    print(f"resilience_normalized={normalized_resilience_auc(result.metrics):.3f}")
    print(f"model_mode={final.get('model_mode', 'evolving_norms')}")
    print(f"fixed_norm_key={final.get('fixed_norm_key', '')}")
    print(f"converged={final.get('converged', False)}")
    print(f"stable_steps={final.get('stable_steps', 0)}")
    print(f"daily_health_delta={final.get('daily_health_delta', 0.0):.6f}")
    print(f"daily_storage_delta={final.get('daily_storage_delta', 0.0):.6f}")
    print(f"daily_alive_changes={final.get('daily_alive_changes', 0)}")
    print(f"top_norm={NORMS[top_norm]['key']}:{NORMS[top_norm]['name']} {final['norm_frequencies'][top_norm]:.3f}")
    print("final_norm_frequencies=" + json.dumps({
        NORMS[idx]["key"]: round(value, 4)
        for idx, value in enumerate(final["norm_frequencies"])
    }))


if __name__ == "__main__":
    main()
