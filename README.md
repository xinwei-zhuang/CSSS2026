# Static Building-Energy ABM Baseline

This repository contains a static agent-based model for comparing two
building-level energy-sharing norms in a neighborhood with load, rooftop solar,
storage, and local shared storage.

The current baseline is intentionally simple:

- agent = building
- land use = fixed cell attribute
- norm = fixed sharing rule
- no external grid
- no rule evolution
- no hierarchy formation
- no rebuilding

## Initial Setting

The main configuration is:

```text
configs/static-shared-pool-annual-no-grid.json
```

Important default settings:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `grid_size` | 36 | 36 x 36 building cells |
| `steps` | 8761 | one full hourly annual profile |
| `share_radius` | 10 | maximum sharing search distance in cells |
| `enable_shared_storage_pool` | true | nearby surplus buildings can form a local pool |
| `normal_grid_support` | 0.0 | no external grid in normal periods |
| `outage_grid_support` | 0.0 | no external grid in no-solar stress periods |
| `resilient_deficit_threshold` | 0.05 | building is marked resilient when cumulative unmet demand is <= 5% |
| `storage_capacity_multiplier` | 24.0 | storage capacity scale |
| `solar_generation_multiplier` | 4.0 | solar generation scale |

## Fixed Norms

The sweep runs two fixed norms:

| Key | Rule | Sharing logic |
| --- | --- | --- |
| `SELF` | selfish | does not share |
| `GEN` | generous | shares whenever possible |

In this static baseline, a building does not change its norm. Each run assigns
all buildings the same fixed norm so the two rules can be compared directly.

## Evaluation Metrics

Only two evaluation metrics are reported in the sweep summary:

| Metric | Meaning |
| --- | --- |
| `alive_buildings_percent` | final alive buildings divided by all buildings, reported as a percent |
| `resilience_normalized` | area under Q(t), where Q(t) is alive-building fraction, normalized to [0,1] |

The normalized resilience metric follows:

```text
R = integral(Q(t) dt) / integral(Q0 dt)
```

Here `Q0 = 1`, so a run that keeps all buildings alive for the full simulated
period has `R = 1`.

## Data

Demand uses the cleaned annual building energy profiles:

```text
../data/energy_profiles_clean/energy_profiles_hourly_used.csv
../data/energy_profiles_clean/building_energy_metadata.csv
```

Solar uses the San Francisco EPW file configured in the JSON file. The simplified
solar model is:

```text
building roof area * hourly solar radiation * PV efficiency * usable roof fraction
```

## Run

Run the two-rule static baseline:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_static_norm_sweep.py --config .\configs\static-shared-pool-annual-no-grid.json --out-dir .\results\static_shared_pool_annual_sweep
```

Run one config directly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_landuse_norm_model.py --config .\configs\static-shared-pool-annual-no-grid.json
```

## Outputs

Latest sweep results are written to:

```text
results/static_shared_pool_annual_sweep/
```

Key files:

```text
static_shared_pool_annual_summary.csv
static_shared_pool_annual_summary.json
static_shared_pool_annual_summary.png
combined_results.html
```
