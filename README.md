# Static Building-Energy ABM Baseline

This repository contains a static agent-based model for testing how fixed
energy-sharing rules perform in a neighborhood with building load, rooftop
solar, storage, and local shared storage.

The current baseline is intentionally simple:

- agent = building
- land use = fixed cell attribute
- norm = fixed sharing rule
- no external grid
- no rule evolution
- no hierarchy formation
- no rebuilding

This is meant to be compared later against a rule-evolution model.

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
| `critical_fraction` | 0.10 | fraction of buildings tagged as critical loads |
| `share_radius` | 2 | local sharing neighborhood |
| `enable_shared_storage_pool` | true | nearby surplus buildings can form a local pool |
| `normal_grid_support` | 0.0 | no external grid in normal periods |
| `outage_grid_support` | 0.0 | no external grid in outage periods |
| `storage_capacity_multiplier` | 24.0 | storage capacity scale |
| `solar_generation_multiplier` | 4.0 | solar generation scale |
| `stress_memory_retention` | 0.96 | stress memory decay factor |

Land use is fixed for each cell and affects load profile and solar/storage
potential. The current land-use types are residential, commercial, and
industrial.

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

Large-area buildings can therefore produce surplus and share with nearby
buildings.

## Fixed Norms

The sweep runs eight fixed norms:

| Key | Rule | Sharing logic |
| --- | --- | --- |
| `ALLC` | generous | shares broadly when possible |
| `SELF` | selfish | shares only when the donor keeps enough buffer or the receiver is critical |
| `DISC` | standing | shares with good-reputation or critical receivers |
| `SJ` | stern judging | reputation-sensitive indirect reciprocity |
| `SHUN` | shunning | shares mainly with good-reputation receivers |
| `CRIT` | critical first | prioritizes critical loads |
| `MKT` | market | shares when deficit/payoff is high |
| `LOCAL` | neighbor loyal | favors nearby receivers |

In this static baseline, a building does not change its norm. Each run assigns
all buildings the same fixed norm so the eight rules can be compared directly.

## Time Periods And Iterations

Each run uses 8761 hourly steps from the annual demand profile and annual EPW
solar sequence. This is not a repeated 24-hour average day.

Rendered frames are saved every 168 steps, roughly once per week. The model does
not stop on convergence in the current baseline, so all runs complete the full
annual sequence unless every building dies.

## Stress Memory

Each building stores a decaying memory of unmet demand:

```text
m_i(t+1) = lambda * m_i(t) + (1 - lambda) * u_i(t)
```

where:

- `m_i(t)` is stress memory for building `i`
- `lambda = stress_memory_retention = 0.96`
- `u_i(t)` is unmet demand fraction at the current hour

This means recent stress matters most, but old stress fades gradually.

## Metrics

Main metrics are written to CSV and JSON.

| Metric | Meaning |
| --- | --- |
| `alive_fraction` | alive buildings divided by all buildings at final step |
| `critical_survival` | alive critical buildings divided by all critical buildings |
| `annual_mean_served_fraction` | mean hourly fraction of demand served across the year |
| `annual_mean_critical_service` | mean hourly fraction of critical demand served |
| `mean_stress_memory` | final average stress memory among alive buildings |
| `critical_stress_memory` | final average stress memory among alive critical buildings |
| `annual_cooperation_successes` | total successful sharing events over the year |
| `mean_pool_members` | average local shared-storage pool size during final week |

Here, resilience means critical-load survival under no external grid support,
reported together with annual service fraction and stress memory. In other
words, a rule is more resilient if critical buildings remain alive, demand is
served consistently, and accumulated stress remains low.

## Run

Run the eight-rule static baseline:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_static_norm_sweep.py --config .\configs\static-shared-pool-annual-no-grid.json --out-dir .\results\static_shared_pool_annual_sweep
```

Run one config directly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_landuse_norm_model.py --config .\configs\static-shared-pool-annual-no-grid.json
```

## Latest Outputs

Latest results are in:

```text
results/static_shared_pool_annual_sweep/
```

Key files:

```text
static_shared_pool_annual_summary.csv
static_shared_pool_annual_summary.json
static_shared_pool_annual_summary.png
```

The latest sweep shows that most non-selfish rules produce similar final alive
fractions, while `SELF` has slightly lower total survival but the highest
critical-load survival. This makes `SELF` a useful baseline for testing whether
adaptive rule evolution can improve overall survival without sacrificing
critical-load protection.

## GitHub Status

The folder is already a Git repository. The current `origin` still points to the
original SakanaAI PD-NCA repository, so pushing to the user's GitHub account
requires replacing `origin` with the user's repository URL first.
