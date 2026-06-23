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
| `share_radius` | 10 | maximum local sharing distance in cells |
| `sharing_min_efficiency` | 0.35 | lower bound on delivered sharing efficiency |
| `sharing_efficiency_decay` | 0.65 | proportional distance loss at the edge of the radius |
| `sharing_loss_exponent` | 1.0 | linear distance-loss curve |
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

## Distance-Decayed Sharing

Surplus can be shared within a 10-cell radius. Sharing is not lossless: the
delivered energy decays with distance. For a donor-receiver distance \(d\), the
delivery efficiency is:

```text
efficiency(d) = max(min_efficiency, 1 - decay * (d / share_radius)^exponent)
```

With the current settings:

```text
efficiency(d) = max(0.35, 1 - 0.65 * d / 10)
```

So nearby sharing is almost lossless, while sharing from the edge of the radius
delivers about 35% of the transferred energy. Donors are ranked by effective
deliverable surplus, so large nearby surplus is tried first.

## Fixed Norms

The sweep runs eight fixed norms:

| Key | Rule | Sharing logic |
| --- | --- | --- |
| `ALLC` | generous | shares broadly when possible |
| `SELF` | selfish | shares only with critical receivers and only from surplus above a high buffer |
| `DISC` | standing | shares with good-reputation receivers, and with critical receivers if storage is adequate |
| `SJ` | stern judging | shares only with good-reputation receivers under stricter storage conditions |
| `SHUN` | shunning | shares only with excellent-reputation receivers and keeps a high storage buffer |
| `CRIT` | critical first | shares almost exclusively with critical receivers |
| `MKT` | market | shares when receiver need exceeds distance cost and reputation-adjusted payoff threshold |
| `LOCAL` | neighbor loyal | uses the 10-cell search radius but only cooperates inside a 3-cell local norm radius |

In this static baseline, a building does not change its norm. Each run assigns
all buildings the same fixed norm so the eight rules can be compared directly.
The shared-storage pool only identifies candidate donors; the final donation
decision must still pass the donor's norm rule. Norms also differ in the maximum
fraction of surplus a donor is willing to transfer.

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
| `mean_pool_members` | average donor memberships in local shared-storage requests during final week |

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
combined_results.html
```

The latest sweep uses `share_radius = 10` with linear distance-decayed sharing
and sharper norm definitions. `DISC` and `SJ` now produce the highest overall
survival, both around 0.695. `SELF` and `CRIT` produce lower total survival but
much higher critical-load survival, around 0.816 and 0.824. `LOCAL` is weakest
because it only accepts near-neighbor cooperation, and `MKT` creates many
attempts but higher stress memory because the payoff threshold filters many
candidate exchanges. This makes the static baseline more useful for testing
whether adaptive rule evolution can improve total survival without sacrificing
critical-load protection.

## GitHub Status

The folder is a Git repository with `origin` set to:

```text
https://github.com/xinwei-zhuang/CSSS2026.git
```
