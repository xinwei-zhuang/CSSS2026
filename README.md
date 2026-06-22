# Petri City: Land-Use Conditioned Norm Evolution

This project models a neighborhood as an artificial-life energy system. The
important conceptual correction is:

```text
agent = building or merged building entity
tissue = slowly changing land-use context
tribe = energy-sharing norm / protocol
```

The model no longer treats solar, load, and storage as three biological
species. Land use is now a slow urban tissue variable rather than a permanent
background map. Buildings decide which cooperation rule to follow, imitate
successful neighbors, mutate, build reputation, survive or fail under energy
stress, and sometimes merge into larger entities.

## Why This Version Exists

The earlier PD-NCA visualization produced spatial patterns, but it did not say
much beyond "the strongest energy phenotype spreads." This version is designed
to produce interpretable research questions:

- Which sharing norms survive in residential, commercial, mixed-use, and
  critical civic land-use tissues?
- Does cooperation cluster around critical loads?
- Do large-area commercial buildings become generators for surrounding cells?
- Does mixed land use stabilize cooperation and critical-load survival?

## State Variables

Each grid cell is one building agent with:

- slowly changing `landuse`
- evolving `norm`
- `entity_id`, which can be shared by multiple merged buildings
- `reputation`
- `health`
- `storage`
- hourly `demand`
- hourly `solar generation`
- `critical` status
- recent `payoff`

## Land-Use Tissues

| Key | Land use | Role |
| --- | --- | --- |
| R | residential | evening-biased demand, smaller roof area |
| C | commercial | daytime demand, larger roof area |
| M | mixed use | blended demand and blended solar potential |
| K | critical civic | critical-load survival target |

Land use changes more slowly than norms or hourly energy service. It is the
"tissue" layer, but it can adapt under sustained neighborhood pressure.

## Norm Tribes

The tribes are leading-eight-inspired indirect reciprocity norms. They are not
literal land-use types. They are protocols for deciding whether to share surplus
energy and how to update reputation afterward.

| Key | Norm | Intuition |
| --- | --- | --- |
| ALLC | generous | share broadly when possible |
| SELF | selfish | share only under strong self-interest |
| DISC | standing | share with good-reputation or critical receivers |
| SJ | stern judging | punish helping bad actors more strongly |
| SHUN | shunning | cooperate only with good-reputation receivers |
| CRIT | critical first | prioritize critical loads |
| MKT | market | share when the local deficit/payoff is high |
| LOCAL | neighbor loyal | favor nearby good-reputation receivers |

This is inspired by the logic of Ohtsuki and Iwasa's "leading eight": donor
actions are judged in relation to the recipient's reputation. In this urban
translation, a donor is a building with surplus energy, a recipient is a
building with deficit, cooperation means sharing energy, and reputation records
whether a building/protocol is considered reliable.

## Real Data Inputs

Demand profiles:

```text
../data/energy_profiles_clean/energy_profiles_hourly_used.csv
```

Building metadata for roof-area estimates:

```text
../data/energy_profiles_clean/building_energy_metadata.csv
```

San Francisco climate data:

```text
C:/UCBcourses/RESEARCH/Lau grant/solar/validation/validation/step 4 PV/USA_CA_San.Francisco-Presidio.994016_TMYx.2009-2023/USA_CA_San.Francisco-Presidio.994016_TMYx.2009-2023.epw
```

Solar generation uses the simplified PV logic:

```text
estimated roof area * hourly SF radiation * PV efficiency correction
```

It is intentionally based on area, not divided by the building's own demand,
because large-area buildings can share surplus with surrounding cells.

## Dynamics

Each hour:

1. Buildings generate solar energy and serve their own demand.
2. Deficit buildings ask nearby surplus buildings for help.
3. Donors decide whether to share based on their current norm.
4. Donor reputation updates according to the norm's assessment rule.
5. Unserved demand damages health, especially around critical loads.
6. Buildings imitate nearby norms with higher payoff, reputation, and health.
7. Occasional mutation introduces alternative norms.
8. Stressed parcels can slowly transition toward better-performing nearby
   land-use tissue.
9. Neighboring buildings can merge into a larger entity. A merged entity shares
   one organization-level norm and can internally reallocate energy with lower
   loss.

A solar shock is applied during the middle of the run to test resilience.

## Run

```powershell
.\.venv\Scripts\python.exe scripts\run_landuse_norm_model.py --config configs\landuse-norm-final.json
```

## Scenario Sweep

Run a parameter sweep across shock severity, sharing radius, rebuild rate, and
random seed:

```powershell
.\.venv\Scripts\python.exe scripts\run_scenario_sweep.py --config configs\landuse-norm-final.json --steps 336 --out-dir results\scenario_sweep
```

The sweep is intended to compare regimes, not to make one final animation for
every parameter set. It writes a compact comparison package:

- `scenario_sweep_results.csv`
- `scenario_sweep_results.json`
- `scenario_sweep_summary.png`
- `scenario_sweep_outcome_map.png`
- `scenario_sweep_representative_snapshots.png`

## Outputs

The final result package is:

```text
results/final_landuse_norm
```

It contains:

- `landuse_norm_evolution.gif`
- `landuse_norm_contact_sheet.png`
- `landuse_norm_final_snapshot.png`
- `landuse_norm_metrics.png`
- `landuse_norm_results.html`
- `landuse_norm_metrics.csv`
- `landuse_norm_metrics.json`

## What To Look For

Read the outputs as a comparison between three layers:

- land-use tissue: what the city physically is
- norm tribe: what cooperation protocol buildings adopt
- service/health: whether energy demand and critical loads survive

The most useful result is not the final color pattern alone. The useful result
is the relation between land use, norm frequency, cooperation rate, and
critical-load survival over time.
