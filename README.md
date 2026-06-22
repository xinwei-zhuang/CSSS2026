# Petri City PD-NCA

Petri City PD-NCA is an experimental artificial-life model of urban energy
tissues. It adapts Sakana AI's Petri Dish Neural Cellular Automata framework to
ask how building-like agents compete, cooperate, die, and survive under
urban-energy constraints.

The model is not a conventional supervised ML model. It does not learn from
target images or labels. Instead, multiple neural cellular automata grow in a
shared grid and are trained through a survival objective shaped by real demand
profiles, area-based solar generation, storage support, and local competition.

## Model Setting

The grid contains three competing urban tissues:

| NCA | Role | Energy interpretation |
| --- | --- | --- |
| 0 | Solar / low-rise support tissue | Roof-area-driven generation and local support |
| 1 | Dense-load / commercial tissue | Higher demand but potentially large solar area |
| 2 | Storage / critical-support tissue | Buffering and critical-load survival |

Each cell has aliveness channels, state channels, and hidden channels. The three
NCAs compete through the original PD-NCA attack/defense mechanism, while a
background environment applies time-varying energy pressure.

## Real Data Inputs

Demand comes from:

```text
../data/energy_profiles_clean/energy_profiles_hourly_used.csv
```

Building metadata comes from:

```text
../data/energy_profiles_clean/building_energy_metadata.csv
```

Solar radiation comes from the San Francisco-Presidio EPW climate file used in
the reference PV notebook:

```text
C:/UCBcourses/RESEARCH/Lau grant/solar/validation/validation/step 4 PV/USA_CA_San.Francisco-Presidio.994016_TMYx.2009-2023/USA_CA_San.Francisco-Presidio.994016_TMYx.2009-2023.epw
```

The simplified solar calculation is:

```text
estimated roof area * hourly SF radiation * PV efficiency correction
```

Roof area is estimated as:

```text
bldgsqft / floors * 0.092903 * usable_roof_fraction
```

Solar generation is intentionally based on area, not on each building's own
load. A large-area building can act as a neighborhood-scale generator and share
energy with surrounding cells.

## Training Objective

The training signal combines:

- territory survival / growth from the original PD-NCA setup
- weak cyclic dependency among the three tissues
- profile-based service ratio from demand, solar generation, and storage support
- critical-load weighting for the storage-support tissue

The logged `Service: [...]` values show the energy survival signal for the three
tissues at each logged epoch.

## Setup

Dependencies are installed in the local virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

If the environment needs to be recreated:

```powershell
uv sync
```

## Run Training

Final reproducible run:

```powershell
.\.venv\Scripts\python.exe src\train.py --config configs\city-petri-final.json
```

Small smoke test:

```powershell
.\.venv\Scripts\python.exe src\train.py --config configs\city-petri-final.json --epochs 12
```

## Render Results

After training, render a checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\render_city_pdnca.py --checkpoint results\final_area_solar_checkpoint --out-dir results\final_area_solar_visualization --steps 36 --sample-every 1
```

The renderer writes:

- `city_pdnca_final.gif`
- `city_pdnca_contact_sheet.png`
- `city_pdnca_final_snapshot.png`
- `city_pdnca_results.html`
- `city_pdnca_metrics.json`

## Final Results

The final cleaned result package is stored under:

```text
results/
```

The latest checkpoint is:

```text
results/final_area_solar_checkpoint
```

The latest visualization package is:

```text
results/final_area_solar_visualization
```

## Current Scope

This version uses real demand profiles, real San Francisco radiation shape, and
metadata-estimated roof area. Storage is still an abstract tissue-level support
term. The next research step is to assign explicit building metadata to each
cell and evaluate resilience directly through critical-load survival after
shock events.
