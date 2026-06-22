# Cities in a Petri Dish Extension

This fork keeps Sakana AI's Petri Dish NCA mechanics, but gives the substrate
an urban-energy interpretation.

## Concept Mapping

The first three NCAs are treated as competing urban tissues:

| NCA | City role | Abstract behavior |
| --- | --- | --- |
| 0 | solar / low-rise support tissue | large usable roof area, surplus support |
| 1 | dense load / commercial tissue | higher demand, stronger growth pressure |
| 2 | storage / critical-support tissue | buffering, critical-load survival |

The original PD-NCA model already has the core ingredients we need: several
adaptive NCAs share one grid, compete through attack and defense channels, and
are pressured by a learnable background environment. The city extension changes
the initial conditions and background pressure so those dynamics become a small
urban-energy experiment instead of a purely abstract substrate.

## What Changed

- `seed_dist = "city_anchors"` places each tissue near a persistent city niche
  instead of scattering all seeds uniformly.
- `city_mode = true` gives each NCA a structured initial state rather than a
  random seed vector.
- `city_daily_cycle = true` turns the background competitor into a simple daily
  pressure field. Daylight favors defense / solar support; night favors attack /
  stress pressure.
- `city_hypercycle_gamma` adds a weak cyclic dependency among the three tissues,
  inspired by the cooperative/competitive loop we discussed for the city model.
- `city_profiles_csv` loads the real hourly building energy profiles and turns
  them into 24-hour demand curves used during training.
- `city_solar_epw` loads San Francisco EPW climate data and turns hourly GHI
  plus air temperature into a simplified PV generation curve.
- `city_building_metadata_csv` estimates usable roof area as
  `bldgsqft / floors * 0.092903 * city_roof_usable_fraction`.
- `city_energy_weight` makes profile-based service survival part of the growth
  objective. Tissues with enough solar / storage to serve their current demand
  get stronger survival weight.

## Run

Tiny smoke run:

```bash
uv run python src/train.py --config configs/city-petri-smoke.json
```

Full city concept run:

```bash
uv run python src/train.py --config configs/city-petri.json
```

If `uv` is not available but the dependencies are installed:

```bash
python src/train.py --config configs/city-petri.json
```

## Current Scope

Demand profiles are read from
`../data/energy_profiles_clean/energy_profiles_hourly_used.csv`. The loader
samples residential and commercial profiles, compresses each yearly profile into
an average 24-hour curve, and maps those curves onto:

- NCA 0: residential / solar-support demand
- NCA 1: commercial / dense-load demand
- NCA 2: mixed demand with high storage and critical-load weighting

Solar generation now follows the simplified notebook logic:

```text
roof_area_estimate * SF climate radiation * PV efficiency correction
```

The EPW climate file supplies hourly `ghi` and `temp_air`. By default the config
uses the June 13 San Francisco-Presidio weather slice, matching the reference
notebook. The PV output is normalized before training so it acts as a stable
survival signal rather than an unbounded physical unit. It is not divided by the
building's own average load: large-area buildings keep their role as
neighborhood-scale generators that can share energy with surrounding cells.

This is still a first bridge, not the final empirical model. It uses real demand
profiles, real SF radiation shape, and metadata-based roof-area estimates, while
storage capacity remains an abstract city-tissue parameter. The next step is to
assign explicit building metadata to cells and score final resilience by
critical-load survival.
