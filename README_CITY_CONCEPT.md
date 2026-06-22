# Cities in a Petri Dish Extension

This fork keeps Sakana AI's Petri Dish NCA mechanics, but gives the substrate
an urban-energy interpretation.

## Concept Mapping

The first three NCAs are treated as competing urban tissues:

| NCA | City role | Abstract behavior |
| --- | --- | --- |
| 0 | solar / low-rise support tissue | high roof-to-load ratio, surplus support |
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

This does not yet load the San Francisco building energy-profile CSVs. Those
profiles are still used by the separate 10x10 HTML visualizer. The next clean
step would be to project hourly demand and solar potential into the NCA state
channels or into the background environment, then score outcomes by
critical-load survival rather than only area growth.
