# SF Terrain Energy Growth ABM

This repository contains a minimal building-level agent-based model for asking
whether block-scale coordination can emerge from local energy exchange, without
predefining blocks, hubs, districts, or a grid hierarchy.

The current SF experiment starts from an empty terrain grid. Buildings appear as
residential startups, choose sites, buy assets, request energy when short, and
offer surplus when they have it. The only building functions are:

- residential
- commercial

PV, batteries, floors, and cables are purchasable assets, not extra building
types.

## Why Block Scale Can Appear

Block scale can appear because energy cannot move arbitrarily across the city.
A building in need broadcasts a request. A donor can make an offer if it has
surplus. If requester and donor are not direct neighbors, the transfer must pass
through adjacent buildings. Those intermediate buildings can collect transit tax
or form paid agreements.

That creates a local network externality:

- one cable helps one pair of neighbors;
- several repeated local agreements make a linked building cluster;
- linked clusters become useful corridors for future exchanges;
- corridors with more traffic accumulate more value;
- once several local clusters connect, a higher-level cluster network can be
  read from the agreement graph.

So the block is not an input. It is a post-run pattern in the paid agreement
graph. In the hierarchy visualization:

- L1 is individual buildings only.
- L2 is linked building clusters. A building with no agreement link is not an L2
  cluster member.
- L3 is a network among L2 clusters, created by paid agreement links crossing
  cluster boundaries.

## Terrain, Solar, And Property Value

The SF grid uses terrain and climate as two explicit assumptions:

- height/elevation is shown as context only;
- property value is based on slope, not elevation;
- flatter land has higher property value because it is easier to build on;
- the property value can also be locally priced from surrounding values;
- if nearby buildings die or lose service, local property value can fall;
- solar potential is based on hourly 2025 climate plus a west-to-east SF fog
  gradient.

The model intentionally uses one property-value factor rather than multiple
pre-engineered land-use scores.

## Energy Exchange Rule

There is no distance-search parameter in this version.

Energy exchange follows local transmission logic:

1. A building in need of energy broadcasts a request.
2. Buildings with surplus can make offers.
3. Direct neighbors can exchange through one adjacent agreement.
4. Non-neighbor exchange must pass through neighboring buildings.
5. Transit buildings can collect a fee or form an agreement.
6. Repeated paid adjacent agreements are read afterward as emergent clusters.

This keeps the decision rule local while allowing larger-scale structure to
emerge from repeated local dependence.

## Commercial Conversion

All buildings start as residential. A residential building may upgrade to
commercial only when money, service, and property value make the commercial
payback plausible. Commercial growth is capped near a 1:9 commercial to
residential ratio so that commercial cells are present but not pre-zoned.

Commercial and residential buildings use different income scaling:

- residential has a smaller property-value scaling factor;
- commercial has a larger property-value scaling factor.

## Key Outputs

Main shareable result:

```text
outputs/sf_terrain_energy_growth/sf_energy_growth_all_results_standalone.html
```

Hierarchy diagram:

```text
outputs/sf_terrain_energy_growth/sf_energy_growth_hierarchy_canopy.png
```

Assumption timeline:

```text
outputs/sf_terrain_energy_growth/sf_assumptions_terrain_climate.html
```

Other output files:

```text
outputs/sf_terrain_energy_growth/sf_energy_growth_summary.json
outputs/sf_terrain_energy_growth/sf_energy_growth_metrics.csv
outputs/sf_terrain_energy_growth/sf_energy_growth_cells.csv
```

## Run

Run the SF terrain energy growth experiment:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sf_terrain_energy_growth.py --out-dir .\outputs\sf_terrain_energy_growth
```

Generate the terrain/climate assumptions timeline:

```powershell
.\.venv\Scripts\python.exe .\scripts\generate_sf_assumptions_timeline.py --day-of-year 209 --out-dir .\outputs\sf_terrain_energy_growth
```

## Data

When available, demand profiles are sampled from:

```text
../data/energy_profiles_clean/energy_profiles_hourly_used.csv
```

If the large profile file is missing, the script falls back to small synthetic
residential and commercial daily load shapes so the experiment remains runnable.

Climate data is cached from NASA POWER 2025 hourly GHI and temperature for San
Francisco. Elevation defaults to a synthetic SF hills fallback unless live
elevation fetching is explicitly enabled.
