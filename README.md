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

## Agent Attributes

Each agent is one building cell. The important attributes are:

| Attribute | Meaning | Current setting |
| --- | --- | --- |
| `function` | Building use type | `residential` or `commercial` only |
| `money` | Cash available for investment and agreements | New residential startups receive about `$10,000` with `$1,500` spread |
| `norm` | Sharing behavior | Fixed generous-only rule: when a building has surplus, it can offer energy to a requester |
| `property_value` | Land/building value factor | Based on local slope; flatter cells have higher value |
| `solar_factor` | Local solar potential modifier | West-side fog lowers potential; east-side sun and south-facing terrain increase it |
| `pv_kw` | Owned PV capacity | Purchased by the building as a capital asset |
| `battery_kwh` | Owned battery capacity | Purchased by the building as a capital asset |
| `floors` | Built floor intensity | Purchased upgrade; steeper land is more expensive to intensify |
| `cables` | Paid adjacent transmission agreements | Each cable is one paid agreement with a neighboring building |
| `service_ratio` | Lifetime served energy / lifetime demand | Affects income and commercial upgrade readiness |

The current property-value scaling factors are intentionally simple:

| Building function | Property-value income scaling |
| --- | ---: |
| residential | `0.45` |
| commercial | `1.25` |

Commercial buildings are therefore more sensitive to property value than
residential buildings, but commercial use is not pre-zoned.

## Agent Equations

Each occupied cell is a building agent `i`. The code still stores property value
as `productivity`; below it is written as `P_i` to match the model concept.

| Quantity | Equation |
| --- | --- |
| Local slope | `s_i = sqrt((z_east - z_west)^2 + (z_south - z_north)^2)` |
| Property value | `P_i = clip(0.72 + 0.34 * (1 - s_i / s_95) + 0.04 * east_i, 0.55, 1.10)` |
| Local solar factor | `F_i = clip((0.58 + 0.52 * east_i + 0.08 * elev_norm_i) * (1 + 0.07 * southness_i), 0.45, 1.24)` |
| Residential entry score | `E_i = 0.55 * norm(F_i) + 0.45 * norm(P_i) + epsilon_i`, where `epsilon_i ~ U(-0.035, 0.035)` |
| Hourly demand | `D_i(t) = load_type_i(t) * floors_i * scale_type_i` |
| Demand scale | `scale_res = 0.82`; `scale_com = 0.12` |
| Hourly PV generation | `G_i(t) = pv_kw_i * climate_solar(t) * F_i` |
| Direct self-service | `S_i^self(t) = min(D_i(t), G_i(t))` |
| Battery charge | `B_i(t+1) = min(Bcap_i, B_i(t) + 0.92 * max(G_i(t) - D_i(t), 0))` |
| Battery discharge | `discharge_i(t) = min(B_i(t), max(D_i(t) - G_i(t), 0) / 0.92)` |
| Served after battery | `S_i(t) = S_i^self(t) + 0.92 * discharge_i(t) + imports_i(t)` |
| Surplus offer | `Q_i(t) = max(G_i(t) - D_i(t) - charge_i(t) / 0.92, 0)` |
| Service ratio | `R_i = lifetime_served_i / lifetime_demand_i` |
| Property-value income effect | `V_i = 1 + alpha_type_i * (P_i - 1)` |
| Income | `Y_i(d) = base_type_i * floors_i * V_i * clip(R_i(d), 0.15, 1.05)` |
| Daily money update | `M_i(d+1) = M_i(d) + Y_i(d) - O&M_i(d) - 0.18 * unmet_i(d) - investments_i(d) - agreement_costs_i(d) + transit_fees_i(d)` |

Income parameters:

| Type | `base_type` | `alpha_type` |
| --- | ---: | ---: |
| residential | `125` | `0.45` |
| commercial | `520` | `1.25` |

Transmission and agreement equations:

| Quantity | Equation |
| --- | --- |
| Path efficiency | `eta_path = 0.965 ^ hops` |
| Delivered energy | `delivered = min(deficit_requester / eta_path, surplus_donor) * eta_path` |
| New agreement cost | `C_path = 650 * missing_edges(path)` |
| Transit fee | `T_path = 0.012 * delivered * transit_nodes(path)` |
| Feasible transfer | transfer occurs only if `M_requester >= C_path + T_path` |
| L2 component | connected component of buildings joined by paid adjacent agreement edges |
| L3 graph | graph among L2 components when agreement links cross component boundaries |

Commercial conversion is a decision rule, not a land-use assumption:

```text
upgrade if:
  function_i = residential
  service_i > 0.24
  money_i > reserve_i + commercial_upgrade_cost_i
  payback_days_i < 260
  random() < conversion_chance_i
```

where:

```text
payback_days_i = commercial_upgrade_cost_i / max(1, expected_commercial_income_i - current_residential_income_i)
conversion_chance_i = clip(0.03 + 0.62 * commercial_pressure + 0.22 * readiness_i, 0, 0.88)
readiness_i = 0.42 * norm(P_i) + 0.26 * service_i + 0.20 * cash_readiness_i + 0.12 * cable_degree_i / 4
```

## Initial Settings And Assumptions

The experiment begins with an empty SF terrain grid:

| Setting | Current value |
| --- | --- |
| Grid | `28 x 28` over a San Francisco bounding box |
| Initial buildings | none; the first frame is empty terrain |
| Buildable cells | determined from the SF terrain mask |
| New entry | about one residential startup per day |
| Startup site choice | `0.55` solar opportunity + `0.45` property value + small noise |
| Solar generation potential | NASA POWER 2025 hourly GHI and temperature at San Francisco |
| Local solar modifier | west-to-east fog gradient plus south-facing aspect adjustment |
| Property value | based on slope, not elevation |
| Property-value neighborhood effect | property value can be priced from surrounding values |
| Death/service feedback | if nearby buildings die or lose service, local property value can fall |
| Building functions | residential first; commercial only by upgrade |
| Commercial target | capped near `1 commercial : 9 residential` |

The current asset and transmission costs are:

| Asset / action | Cost |
| --- | ---: |
| PV | `$2,410 / kW` |
| Battery | `$4,660 / 10 kWh` |
| Cable / adjacent agreement | `$650` |
| Floor upgrade | `$6,000` |
| Commercial upgrade | `$3,600` |
| Transmission efficiency | `0.965` per edge |
| Transit tax | `$0.012 / kWh` per transit node |

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
