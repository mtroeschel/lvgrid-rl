# System architecture: an RL training environment for autonomous low-voltage grid control

**Version:** 1.0 — consolidated after M2. Supersedes the German draft versions
0.1–0.5; all decisions taken there are folded in, and everything that M1 and M2
disproved has been corrected against the data.

**Stack:** Python ≥ 3.11 · pandapower + SimBench · Gymnasium · Stable-Baselines3
(+ sb3-contrib) · PyTorch · Hydra/OmegaConf · pandas/pyarrow · Optuna · uv

---

## 1. Purpose and design principles

### 1.1 The control task

One or more RL agents control a low-voltage grid such that

- the voltage band is respected according to the **percentile criterion of
  EN 50160** — assessed on ten-minute mean values over weekly intervals, not per
  time step (§6.6),
- lines and the distribution transformer are not overloaded,

using the following actuators:

| Actuator | Manipulated variable | Typical degrees of freedom |
|---|---|---|
| PV system | active power curtailment, reactive power | `p_curtail ∈ [0,1]`, `q ∈ [-Q_max, Q_max]` |
| Battery storage | charge/discharge power, optionally Q | `p ∈ [-P_max, P_max]`, state-of-charge dynamics |
| Heat pump | block/release, power limit | binary, discrete (SG-Ready) or continuous |
| EV charge point | charging power per session | `p ∈ [0, P_max]` or `{0, 4.2 kW, P_max}` |

Constraints and comfort objectives — energy delivered by departure, building
temperature, storage state — are part of the objective function, not of the grid
restriction. That is what makes the task interesting as a sequential decision
problem.

### 1.2 Design principles

1. **Strict layering.** Data → scenario → grid model → asset models → Gym
   environment → agent → evaluation. Each layer can be replaced without changing
   the ones above it.
2. **Configuration rather than code.** Every experiment is fully described by a
   YAML configuration (Hydra). No experiment is defined by editing source.
3. **Everything random is seeded and recorded.** A run is determined by
   `(code_commit, config_hash, data_manifest_hash, seed)`.
4. **Reference methods and RL policies share one controller interface.**
   Rule-based methods, OPF, MPC and RL then run through exactly the same
   evaluation path.
5. **The agent sees only what a real controller could see.** Observability is
   configurable: full state as a research upper bound, realistic measurement
   points otherwise.
6. **Simulation and control step sizes are separate.** `sim_dt ∈ {1, 2, 5, 10}
   min`, `control_dt = k · sim_dt`. The restriction follows from the ten-minute
   averaging interval of EN 50160 (§6.6).
7. **The single agent is a special case, not a special path.** The environment
   always holds a per-actuator structure; the centralised single agent is a view
   over one partition containing all actuators. Multi-agent operation is later a
   configuration change, not a rebuild (§6.3).
8. **The agent is an untrusted component.** Grid integrity is not guaranteed by
   the policy but by a separate, verified certifier that the agent cannot
   influence, parameterise or bypass (§7). Everything the agent does is
   optimisation inside a provably admissible set.

---

## 2. Overall architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L6  Experiment and orchestration                                            │
│     Hydra configs · run registry · sweeps (Optuna) · seed management        │
├──────────────────────────────────────────────────────────────────────────────┤
│ L5  Evaluation and visualisation                                            │
│     KPI engine · statistical aggregation (IQM/bootstrap) · plots · replay   │
├──────────────────────────────────────────────────────────────────────────────┤
│ L4  Agents and reference methods       (shared controller interface)        │
│     SB3 factory · policies/extractors · baselines · OPF · MPC               │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  Gymnasium environment                                                   │
│     ObsBuilder · ActionMapper · RewardComposer · EpisodeSampler · wrappers  │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  Asset and flexibility models                                            │
│     PV · BESS · heat pump (+ thermal store) · EVSE (+ session model)        │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  Grid and physics                                                        │
│     loader · scenarios · power flow engine · PQ aggregation · metrics       │
├──────────────────────────────────────────────────────────────────────────────┤
│ L0  Data layer                                                              │
│     source adapters · resampling · validation · Parquet cache · manifest    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Repository layout

```
lvgrid-rl/
├── pyproject.toml            # uv, lock file committed
├── configs/                  # Hydra: grid, scenario, data, env, agent, baseline
├── src/lvgrid_rl/
│   ├── core/                 # schemas, protocols, units, information ordering
│   ├── data/                 # L0
│   ├── grid/                 # L1
│   ├── components/           # L2
│   ├── env/                  # L3
│   ├── agents/  baselines/   # L4
│   ├── eval/  viz/           # L5
│   └── experiment/           # L6
├── scripts/                  # prepare_data, survey_grids, run_baseline, train
├── tests/
└── results/                  # run artefacts, not versioned
```

---

## 3. L0 — Data layer

### 3.1 Tasks

1. **Acquisition** (`data/sources/`): one adapter per source, uniform return.
2. **Validation**: monotonic timestamps, UTC, quantified gaps, checked units,
   plausibility bounds.
3. **Gap filling**: short gaps linear, longer ones from the same time of day on
   the preceding matching weekday; every substitution is recorded in a mask.
4. **Resampling onto `sim_dt`** — the critical step, see §3.3.
5. **Normalisation**: profiles are stored per-unit and scaled per asset only in
   the scenario.
6. **Cache and manifest**: Parquet keyed by a hash over source, version, period,
   step size and policies; `manifest.json` holds a SHA-256 over the content.

### 3.2 Usable time series

**Primary source: SimBench.** Consistent annual series for load, generation and
storage at **15 minutes for a full year (2016)**, available through
`simbench.get_simbench_net()`. Grid and time series are matched and already
linked. Verified against SimBench 1.6.1, grid `1-LV-rural1`:

| Profile prefix | Meaning |
|---|---|
| `H0-*`, `L1-*`, `L2-*`, `G*` | household, agriculture, commercial |
| `Air_*`, `Soil_*` | air- and ground-source heat pumps, bivalent operating mode in the name (`Alternative`, `Parallel`, `Semi-Parallel`) |
| `HLS_*_3.7`, `_11.0`, `_22.0` | **charge points**, connection power in kW in the name |
| `PV5`, `PV6`, `PV8` | photovoltaics |
| `Storage_*` | home storage, coupled to a PV and a load profile |

Load profiles come as separate `*_pload` and `*_qload` series. Reactive power is
carried as its own column group, not derived from a power factor: `H0-A_qload`
ranges from −0.26 to 1.20, so capacitive behaviour is in the data.

The charge point profiles are **fixed load curves, not a description of
flexibility**: arrival, departure and energy demand per session cannot be
reconstructed. emobpy remains necessary for the control task; until then charge
points are uncontrollable load.

**Development stages.** The control task only exists from stage 2 onwards. For
`1-LV-rural1`: `--0-sw` has 13 households and 4 PV systems and nothing else;
`--1-sw` adds 4 storage units and one heat pump; `--2-sw` has 28 loads including
8 heat pumps and 7 charge points, 8 PV systems and 5 storage units.

**High-resolution household load**

- **HTW Berlin** — 74 single-family households, annual profile on a
  **one-second** basis, with phase-resolved active and reactive power. The right
  source for realistic coincidence and peak effects at `sim_dt = 1 min`.
- **WPuQ (ISFH Hameln)** — 38 single-family houses in Lower Saxony, 2018–2020,
  **native resolution 10 s**, provided at 10 s / 1 min / 15 min / 60 min. Holds
  **household and heat pump load separately per building**, plus the substation
  aggregating 68 households, plus weather. The most valuable external dataset
  here, because it supplies a real aggregation scale.
- **OPSD Household Data (CoSSMic, Konstanz)** — 11 households, 3-minute then
  **one-minute** intervals, including PV, some heat pumps and EV charging.

**Heat pumps**

- **WPuQ** (above) — measured heat pump load at 10 s.
- **HEAPO** — 1,408 real heat pump households in the canton of Zurich, smart
  meter data at **15 minutes**, 2018-11 to 2024-03, with metadata and weather.
- **when2heat (OPSD)** — synthetic heat demand and **COP** series for 28
  European countries, **hourly**. Not usable as a load profile, but a good
  source for COP characteristics and the annual shape of heat demand.

**Electric mobility.** What the agent needs is *flexibility*, not a load profile:
arrival, departure, energy demand and maximum power per session.

- **emobpy (DIW Berlin)** — generates four series per vehicle from the German
  *Mobilität in Deutschland* survey: mobility/location, driving consumption,
  **grid availability** and grid charging demand, by default at **15-minute**
  steps. The cleanest source for the flexibility description needed here.
- **ElaadNL Open Data** — normalised distributions for **arrival time,
  connection duration, energy demand and charging curves**, separately for
  private, workplace and public charging. For calibrating and validating the
  session generator.
- **ACN-Data (Caltech/JPL)** — session-level charging data, workplace context,
  US. A second, independent behavioural pattern for generalisation tests.

**PV and weather**

- **DWD Open Data / CDC** — station measurements of global irradiance and other
  quantities at **10 minutes**. Directly usable for `sim_dt ≥ 10 min`; for one
  minute, interpolation is required and must be documented as a limitation.
- **PVGIS / CAMS Radiation Service** — satellite-based irradiance, 15-minute and
  1-minute products, accessible through `pvlib.iotools`.
- **PV modelling** with `pvlib` rather than ready-made generation profiles,
  which allows orientation and shading to be varied per system.

**Reference and fallback**

- **BDEW standard load profiles** via `demandlib`, 15 minutes — as a reference
  case and for "SLP vs. measurements" sensitivity studies.
- **Marktstammdatenregister (MaStR)** — real installed capacity distributions
  for parameterising growth scenarios (not time series).

### 3.3 Resampling onto a uniform step size

Target step size `sim_dt ∈ {1, 2, 5, 10} min`, default 5 min. **15 minutes is
deliberately not permitted:** EN 50160 assesses ten-minute mean values, and 15
does not divide 10, so an assessment window could not be formed from simulation
steps without interpolation. Per quantity kind:

| Quantity | Downsampling | Upsampling |
|---|---|---|
| Power (load, generation) | mean (energy preserving) | piecewise constant |
| Energy meter reading | difference over the interval | monotone interpolation, then difference |
| Global irradiance | mean | clear-sky-index interpolation, not linear on GHI |
| Temperature | mean | linear |
| Binary availability (EV plugged in) | majority vote | piecewise constant |
| EV sessions | **not resampled** — events stay events; arrival and departure are rounded onto the grid, energy demand is preserved | – |
| States, state of charge | end value of the interval | linear |

Two consequences that were confirmed in M1:

- **SimBench must be upsampled, and that adds no information.** The 15-minute
  originals are already smoothed; going to 5 minutes piecewise constant preserves
  energy but does not recover peaks. Where short-term dynamics matter, WPuQ or
  the HTW profiles are required. This coupling between the voltage criterion and
  the choice of data source is easy to miss.
- **Time zones are a real trap.** The SimBench timestamps are German local time
  *including* daylight saving. Read naively the index is neither monotonic nor
  unique — four quarter-hours are missing on 2016-03-27, four occur twice on
  2016-10-30. Internally everything is UTC; calendar features are derived from
  local time.

### 3.4 Scenarios

Scenario construction lives in L1 (§4.4), because it modifies the grid rather
than the time series. The data layer supplies the normalised profiles and the
asset ratings; the scenario decides how they are scaled and where they sit.

---

## 4. L1 — Grid and physics

```
grid/
├── loader.py      # SimBench loading, ZIP repair, connection points, index caches
├── scenario.py    # declarative scenario modifications, scenario library
├── powerflow.py   # PandapowerEngine with run and run_hypothetical
├── pq.py          # ten-minute averaging, EN 50160 assessment, budget tracking
└── metrics.py     # instantaneous violation metrics, P4 fallback report
```

### 4.1 Loading and the ZIP repair

SimBench grids do **not** converge with pandapower 3.x out of the box — not only
at development stage 2, but for every low-voltage grid including the trivial
`--0-no_sw`. simbench 1.6.1 leaves the ZIP load model columns
(`const_z_p_percent` and its three siblings) as `NaN`; pandapower 3.3.3
propagates that into the Jacobian, which becomes exactly singular.

The failure mode is misleading. pandapower reports "Power Flow nr did not
converge", which suggests an infeasible operating point, and `pp.diagnostic`
blames implausible impedance values and non-continuous bus indices — both red
herrings. The discriminating observation is that `fdbx` and `fdxb` converge while
`nr` does not, and those two ignore the voltage-dependent load model.

`fix_zip_load_model()` fills the columns with zero, which means a constant-power
load — pandapower's own default for a new load. Two tests guard this: one asserts
the defect itself, so it turns red once simbench fills the columns and the repair
can be removed; one asserts the repair is idempotent.

### 4.2 Connection points

EN 50160 applies at the point of common coupling, not at every node. A bus counts
as a connection point if at least one customer-side element is attached — load,
generator **or** storage. Generators are not optional: a PV system or battery on a
bus without load is just as much a point of common coupling, and such buses tend
to sit at the feeder end where the voltage criterion becomes binding. For
`1-LV-rural1--2-sw` this gives 13 of 15 buses.

Equivalent infeeds that model an adjacent grid rather than a customer are excluded
through a per-grid list in `configs/grid/*`, never guessed from element names. The
resulting bus list is checked once per grid and then fixed with a hash, so a silent
change cannot shift the KPI unnoticed. The conservative "all buses" variant is
computed alongside in evaluation; the difference between the two figures is the
sensitivity to this modelling decision and belongs in the results report.

### 4.3 Power flow engine

`PandapowerEngine` is a thin facade over `pp.runpp` that (a) resolves index
arrays once at load time rather than per step, (b) warm-starts from the previous
solution, (c) treats non-convergence as a defined return value rather than an
exception — with a flat-start retry first — and (d) can be called
**hypothetically** without side effects on the live grid.

The hypothetical call is invariant I5. It uses a dedicated shadow grid rather
than snapshot-and-restore, which is both faster and safer: there is no restore
step that could be incomplete, and pandapower writes a number of result tables
whose full extent is easy to miss.

**Measured runtime** on `1-LV-rural1--2-sw` (15 buses): 23 ms per warm power flow
step, about 45 ms per baseline step including a shadow power flow. A full year at
five minutes is roughly 26 minutes. This is the figure against which the surrogate
mode below has to justify itself, and it is the reason the training step size is
chosen deliberately (§12, M3).

**Optional surrogate mode.** A trained neural network or a linearised sensitivity
matrix (∂U/∂P, ∂U/∂Q) as a fast stand-in for pre-training, with subsequent
fine-tuning on the exact AC power flow. Implemented as a separate
`PowerFlowEngine`, so the same environment can use either.

### 4.4 Scenarios

A survey of all six SimBench low-voltage grids at development stage 2 produced
**no voltage band violation at all** — the worst value across the year is about
1.083 pu against a 1.10 limit, and EN 50160 averages over ten minutes on top of
that. What binds is thermal: 209 % transformer loading in `1-LV-rural1`, 123 %
line loading in `1-LV-semiurb4`. Reproduce with `scripts/survey_grids.py`.

Also worth recording, because it contradicts the intuition that guided the initial
grid choice: `1-LV-rural1` is **not** a long-feeder grid. It has 0.56 km of line in
total across 15 buses, the longest single line being 137 m. It is electrically
short and stiff; the bottleneck is the 160 kVA transformer against 468 kW of
installed PV.

The published scenarios are therefore a sound comparability baseline but not a
sufficient test bed for a controller whose stated purpose includes voltage band
violations. `ScenarioSpec` provides two declarative levers, both logged in the run
manifest so the distance from the published reference stays visible:

- **the point of common coupling** — slack voltage, transformer tap position,
  station rating. The physically more honest lever: a weaker upstream connection
  or an unfavourable tap produces voltage problems without inventing installed
  capacity that nobody has.
- **penetration and sizing** — scaling factors per asset category.

Deliberately not a random scenario generator. Every modification is an explicit
number, because a scenario that cannot be written down cannot be reported either.
`apply_scenario` returns how many elements it touched, so a scenario that silently
changes nothing — because a category is absent from the grid — is visible rather
than guessed at.

Measured on `1-LV-rural1--2-sw`:

| Scenario | vmax | EN 50160 pass rate | budget max | overloaded steps |
|---|---|---|---|---|
| `reference` | 1.0747 | 1.0000 | 0.00 | 8.7 % |
| `weak_connection` | 1.0941 | 1.0000 | 0.00 | 10.7 % |
| `moderate_growth` | 1.1064 | **0.8462** | 0.72 | 12.6 % |
| `voltage_stress` | 1.1588 | **0.0000** | 3.06 | 18.9 % |
| `undervoltage_stress` | 0.9275 | 1.0000 | 0.00 | 7.9 % |

`moderate_growth` is the working scenario: 15 % of bus-weeks fail with a peak
budget utilisation of 0.72 — not trivially satisfied, not hopeless. Note that
raising the point of common coupling alone is not enough; the percentile filter
absorbs it. Only together with growth does the criterion start to bind.
`voltage_stress` and `undervoltage_stress` go beyond the SimBench data and are
marked as extreme cases in their notes and configuration files, with a test
asserting that they are.

### 4.5 Metrics

`metrics.py` reduces a grid state to the quantities reward and KPIs need:
extreme voltages, summed excursion depth outside the K95 band, number of buses
outside each band, worst line and transformer loading, summed loading above
100 %, losses. These are instantaneous quantities and a *secondary* indicator,
kept for comparability with literature that uses a hard band. The criterion this
project is assessed against works on ten-minute means (§6.6).

---

## 5. L2 — Asset and flexibility models

All actuators implement a common protocol. This is what allows grid, actuator and
agent configuration to be varied independently.

```python
class FlexAsset(Protocol):
    asset_id: str
    bus: int
    ratings: AssetRatings  # S_max, P_max, fuse rating, contracted capacity

    def action_spec(self) -> ActionSpec: ...  # box in PHYSICAL units
    def obs_spec(self) -> ObsSpec: ...
    def initial_state(self, rng) -> AssetState: ...

    def to_setpoint(self, s, action, info) -> Setpoint:
        # action -> (p_mw, q_mvar). Affine and invertible; every limitation is
        # reported in Setpoint.clipping_info, never applied silently.
        ...

    def dynamics(self, s, sp, x, g) -> tuple[AssetState, AssetOutcome]:
        # Pure function: same input, same output, no side effects. Enables
        # hypothetical roll-outs.
        ...
```

Three commitments in this protocol that look like formalism but keep later
certification reachable (§14):

- **`dynamics` is a pure function**, `AssetState` a copyable value object. A
  predictive safety filter has to roll asset states forward over a horizon;
  with state-mutating methods that is impossible without rewriting the models.
- **Actions are physical power quantities**, not normalised fractions or target
  values. A state-of-charge target produces a non-box feasible set in action
  coordinates, whereas the certified feasible set is formulated in injections.
  Normalisation for the agent happens in the `ActionMapper` and is affine there.
- **`ratings` are maintained from the start.** Fuse rating, contracted capacity
  and `S_max` are the basis of the deterministic uncertainty sets (§7). Recording
  them now costs nothing; adding them later is manual work across all scenarios.

**PV.** Potential from the data; action is a curtailment factor and/or reactive
power within the inverter capability diagram (`S_max` circle, cosφ limits). Modes:
`p_only`, `q_only`, `pq`.

**BESS.** State-of-charge integration with separate charge and discharge
efficiencies, SoC limits, C-rate limits, optional self-discharge and a simple
degradation model as a cost term. Actions are projected onto the physically
feasible range, and the projection is reported so that a policy systematically
proposing infeasible actions is visible.

**Heat pump.** Not a load profile but a **shiftable load with state**: heat demand
from data, COP(T_ambient, T_flow) from when2heat characteristics, and a store.
Stage 1 (fixed for M4) is a buffer store as an energy reservoir with losses and an
admissible temperature band; stage 2 would be a 1R1C/2R2C building model with a
comfort band for room temperature; stage 3 SG-Ready levels as a discrete action
space. Plus minimum run and idle times and a maximum blocking duration. Comfort
violation is the integral of the band shortfall.

**EV charge point.** Session based: the scenario generates a sequence
`(t_arrival, t_departure, E_demand, P_max, soc_arrival)` per charge point. Charging
power is controllable between arrival and departure; energy not delivered by
departure is booked as a penalty. V2G as a later extension, initially disabled.

---

## 6. L3 — Gymnasium environment

```
env/
├── lv_grid_env.py   # LVGridEnv(gymnasium.Env)
├── obs.py           # ObservationBuilder
├── actions.py       # ActionMapper (flat / per asset / hierarchical)
├── reward.py        # RewardComposer and terms
├── episodes.py      # EpisodeSampler
├── splits.py        # week characterisation, stratified split, embargo
├── forecast.py      # forecast error model
└── wrappers.py      # normalisation, action masking, logging
```

### 6.1 One step

```
step(a):
  1. ActionMapper: a (agent space) -> setpoints per asset
  2. env.safety.transform(...)      -> possibly modified setpoints
  3. for each asset: to_setpoint()  -> write p/q
  4. for the k simulation steps inside one control_dt:
       set exogenous profiles at t (uncontrollable load, PV potential, T_ambient)
       PowerFlowEngine.run(net)     -> GridState
       PQAggregator.add_sample(vm_pu at connection points)
       metrics.violation_metrics(GridState)
       for each asset: dynamics()
       accumulate partial rewards
  5. RewardComposer: weighted sum, individual terms in info[]
  6. ObservationBuilder: observation for t+1
  7. determine terminated / truncated
```

**Information ordering.** Steps 1 to 3 may access only quantities known at the
decision point — the state at `t` and forecasts, never the realisation during
`[t, t+control_dt)`. This is enforced at runtime by a `DecisionScope` that raises
on any access to a later time step. The reason is not pedantry: implicit
clairvoyance is invisible during training but voids every later safety statement,
because a real controller does not have that information. Such a bug is spread
diffusely across the code and is practically impossible to remove afterwards.

### 6.2 Observation

The `ObservationBuilder` assembles the observation from declaratively configured
feature groups:

| Group | Contents |
|---|---|
| `time` | sin/cos of time of day and season, weekday flag |
| `measurements` | `vm_pu` at selected measurement buses, transformer loading, feeder currents — selected via `sensor_config` |
| `asset_state` | SoC per BESS, thermal state per heat pump, plugged-in/remaining energy/remaining time per EVSE |
| `local_power` | P/Q of the own asset and of the house connection |
| `pq_budget` | consumed share of the permitted ten-minute window budget, remaining duration of the weekly interval, partial mean of the open window — **mandatory**, see §6.6 |
| `forecast` | PV potential, load, ambient temperature and EV departures over the next H steps; error model per §6.7 |
| `history` | ring buffer of the last L steps (alternatively `RecurrentPPO`) |

`sensor_config` is an important research switch: `full_state` (all bus voltages —
unrealistic, but an upper bound) versus `realistic` (substation plus feeder-end
measurements plus local house connection values). Observation normalisation uses
fixed physical scales rather than learned statistics, which is required for the
multi-agent path and better for reproducibility anyway.

### 6.3 Action space

Three modes, all through the same `ActionMapper`:

1. **Flat centralised** — one `Box` vector over all actuators. Simplest entry
   point, scales poorly.
2. **Parameterised per asset type** — the agent emits a parameter vector per
   asset type, applied to all instances through a type-specific encoder
   (parameter sharing, invariant to the number of assets).
3. **Multi-agent** — one agent per actuator or per feeder; the environment
   additionally offers a PettingZoo-compatible interface.

**Commitment:** start with mode 1 (M3), switch to mode 2 from M4; mode 3 stays
open. So that the multi-agent case does not have to be rebuilt later, four rules
apply from M3:

1. The environment always holds an `AssetGraph` with a stable actuator ordering
   and an `agent_partition` mapping. The single agent is a `FlatView` over a
   partition containing all actuators — there is no second code path.
2. Observation, action and reward building blocks are computed **per actuator or
   per partition** and only aggregated at the end. No logic may assume a flat
   vector of fixed length.
3. No learned global observation normalisation (`VecNormalize` on observations),
   but fixed physical scales per feature.
4. A `PettingZooAdapter` exists from M3, even unused, together with a
   **regression test**: the single-agent environment and the multi-agent
   environment with one partition must produce bit-identical trajectories given
   identical actions. That test is the only reliable protection against the two
   paths drifting apart over months.

The **safety layer** is not a Gym wrapper but a component *inside* the
environment, because a wrapper only sees the observation whereas a certifier
needs the full system state. Two intervention points exist from M3, both with a
null implementation: `env.safety` (called before setpoints are written, receives
`SystemState`) and `info["action_mask"]` (always present; with `none` it permits
everything), so a masking policy outside the environment can use it without the
environment changing.

### 6.4 Reward

Terms fall into two categories — **`objective`** (costs to minimise) and
**`constraint`** (limits to respect). That separation is what lets fixed weights
and the Lagrangian variant run from the same configuration.

```yaml
reward:
  mode: fixed_weights           # fixed_weights | lagrangian
  objective:
    pv_curtailment:    {weight: -1.0,  unit: kWh}
    ev_unserved:       {weight: -5.0,  unit: kWh, at: departure}
    hp_comfort:        {weight: -2.0,  unit: Kh}
    bess_degradation:  {weight: -0.2,  unit: kWh_throughput}
    action_smoothness: {weight: -0.05}
    grid_losses:       {weight: -0.1,  unit: kWh}
  constraint:
    en50160_k95:       {weight: -10.0, limit: 0.05, shaping: potential}
    en50160_k100:      {weight: -50.0, limit: 0.0}
    thermal_overload:  {weight: -10.0, limit: 0.0, form: hinge, limit_pct: 100}
  normalization: fixed_scale
```

**Both constraint terms are active from M3.** The M2 survey showed that thermal
overload is the dominant binding constraint in the reference scenarios and remains
significant under `moderate_growth` (12.6 % of steps). A minimal environment with
only a voltage term would optimise against a criterion that barely binds.

**Mode `fixed_weights`** (starting configuration): all terms are scalarised.

**Mode `lagrangian`** (planned alternative): only `objective` terms form the
reward; each `constraint` term is additionally reported as a cost signal in
`info["cost/<name>"]`. A `LagrangianCallback` estimates the expected cost `J_c`
per episode and updates multipliers by dual ascent `λ ← [λ + η·(J_c − d)]₊`; a
wrapper forms `r_eff = r_obj − Σ λ_i·c_i`. The practical advantage here: the
limits `d` are stated in physical or normative units. For EN 50160, `d = 0.05` is
**exactly the standard's criterion** — no penalty weight needs guessing.

Two pitfalls worth knowing before implementing:

- A changing λ makes the reward **non-stationary**. With off-policy algorithms the
  replay buffer holds rewards computed with a stale λ. The clean route is to store
  costs separately and recompute `r_eff` on sampling; the pragmatic route is to
  update λ much more slowly than the policy learns. Validate the Lagrangian
  variant with PPO first, then transfer to SAC/TQC.
- The dual problem oscillates easily. λ needs an upper bound, `J_c` an
  EMA-smoothed estimator, and η should be well below the policy learning rate.

Every term is returned **separately in `info["reward/<term>"]`**. Without that
decomposition it is impossible to reconstruct later which term dominated
learning. Independently of the reward, the physical KPIs are recorded unweighted:
**the comparison against reference methods is made on KPIs, never on the reward**
— otherwise the policy whose objective one defined oneself wins trivially.

### 6.5 Episodes and data splits

The split unit is the **week**, because that is the assessment interval of
EN 50160 and because a standard-conforming pass rate needs complete calendar
weeks with a zero initial budget.

**Why not a chronological block split.** The obvious design — train on Jan–Sep,
validate on Oct, test on Nov–Dec — was the original plan and is wrong here.
Measured on `1-LV-rural1--2-sw` over the 51 complete weeks of 2016:

| | PV energy per week | heat pump energy per week |
|---|---|---|
| annual range | 301–12,065 kWh | 27–1,789 kWh |
| Nov–Dec (8 weeks) | 301–3,831 kWh | 602–1,324 kWh |
| share of the range covered | **30 %** | 41 % |

The spread is not the decisive part; the position is. The annual PV quantiles of
those eight test weeks are 0.29 / 0.12 / 0.16 / 0.18 / 0.06 / 0.14 / 0.10 / 0.00
— **every test week lies below the 30th percentile, and one is the weakest PV week
of the year.** Since the voltage band violations under `moderate_growth` are
PV-driven overvoltage, such a test set measures the headline KPI exactly where the
problem does not occur. The reported pass rate would be high and uninformative.

The usual justification for a chronological split is leakage avoidance plus
deployment fidelity: adjacent windows are correlated, and forecasting models
should be trained on the past and applied to the future. This is not a
forecasting task — the policy is a controller, and the question is whether it
works in operating situations it has not seen. With a single year of data, a
chronological split unavoidably confounds "unseen data" with "unseen season", so a
failure cannot be attributed to either.

**Chosen design: a stratified week split with an embargo.**

1. Each of the 51 weeks is characterised by a feature vector computed from
   **exogenous quantities only** — PV energy, heat demand, EV coincidence, peak
   net reverse flow. No controller, no power flow, hence no leakage through the
   label.
2. Two axes with three quantile bins each, giving nine strata. Three axes with 27
   strata would be too fine for 51 weeks.
3. Weeks are drawn from each stratum proportionally into train, validation and
   test.
4. **Embargo:** weeks immediately adjacent to a test week are dropped from
   training. Weather autocorrelation operates on the scale of days, so adjacent
   weeks are genuinely similar; this is the standard remedy and costs roughly ten
   training weeks, which is acceptable.

The split is drawn with a dedicated `split_seed`, the resulting week lists are
committed to the configuration and carried in the run manifest, so it stays
identical across runs, agents and baselines.

**Three sets, reported separately.**

- **Test set** (stratified, representative) — the primary KPI.
- **Stress weeks** (curated: highest PV infeed, coldest period with heat pump
  peak, weekend with high EV coincidence) — a separate, named set. Blending them
  into the test aggregate makes the number unreadable: a pass rate of 0.85 could
  mean "fails on the one extreme week" or "fails everywhere a little".
- **Held-out month** — one deliberately withheld contiguous month as a harder
  test of temporal generalisation. The reason: season enters the observation
  through the sin/cos features, so the agent can condition on it. Under an
  interleaved split it has seen seasonally similar weeks for every test week, and
  the generalisation claim is weaker than it sounds. Two numbers answering two
  different questions.

**From M5 the dilemma dissolves.** WPuQ spans 2018 to 2020, which permits a
genuine year holdout — train on 2018/19, test on 2020. That is the clean split, in
which stratification and leakage avoidance no longer conflict. SimBench provides
only one year, so the stratified split is a workaround here, if a justified one.

Implementation in `env/splits.py`: week characterisation, stratified assignment,
embargo, and serialisation of the resulting week lists.

- **Episode length and budget randomisation.** The criterion refers to a whole
  week, but week-long episodes are long and expensive. Chosen solution: episodes
  of 2–7 days with a **randomised initial budget** — on reset, the consumed window
  budget and the remaining duration of the current week are drawn at random. The
  agent then sees every budget regime without week-long episodes. The evaluation
  set, by contrast, contains **complete calendar weeks with a zero initial
  budget**, so the reported KPI is standard-conforming.
- **Discounting is not a free choice.** The effective horizon `1/(1−γ)` must cover
  the criterion horizon. At `control_dt = 15 min` a week is 672 decisions → `γ ≈
  0.997`; at 5 min it is 2016 → `γ ≈ 0.999`. Configuration validation derives a
  consistent `γ` and warns on deviation. A habitual `γ = 0.99` would simply not
  see the weekly horizon.
- **Curriculum** (optional) — start at moderate penetration, increase over
  training; a separate sampler implementation so it stays switchable.
- **Fixed evaluation set** — an immutable list of (week, scenario seed) pairs per
  set, against which *all* agents and *all* baselines are evaluated.
- `truncated = True` at the window end; `terminated` only on an irrecoverable
  state (power flow divergence).

### 6.6 Implementing the EN 50160 percentile criterion

This is the decision with the widest consequences, so it is collected here rather
than scattered across layers.

**The criterion.** Two nested conditions, both on **ten-minute mean values** of
the RMS voltage over a **weekly interval**:

- **K95** — 95 % of the ten-minute means of each week within ±10 % U_n
- **K100** — all ten-minute means within +10 % / −15 % U_n (low voltage)

With 1008 ten-minute windows per week, K95 permits **up to 50 violating windows**
per bus and week. Assessment is at the point of common coupling (§4.2).

**Consequence 1 — step size.** 15 does not divide 10, hence `sim_dt ∈ {1, 2, 5,
10}` (§3.3). `control_dt` remains a multiple of `sim_dt` and may be offset against
the assessment grid — a 15-minute control cycle on 5-minute physics is valid and
realistic — but the window alignment is logged so artefacts stay visible.

**Consequence 2 — the reward is no longer definable per step.** Whether an
excursion "counts" depends on the distribution of the whole week. The problem is
not Markovian without extra state, and the cost signal is terminal. Three building
blocks resolve this:

1. **`PQAggregator`** forms the ten-minute means. Criterion-relevant events occur
   only at window boundaries. It holds the running partial sum, not the samples —
   over a year at one-minute resolution that is the difference between a few
   hundred bytes and several hundred megabytes.
2. **Budget state in the observation.** Consumed windows, remaining duration of
   the week and the running partial mean enter as the `pq_budget` feature group.
   Without them an optimal policy cannot exist, because the agent cannot tell
   whether an excursion still fits the budget or breaks it. With many buses this
   is condensed: maximum budget utilisation, number of buses above 80 %, number
   already violating.
3. **Potential-based shaping.** For credit assignment over ~672–2016 steps, the
   terminal cost is supplemented by a potential `Φ(consumed budget, excursion
   depth, remaining time)`: `r_shaped = r + γ·Φ(s′) − Φ(s)`. Potential-based
   shaping does not change the optimal policy but provides a dense signal.
   Additionally a small directly weighted depth term over `|ΔU|` with a deadband
   as a starting aid, switchable off in an ablation — strictly speaking it is
   foreign to the criterion and may make the agent more conservative than the
   standard requires.

K100 is treated as a hard constraint with a high per-event weight.

**Consequence 3 — episodes and γ:** see §6.5.

**Consequence 4 — the reference methods change mathematical form.** B7–B9 no
longer optimise against a box restriction but against a **cardinality
constraint** ("at most 50 windows outside per bus and week"). That is mixed
integer: one binary per window and bus plus a cardinality restriction. Three
workable routes: (a) MILP with big-M on the linearised grid model — solvable for a
weekly horizon and a few dozen assessed buses, but not cheap; (b) a CVaR
relaxation as a convex approximation — conservative, fast; (c) sequential
loosening of the box restriction until the budget is exhausted. Recommendation:
(b) for the first pass, (a) for the final evaluation of `mpc_oracle`. What matters
is that the reference methods optimise against the **same** criterion they are
then assessed against.

**Consequence 5 — baselines must be tuned too.** The deadbands and slopes of
B3/B4 and the thresholds of B5/B6 are implicitly designed for a hard ±10 %
criterion. Under K95 the optimal parameterisation is different, because tolerance
is permitted. Every baseline therefore gets its own documented parameter search
against the same target KPI on the validation period. Without that, the RL agent
wins against a deliberately badly tuned reference — the most common methodological
flaw in the RL energy literature, and the one that makes results attackable
fastest.

**A note on averaging.** An instantaneous value of 1.14 pu paired with 1.04 pu
averages to 1.09 and does not count as a violation. A controller assessed against
instantaneous limits is strictly more conservative than the standard requires.
This is also why `undervoltage_stress` reaches 0.9275 pu instantaneously and still
passes: the ten-minute mean absorbs it.

### 6.7 Forecast error model

| Mode | Description |
|---|---|
| `perfect` | exact future values — a declared special case and upper bound |
| `persistence` | naive continuation, for PV on the clear-sky index |
| `synthetic` | realisation-consistent error model (**default**) |
| `model` | a real forecasting model — later expansion stage |

The central property of `synthetic`: the forecast is **generated from the
realisation** (`forecast = realisation ⊕ error`), not drawn independently.
Otherwise forecast and realisation are statistically inconsistent and the policy
can partially invert the error — a leak that looks like skill during training and
vanishes in evaluation.

Error structure: **PV** multiplicative on the clear-sky index, AR(1)-correlated
over the horizon, heteroscedastic; **load** additive with a diurnal σ(h);
**heat demand** through the ambient temperature forecast propagated by the thermal
model; **EV** the dominant uncertainty — departure time and energy demand are
distributions, not point values.

`forecast_seed` is separate from the training and scenario seeds, so the same
episode reproducibly gets the same forecast. The σ(h) parameters are calibrated
against literature values and live in the configuration, not in the code.

---

## 7. Safety: heuristic mechanisms and the certified target architecture

### 7.1 Heuristic mechanisms as comparison arms

The mechanisms in this section **reduce** violations but guarantee nothing. They
remain as comparison arms, because the scientifically interesting question is:
*what does the guarantee cost compared to the heuristic?*

Masking, projection and replacement attack at three different points and impose
different requirements, so the layer is decomposed into three orthogonal axes.

**Axis 1 — `FeasibilityModel`: where does the admissibility information come
from?** This is the crux and is often glossed over: **whether an action violates
the voltage band is only known after the power flow.** Anything intervening
*before* execution needs a predictor.

| Implementation | Description | Cost |
|---|---|---|
| `sensitivity` | linearised ∂U/∂P, ∂U/∂Q, ∂I/∂P at the operating point | very cheap, accurate locally, poor for large actions |
| `learned` | neural network on (state, action) → limit violation | cheap, accuracy unclear |
| `exact` | true AC power flow per candidate action | expensive; **not a certifier** — it evaluates *one* point and says nothing about the uncertainty set for the coming interval |

**Axis 2 — `SafetyMechanism`: how is the intervention made?**

| Mode | Intervention point | Requirements |
|---|---|---|
| `none` | – | reference arm |
| `mask` | **before** sampling, on the policy distribution | **discrete** action space; mask known before the power flow (`MaskablePPO`) |
| `project` | **after** sampling, continuous | convex (linearised) feasible set; QP `min ‖a′ − a‖²` |
| `replace` | after sampling, binary | a trusted fallback controller (B3/B4 droop) |
| `safety_layer` | after sampling, learned | pre-trained linear cost model (Dalal et al.) |

**Axis 3 — `LearningCoupling`: what does the algorithm see?** In practice more
consequential than the choice of mechanism, and frequently decided implicitly:

| Mode | What lands in the buffer | Effect |
|---|---|---|
| `store_proposed` | the **proposed** action `a` | policy does not learn it was corrected; gradients biased against the executed dynamics |
| `store_executed` | the **executed** action `a′` | consistent for off-policy Q-learning, but changes the behaviour density (violates PPO's importance sampling assumption) |
| `differentiable` | `a′`, gradients flow through the projection | cleanest in theory, needs a differentiable QP layer and much more compute |
| `penalty_only` | `a` plus a penalty for the intervention | mechanism acts at runtime only, not in the learning signal |

**What must be measured for the comparison to mean anything.** The violation rate
alone conflates the quality of the mechanism with the quality of the feasibility
model and with conservatism. `eval/safety_kpis.py` records: intervention rate;
intervention magnitude `‖a′ − a‖` as a distribution; **residual violation rate**
after the exact power flow; **false positive rate** (interventions the exact power
flow shows were unnecessary — overconservatism); **false negative rate** (no
intervention but a violation — model error); conservatism cost attributable to
interventions; decision time in ms; and violations **during** training, which is
the actual purpose of safe RL.

The two error rates require a **shadow power flow**: the unmodified original
action is additionally computed exactly. That doubles the cost and is therefore
enabled on the evaluation set and on training samples only.

**Three limitations to accept up front.** Masking exists in SB3 only through
`MaskablePPO`, so a comparison across all mechanisms must fix the algorithm to
PPO, and it needs a discretisation of curtailment and charging power. To keep
discretisation from becoming a second confounder, the environment carries an
`action.discretization` switch and the masking arm is compared against equally
discretised projection and replacement arms. Mask computation scales with the
action space size, which rules out `exact` for masking. And most importantly:
**under EN 50160 "safe" is not defined per step** — K95 is a weekly budget that an
action-level mechanism cannot enforce directly, so it must work against a stricter
surrogate and is structurally more conservative than the standard requires.

That last point yields a clean division of labour, which may itself be a result of
the work:

> **K100 through the safety mechanism (a genuine hard limit), K95 through the
> Lagrangian formulation (a budget criterion).** Each of the two normative
> criteria is addressed by the method whose guarantee form matches it — an
> instantaneous guarantee against an instantaneous criterion, an expectation
> guarantee against a percentile criterion.

### 7.2 Certified safety (target architecture)

The requirement is a dependable guarantee, not a safety heuristic. That is
achievable — but only as a **conditional** guarantee whose preconditions are fully
disclosed. In critical infrastructure the value lies precisely in that
explicitness.

**Guaranteed (target statement):** for every state inside the certified operating
envelope and for **every** realisation of the uncontrollable injections within the
specified uncertainty set, the executed action leads to a solution of the AC power
flow equations that respects all voltage and equipment limits (K100, thermal) —
and at every future point in time an admissible continuation exists.

**Not guaranteed**, and this belongs in the results report: validity against
reality (the guarantee is model-based); customer comfort (the shield may curtail
PV completely and block heat pumps and charge points — grid integrity is hard
constrained, comfort is a soft objective, and that hierarchy is a decision);
safety under faults and topology changes (the guarantee holds per topology).

**Four preconditions**

| | Precondition | How it is established | If violated |
|---|---|---|---|
| **P1** | the admissibility statement is *sufficient*, not approximate | convex restriction instead of linearisation | no guarantee, only heuristic |
| **P2** | the uncertainty set is certified | physical bounds or conformal prediction | the guarantee holds only for cases declared "typical" afterwards |
| **P3** | recursive feasibility | predictive safety filter with a terminal fallback set | the agent can manoeuvre into states with no admissible action |
| **P4** | the fallback action is itself admissible | **verified offline per grid × scenario** | **no controller can guarantee anything — the grid is inadequate** |

**P4 has been verified and holds.** With PV fully curtailed and the flexible loads
at minimum, no examined step violates a limit in any scenario of
`1-LV-rural1--2-sw`; the worst loading under the fallback is 85.7 %. A hard
guarantee is therefore reachable in this grid, and grid reinforcement is not the
obstacle. Note that this only became true once the *full* fallback definition was
used: curtailing PV alone leaves the transformer overloaded, and storage must be
driven from its profile rather than left at nominal values.

**Building block 1 — certified AC admissibility.** The right mathematical object
is the **convex restriction** of the power flow feasibility set: it identifies a
convex subset of injections in which the existence of a power flow solution is
guaranteed and that solution satisfies the operational constraints. The contrast
with the common relaxations matters: a relaxation is an *outer* approximation and
worthless for guarantees; only an *inner* approximation carries a sufficient
condition. The **robust** convex restriction extends this to uncertainty in the
injections, respecting the operating limits for all realisations within a
specified set. This turns `project` from a heuristic into a provable method: the
projection targets the robust convex restriction, which is a convex (SOCP/QCQP)
problem solvable per control step.

*Fallback if convex restriction proves too expensive at this grid size:*
linearisation **plus a rigorous remainder bound** (interval arithmetic on the
Taylor remainder, with outward rounding). Less sharp, but still sufficient. Plain
linearisation without a remainder bound is not.

Literature: Lee, Nguyen, Dvijotham, Turitsyn, *Convex Restriction of Power Flow
Feasibility Sets*, IEEE Trans. Control of Network Systems 6(3), 2019
(arXiv:1803.00818) · Lee, Turitsyn, Molzahn, Roald, *Robust AC Optimal Power Flow
with Robust Convex Restriction*.

**Building block 2 — certified uncertainty sets.** The guarantee is only as good
as the set it quantifies over. Two routes, giving different statements, and both
should be reported: **deterministic/physical** (house connection fuse, contracted
capacity, inverter `S_max`, maximum heat pump and charge point power) yields a
guarantee with no residual probability but is very conservative; **data-driven
with a finite-sample guarantee** (conformal prediction or the scenario approach on
the series from §3.2) is sharper but probabilistic, and δ must be stated. The
difference between the two — how much curtailment the worst-case assumption costs
— is one of the more interesting numbers of the work.

**Building block 3 — recursive feasibility.** Single-step safety is not enough.
Every actuator except PV has state: charging now can fill a battery so that no
downward flexibility remains later; a blocked EV must charge before departure. The
tool is the **predictive safety filter**: it checks in real time whether a proposed
learning-based input is safe by searching for a safe backup trajectory towards a
known safe set, and modifying the input when necessary establishes safety for all
future times. The structure fits: any RL algorithm that would have controlled the
original system can instead be applied to the safe system, giving a certifiably
safe RL application, and the filter intervenes minimally — only when safety cannot
be guaranteed.

Concretely: the **terminal fallback set** is the set of states from which the
fallback rule (PV fully curtailed, flexible loads at minimum, BESS free to support
the grid) stays admissible for all uncertainty realisations — its non-emptiness is
P4. The **backup trajectory** spans a horizon of typically 2–8 h, long enough to
discharge a full battery and complete an EV session, computed on the robust convex
restriction model. The agent's action is certified if such a trajectory exists;
otherwise the first step of the backup trajectory is executed.

Literature: Wabersich & Zeilinger, *Linear Model Predictive Safety Certification
for Learning-Based Control*, CDC 2018 (arXiv:1803.08552) · Wabersich & Zeilinger,
*A predictive safety filter for learning-based control of constrained nonlinear
dynamical systems*, Automatica 129:109597, 2021 · Wabersich et al., *Data-Driven
Safety Filters*, IEEE Control Systems Magazine, 2023.

**Building block 4 — verification of the implementation.** The guarantee lives in
the code, not in the paper. The shield is a separate module with its own test
suite; it shares no state with the agent, takes no policy-supplied parameters, and
is active during training **and** evaluation — otherwise a train/deploy mismatch
voids the guarantee in operation. A **falsification search** runs as a CI job:
adversarial search over (state, action, uncertainty realisation) for a case
classified as certified-safe that violates a limit in the exact AC power flow.
Every hit is a **defect of the certifier**, not a tolerable statistic.
`residual_violation_rate` therefore becomes a test assertion with target exactly
zero. Interval and remainder computations need outward rounding: a guarantee that
flips at the last digit is not one.

**Architectural consequences.** `SafetyCertifier` answers two-valued and
asymmetrically — `CERTIFIED_SAFE` or `NOT_CERTIFIED`, never "unsafe".
`NOT_CERTIFIED` means "not provably admissible", which suffices to trigger the
fallback; a certifier may be conservative but never optimistic. The shield's
limits are set at **K100 (+10 %/−15 %), not ±10 %**: enforcing ±10 % hard would be
needlessly conservative and would leave the K95 budget trivially unused. Learning
coupling is fixed to `store_executed`, so the agent learns on the *safe system* as
the predictive safety filter intends. Finally, **compute time is a risk, not just
a KPI**: if certification per step costs as much as an MPC step, the argument "RL
is cheaper at runtime than MPC" loses its substance, so `certification_time_ms`
belongs in the headline results.

**Two limitations touching the modelling frame.** The available convex restriction
formulations assume a balanced three-phase or single-phase equivalent. Real
low-voltage grids are unbalanced and the data supports that (the HTW profiles are
phase-resolved). **Commitment:** the certified arm works on the balanced model;
unbalance is a separate sensitivity study *without* a guarantee, and this must be
disclosed because unbalance is relevant for voltage problems in LV grids. Second,
the certifier needs the system state. With `sensor_config: realistic` the state is
only partially observed, so a hard guarantee requires a certified error bound on
the state estimate, entering the uncertainty set. The shield **may** receive more
information than the agent — it is a separate, better instrumented, trusted
component — but the measurement infrastructure that implies is a statement about
realisability and must be quantified.

---

## 8. L4 — Agents and reference methods

### 8.1 Shared controller interface

```python
class Controller(Protocol):
    def reset(self, info: InformationSet) -> None: ...
    def act(self, info: InformationSet) -> np.ndarray: ...
```

RL policies, rule-based methods, OPF and MPC implement the same interface, and
`eval/runner.py` knows only `Controller`. Methods that need more than the agent's
information set — OPF needs the grid model, MPC the forecast — receive that
through `info["privileged"]`, and the privileged access is explicitly flagged in
the results report.

### 8.2 RL agents

`agents/factory.py` builds an SB3 model from a YAML specification:

```yaml
agent:
  algo: sac                 # ppo | a2c | sac | td3 | ddpg | dqn
                            # sb3-contrib: tqc | qrdqn | recurrent_ppo | trpo | crossq
  policy: MultiInputPolicy
  policy_kwargs:
    net_arch: [256, 256]
    features_extractor: asset_encoder   # mlp | asset_encoder | gnn | attention
  hyperparams: {learning_rate: 3.0e-4, batch_size: 256, ...}
```

Feature extractors: `MlpExtractor` as the baseline; `AssetSetEncoder` with
type-specific encoders and pooling over assets of the same type, which allows
transfer to grids with a different asset count; `GraphExtractor` doing message
passing on the grid topology, which fits because voltage coupling is local along
feeders; `AttentionExtractor` over asset tokens.

Combined with the three action space modes and the choice of MLP/recurrent/graph,
comparing agent architectures is one line of configuration. Hyperparameter search
runs through Optuna with a TPE sampler and median pruner, search spaces declared
per algorithm.

### 8.3 Reference methods

These matter as much as the agents, because they define the scale on which RL
results are read.

| # | Method | Role |
|---|---|---|
| B0 | `do_nothing` | lower bound, uncontrolled grid |
| B1 | `random` | sanity check against degenerate policies |
| B2 | `fixed_cap` | rigid infeed limit (70 % / 60 % of rated power) |
| B3 | `cosphi_p` / `q_u_droop` | local reactive power control per VDE-AR-N 4105 |
| B4 | `p_u_droop` | voltage-dependent active power curtailment |
| B5 | `en14a_dimming` | power reduction of controllable devices (§14a EnWG, 4.2 kW at the connection point) |
| B6 | `greedy_local` | PV surplus charging for BESS/EV, "as late as necessary" charging |
| B7 | `opf_myopic` | centralised AC OPF per time step, no foresight |
| B8 | `mpc_forecast` | multi-step optimisation with a realistic forecast |
| B9 | `mpc_oracle` | multi-step optimisation with perfect foresight — **upper bound** |

B7–B9 optimise on a reduced, linearised grid model (LinDistFlow or pandapower
sensitivities) and are recomputed in the full AC model afterwards, so the physical
assessment is identical for all methods. B9 is the most important reference: "RL
reaches X % of oracle performance at Y % of the compute time" is a defensible
statement; "RL beats doing nothing" is not. On the mathematical form these take
under the percentile criterion, and on the necessary re-tuning of B3–B6, see §6.6,
consequences 4 and 5.

---

## 9. L5 — Storage, evaluation, visualisation

### 9.1 Run layout

```
results/<experiment>/<run_id>/
├── config_resolved.yaml     # fully resolved Hydra config
├── manifest.json            # code_commit, data_manifest_hash, seeds, environment
├── env_spec.json            # observation/action space definition
├── checkpoints/             # model_<steps>.zip, vecnormalize.pkl
├── tb/                      # TensorBoard events
├── metrics/
│   ├── train_scalars.parquet
│   └── episode_summary.parquet
├── traces/                  # per-step traces of evaluation episodes
└── eval/                    # kpi_table.parquet, report.html
```

Two granularities are deliberately separated: **scalars** are always written
(small); **traces** only for evaluation episodes and sampled training episodes,
otherwise the data volume explodes. `run_id = sha256(code_commit ‖ config_hash ‖
data_manifest_hash ‖ seed)[:12]`.

### 9.2 Reproducibility — concrete measures

- **Seeds separated by purpose:** `scenario_seed`, `episode_seed`, `train_seed`,
  `eval_seed`, `forecast_seed`, derived deterministically via `SeedSequence.spawn`
  rather than `base + 1`, which can produce correlated streams. Worker seeds are
  derived from the purpose seed and the worker index. Separating scenario from
  training seed is essential: otherwise "five seeds" accidentally means "five
  different grids".
- `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `OMP_NUM_THREADS=1`. For small LV grids, CPU training is often faster *and*
  easier to make deterministic than GPU.
- **Environment frozen:** `uv.lock` committed, CI runs `uv sync --locked`, so a
  dependency change without an updated lock file cannot slip through. pandapower
  and simbench versions matter — profile mappings and solver defaults differ
  between versions, and the ZIP defect in §4.1 is exactly such a case.
- **Data frozen:** SHA-256 per prepared frame, versioned preparation pipeline,
  cache key including the policy parameters and a `PIPELINE_VERSION`. Without the
  latter, a corrected resampling rule would silently have no effect on cached
  results.
- **Reproduction test as a CI job:** a short run executed twice with the same seed
  must produce bit-identical training scalars. This finds reproducibility leaks
  more reliably than any documentation.

### 9.3 KPI engine

**Grid quality — headline KPI, standard-conforming**

- **`en50160_pass_rate`** — share of (bus, week) pairs satisfying K95 *and* K100.
- **`p95_vm_pu`** per bus and week, plus the worst-bus value — a *continuous*
  quantity, indispensable because the pass rate is binary and improvements below
  the threshold would otherwise be invisible.
- **`budget_utilisation`** — consumed share of the 50 permitted windows. 0.98 and
  1.02 differ like night and day in the pass rate but are operationally almost
  identical; reporting both prevents false conclusions.
- Secondary, for comparability with literature using a hard band: ±10 % violations
  per time step, `max|ΔU|`, `∫|ΔU| dt`.
- Overload duration and integral for lines and transformer in MVA·h.

**Cost of control** — curtailed PV energy (kWh and % of potential), undelivered EV
energy at departure, heat pump comfort violation in K·h and maximum blocking
duration, BESS throughput and equivalent cycles, grid losses, switching actions.

**Learning behaviour** — learning curves per reward term, sample efficiency,
stability across seeds, compute time per decision (relevant against MPC/OPF).

**Safety mechanisms** (when active) — intervention rate and magnitude, residual
violation rate, false positive and false negative rates against the shadow power
flow, conservatism cost, violations during training, `certification_time_ms`, and
for the certified arm the size of the certified operating envelope and the
falsification search hit count (target zero).

**Reporting by set.** Every KPI is reported per evaluation set — test, stress
weeks, held-out month — and never aggregated across them. A single blended number
cannot be read: a pass rate of 0.85 could mean failure on one extreme week or a
little failure everywhere (§6.5).

**Statistical evaluation.** At least 5, better 10 seeds per configuration.
Aggregation via **IQM with bootstrap confidence intervals** and performance
profiles (following `rliable`) rather than mean ± standard deviation: RL
distributions across seeds are regularly skewed and multimodal, where the mean
misleads. Additionally paired comparisons against baselines on the fixed
evaluation set.

### 9.4 Visualisation

Static plots for publication: learning curves with seed confidence bands; box
plots of KPIs for RL variants against B0–B9; a voltage heat map (bus × time) per
evaluation episode with violations marked; annual duration curves; **a Pareto plot
of curtailment energy against violation energy** — the core figure for "what does
grid security cost"; a grid map coloured by violation frequency; and action
profiles over the day, which are essential for interpretability.

An interactive dashboard (Streamlit + Plotly) for episode replay with a time
slider showing voltages, loadings, actions and reward terms in sync, plus run
comparison from the registry. Reports are regenerated from the Parquet files
through `scripts/report.py`, never assembled by hand from notebooks.

---

## 10. L6 — Experiment orchestration

```
scripts/prepare_data.py    # raw data -> validated, resampled cache + manifest
scripts/survey_grids.py    # survey of candidate grids
scripts/run_baseline.py    # uncontrolled annual run, EN 50160, P4 check
scripts/train.py           # Hydra entry point: one run
scripts/sweep.py           # Optuna / multirun over seeds and agent variants
scripts/evaluate.py        # controller (RL checkpoint or baseline) on the eval set
scripts/report.py          # study -> KPI tables, plots, HTML report
```

A study is a YAML file spanning the cross product:

```yaml
grid:      [simbench_1-LV-rural1-future]
scenario:  [moderate_growth, voltage_stress]
agent:     [ppo_mlp, sac_mlp, tqc_mlp, recurrent_ppo, sac_gnn]
seed:      [1, 2, 3, 4, 5]
baseline:  [do_nothing, q_u_droop, en14a_dimming, opf_myopic, mpc_oracle]
```

Model selection (`best_model`) happens on the **validation weeks**, reported
figures on the **test weeks**; the stress weeks and the held-out month are
reported separately and never used for selection (§6.5). Otherwise the result is
optimistically biased.

---

## 11. Related work worth studying

- **OPF-Gym** (Digitalized Energy Systems, Oldenburg) — a Gymnasium-compatible
  framework for OPF problems using pandapower and SimBench grids with their time
  series, including voltage control with reactive power. The closest existing work
  to this project; worth a close look.
- **PowerGridworld** (NREL) — lightweight modular framework for multi-agent Gym
  environments in energy systems, with a pandapower power flow model. A good
  reference for the L2/L3 component structure.
- **Gym-ANM** — RL environments for active network management in distribution
  grids.
- **CommonPower** — safe RL in a smart grid context with a Pyomo-based model and a
  forecast interface; relevant for the safety layer and B8/B9.
- **CityLearn** — demand response and building energy; a good model for benchmark
  discipline and evaluation protocol.

---

## 12. Roadmap

M0 to M2 are complete. The remaining cut differs from the original plan in three
ways, all of them consequences of what M1 and M2 measured: the minimal environment
is a well-posed control problem rather than a teaching step, compute time is a
real constraint that shapes the training configuration, and the certified shield
is known to be reachable because P4 holds.

| Milestone | Contents | Acceptance criteria | Invariants due |
|---|---|---|---|
| **M0** ✅ | skeleton, core schemas, reproducibility chain, CI, extensibility contract | manifest written, contract tests in place | I3, I7 (null) |
| **M1** ✅ | data layer: SimBench adapter, UTC time base, resampling, Parquet cache with content hash, reactive power | energy-preserving resampling verified on the real year; cache key sensitive to policies | I6 |
| **M2** ✅ | grid layer: loader with ZIP repair, power flow engine with hypothetical call, EN 50160 assessment, scenarios, P4 check | uncontrolled annual run per scenario documented; P4 verified; runtime measured | I5 |
| **M3** | minimal environment: PV curtailment only, flat action space, `pq_budget` features, both constraint terms, `PettingZooAdapter` + equivalence test, baselines B0–B4 with their own parameter search, **stratified week split** (`env/splits.py`) | `check_env` passes; PPO beats B0 and the tuned droop baselines on `en50160_pass_rate` **and** overload integral under `moderate_growth`; single- and multi-agent paths bit-identical; test set covers all nine strata and its PV quantiles span the year | I1, I3 (env), I4, I7 (env) |
| **M4** | full actuator set: BESS, heat pump (buffer store), EVSE with session model; action mode 2 (per asset type); EV session generation from emobpy, calibrated against ElaadNL | all actuators active in one episode; clipping and comfort violations counted correctly; pure-function dynamics verified per asset type | I2 |
| **M5** | data extension and forecasts: WPuQ and HTW for 1-minute validation, forecast error model with `perfect` as a special case | identical policy evaluated at `sim_dt = 10` and `sim_dt = 1`; difference reported; forecast leak test passes | — |
| **M6** | evaluation chain and reference methods B5–B9, KPI engine, statistical aggregation, reports | KPI table RL vs. all baselines including `mpc_oracle`; reproduction test green in CI | — |
| **M7** | agent architecture comparison: agent factory, feature extractors, Optuna sweeps; Lagrangian reward variant | study over ≥ 4 architectures × 5 seeds with IQM confidence intervals; fixed weights vs. Lagrangian compared on KPIs | — |
| **M8** | safe RL, heuristic: `FeasibilityModel` (sensitivity), mechanisms `project`/`replace`/`mask`, shadow power flow, safety KPIs | mechanism comparison at fixed PPO including false positive/negative rates | — |
| **M9** | certified shield (own sub-project): robust convex restriction, certified uncertainty sets, predictive safety filter, falsification CI | `residual_violation_rate == 0` as a build condition; certified operating envelope documented; cost of the guarantee against M8 quantified | — |
| **M10** | generalisation and multi-agent: transfer across grids and scenarios, action mode 3 | training on grid A, test on grid B; multi-agent arm evaluated | — |

**Notes on the cut**

*M3 is not a toy.* Under `moderate_growth` the fallback action brings loading from
280 % to 55.6 % and voltage to 1.051 pu, so PV curtailment alone resolves both the
voltage and the thermal problem. M3 is therefore a complete control problem with a
genuine cost trade-off, and B2/B3/B4 are real competitors rather than straw men.

*Training step size is a design decision.* At 23 ms per power flow step,
10⁶ training steps at `sim_dt = 5 min` cost about 6.4 hours of power flow per
worker. Recommendation: `sim_dt = 10 min` for training — the coarsest step size
that still represents EN 50160 exactly, one power flow per assessment window,
halving the cost — with final evaluation at 1 or 5 minutes using the same policy.
With eight parallel workers a run lands under an hour, which makes sweeps over
five seeds affordable. Verifying that the 10-minute policy holds up at one minute
is an M5 deliverable, not an afterthought.

*The evaluation split moves into M3.* The fixed evaluation set is needed as soon
as the first policy is compared against a baseline, so the week characterisation
and the stratified split are an M3 deliverable rather than part of the later
evaluation chain. Committing the week lists early also means every milestone from
M3 onwards reports on the same weeks, which is what makes the numbers comparable
across the project.

*M5 before M6.* Forecasts and high-resolution data come before the evaluation
chain, because the reference methods B8/B9 need a forecast interface and because
the 1-minute validation changes what the KPIs mean.

*M9 is a sub-project, not a milestone beside others.* Robust convex restriction
plus predictive safety filter plus falsification infrastructure is substantial
work. The cut is deliberately such that M3–M8 produce usable results without a
finished shield, and M9 is additive.

---

## 13. Decision record

| # | Question | Decision | Where |
|---|---|---|---|
| D1 | one agent or several? | **single agent first**, multi-agent kept open | §1.2 (principle 7), §6.3 |
| D2 | voltage criterion | **EN 50160 percentile criterion** (K95 on ten-minute means per week, K100 with −15 %) | §6.6 |
| D3 | heat pump model | **stage 1: buffer store** for M4; `ThermalModel` protocol admits 1R1C later | §5 |
| D4 | reactive power | Q implemented as an action option, main study P-dominated; Q required for B3 regardless | §5 |
| D5 | reward weighting | **fixed weights first**, constrained RL with Lagrange multipliers as a full alternative | §6.4 |
| D6 | forecasts | **error model with `perfect` as a special case** | §6.7 |
| D7 | connection points | buses with a load **or** generator/storage element; exclusion list for equivalent infeeds; "all buses" as a sensitivity variant | §4.2 |
| D8 | safety layer | not a switch but **three orthogonal axes** (feasibility model × mechanism × learning coupling) plus its own KPIs | §7.1 |
| D9 | guarantee level | **hard, conditional guarantee** via robust convex restriction and a predictive safety filter; the agent is untrusted | §7.2 |
| D10 | timing of certification | **later, as its own sub-project (M9)**; extensibility secured by **seven tested invariants** | §14 |
| D11 | working grid and scenario | `1-LV-rural1--2-sw` with `moderate_growth`; extreme cases as declared test scenarios | §4.4 |
| D12 | language | English throughout, including error messages | `CONTRIBUTING.md` |
| D13 | evaluation split | **stratified week split with an embargo** instead of a chronological block; stress weeks and a held-out month reported separately | §6.5 |

### 13.1 What D2 actually costs

The move from a hard band to the percentile criterion is normatively correct and
methodologically harder than it looks. Four points that cost time:

1. **`sim_dt = 15 min` drops out**, which is the natural resolution of the SimBench
   series. They must be upsampled, which adds no information. The dependency
   between the voltage criterion and the choice of data source was not visible
   before.
2. **The reward is terminal and not Markovian** without the budget state. An agent
   that cannot see how much tolerance is left is solving a problem with hidden
   state.
3. **γ is bound to the weekly horizon.** At a 15-minute control cycle, `γ = 0.99`
   is an effective horizon of about 25 hours — the criterion would be invisible.
4. **The reference methods become mixed integer.** This affects `mpc_oracle`, the
   most important comparison value.

### 13.2 Open questions

1. **Is the budget per bus or grid-wide?** Standard-conforming is per connection
   point, which makes the KPI strict (the worst bus decides) and the budget state
   high-dimensional. Current choice: KPI per bus, observation condensed (§6.6).
2. **K100 is asymmetric.** For PV-dominated overvoltage the lower limit rarely
   binds; for heat pump and EV-dominated undervoltage it does. The measurements in
   §4.4 show that `undervoltage_stress` still passes because the ten-minute mean
   absorbs the excursion — reaching the lower criterion needs a harder scenario.
3. **How is the Lagrangian variant evaluated?** It optimises a different objective
   than the fixed-weight variant, so the comparison runs on physical KPIs and the
   curtailment/pass-rate Pareto plot only.
4. **Confounding in the safe-RL comparison.** The masking arm forces discrete
   actions and PPO (§7.1). Open whether the study adopts the discretisation for
   *all* arms (cleaner comparison, different problem) or runs two comparison
   levels. Proposal: two levels.
5. **Calibration of the forecast errors** — defensible σ(h) values for PV and load
   at individual asset and house connection level, not portfolio level where
   balancing makes the error much smaller. A small research step before M5.
6. **Unbalance.** The certified arm works on the balanced model (§7.2). Whether
   unbalanced certification has to become part of the project is a scope question
   with a substantial answer either way.
7. **Measurement infrastructure.** A hard guarantee under realistic sensing needs
   a certified error bound on the state estimate. Which infrastructure may be
   assumed, and is that assumption practically defensible?
8. **Stratification axes and embargo width.** Two axes with three bins is a
   compromise forced by having only 51 weeks; which two axes matter most is an
   empirical question, and the answer may differ between an overvoltage-dominated
   and an undervoltage-dominated scenario. The one-week embargo is likewise a
   default rather than a measured value. Both should be revisited once a genuine
   multi-year holdout is available (M5), at which point the stratification becomes
   a convenience rather than a necessity.

---

## 14. Extensibility contract

Decision D10 defers certification to M9 but requires the architecture to be able
to take it later. That promise is only credible if it is checked: **promises of
modularity decay silently.** Six months without a test and some reasonable
shortcut has used up the extensibility without anyone noticing.

The following seven invariants each have a test in `tests/test_invariants.py`
running in CI from M0. Open ones are marked `xfail(strict=True)`, so when the
component appears and the test turns green unexpectedly, **the build breaks** and
forces the marker to be removed. The number of XFAIL reports is the contract's
debt count, readable in every CI run. The last column is the actual argument: all
seven are cheap now and expensive to impossible later.

| | Invariant | Why M9 needs it | Status | Cost of retrofitting |
|---|---|---|---|---|
| **I1** | `SystemState` complete and separate from `Observation` | the certifier needs the full state, the agent may see less | due M3 | high — touches every environment component |
| **I2** | asset dynamics as **pure functions** over a copyable `AssetState` | the predictive safety filter rolls states forward hypothetically | due M4 | **very high** — rewriting every asset model |
| **I3** | strict information ordering: action construction uses only `InformationSet(t)` | a guarantee based on information the real controller lacks is not one | **satisfied** (mechanism M0, environment M3) | **very high** — the bug is diffuse and hard to find |
| **I4** | actions in physical units, normalisation affine and invertible, clipping reported | the certified feasible set is formulated in injections; projection needs box coordinates | due M3 | medium-high — changes the action space and all results |
| **I5** | `PowerFlowEngine.run_hypothetical` without side effects | backup trajectories and falsification search need it | **satisfied** (M2) | medium |
| **I6** | exogenous inputs carry `bounds`; `ratings` maintained per asset | uncertainty sets need bounds, not point values | **satisfied** (M1) | medium; ratings are manual work across scenarios |
| **I7** | two safety intervention points with null implementations | the shield sits inside the environment, masking outside | due M3 (null impl. satisfied M0) | low-medium |

Two further points that are not invariants but serve the same purpose: the
**phase model** is a configuration switch (`balanced | unbalanced`, default
balanced), because hard-wiring either one blocks a later study; and the **unit
canon** is a single internal representation, documented in the schema and enforced
by a test, because interval computations in certification do not forgive hidden
conversions.

### 14.1 What the contract does not deliver

It keeps the structure open, not the results. When M9 later restricts the action
space — which is the purpose of a shield — the training results from M3 to M7
change and have to be recomputed. That is not an architectural flaw but the nature
of the thing: the agent then optimises inside a different, smaller set. The
invariants guarantee that this rerun is a configuration change rather than a
reimplementation.

Also not covered: if the convex restriction turns out too slow at these grid
sizes, model reduction may be required, which touches L1. It is therefore worth
measuring the runtime of an SOCP on the target grid once, early, to quantify that
uncertainty before M9 starts.
