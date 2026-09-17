# lvgrid-rl

Training environment for reinforcement learning agents that autonomously control
low-voltage distribution grids. The agents resolve voltage band violations as
well as line and transformer overloads by curtailing PV infeed, shifting heat
pump and charging loads, and controlling battery storage.

Grid simulation with [pandapower](https://pandapower.readthedocs.io) and
[SimBench](https://simbench.de), agents with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io).

**Status: M2** -- skeleton, core schemas, data layer and grid layer. There is no
agent yet. What exists is the interface layer everything else builds on, a
verified reproducibility chain, time series preparation from SimBench, an AC
power flow engine, the EN 50160 assessment and configurable scenarios.

The full architecture is in `docs/architecture.md`.

## Installation

Recommended with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra sim              # M1: data layer including pandapower/simbench
uv sync --extra sim --extra rl --extra config   # from M3
```

`uv sync` installs the `dev` dependency group by default; the heavier
dependencies live in extras. Note that `--extra sim` is required for the data
layer -- without it, `pytest` silently skips 39 tests.

Alternatively with pip (requires pip >= 25.1 for `--group`):

```bash
pip install -e ".[sim]" --group dev
```

## Quick start

```bash
uv run python scripts/prepare_data.py --code 1-LV-rural1--2-sw --sim-dt 5
uv run python scripts/survey_grids.py --samples 150
uv run python scripts/run_baseline.py --scenario moderate_growth --stride 24
uv run pytest -v
```

`prepare_data.py` reads the SimBench time series, converts the time axis to UTC,
resamples to the simulation step size and writes a Parquet cache with a content
hash. `scripts/train.py` does not train anything yet; it resolves the
configuration, derives the seeds and writes a run manifest -- which verifies the
reproducibility chain before the first model exists.

## Two conventions to know before your first contribution

**Signs.** Consumer reference direction throughout, for *all* asset types:
`p_mw > 0` means drawing from the grid, `p_mw < 0` means feeding in. A PV system
therefore always has `p_mw <= 0`. Conversion to the pandapower convention
(`sgen` counts the other way) happens exclusively in the grid adapter. See
`lvgrid_rl.core.units.SIGN_CONVENTION`.

**Units.** MW, MVar, MWh, per-unit, percent (0-100, not 0-1), degrees Celsius,
kelvin, W/m². Every numeric field of a schema type carries a unit suffix, and a
test enforces it. The reason is not tidiness: the certification planned later
computes with intervals, and intervals do not forgive hidden conversion factors.

## Layout

| Package | Layer |
|---|---|
| `core` | Schemas, protocols, unit convention, information ordering |
| `data` | L0 data layer: sources, resampling, cache, scenarios |
| `grid` | L1 grid and physics |
| `components` | L2 asset and flexibility models |
| `env` | L3 Gymnasium environment |
| `agents`, `baselines` | L4 agents and reference methods |
| `eval`, `viz` | L5 KPIs, evaluation, visualisation |
| `experiment` | L6 orchestration and reproducibility |

## Time step sizes

`sim_dt` is restricted to 1, 2, 5 or 10 minutes. The reason is the voltage
criterion: EN 50160 assesses 10-minute mean values, so the simulation step must
divide 10 minutes exactly. This rules out 15 minutes -- which happens to be the
native resolution of the SimBench time series, so those are upsampled piecewise
constant, preserving energy.

`gamma` is derived from `control_dt` rather than chosen: the criterion refers to
a weekly interval, so the effective horizon has to cover one week. At a 15-minute
control step that gives 0.99851.

## Extensibility contract

The hard safety guarantees (robust convex restriction, predictive safety filter)
are deferred to M7b. So that they can be added later without rework, seven
invariants apply, each with a test in `tests/test_invariants.py`. Open ones are
marked `xfail(strict=True)`: once the corresponding component exists and the test
turns green unexpectedly, **the build breaks** and forces the marker to be
removed. The number of XFAIL reports is the contract's debt count.

| | Invariant | Due | Status |
|---|---|---|---|
| I1 | `SystemState` complete, separate from `Observation` | M3 | partially green |
| I2 | Asset dynamics as pure functions | M4 | open |
| I3 | Strict information ordering | M3 | mechanism green |
| I4 | Physical actions, affine invertible normalisation | M3 | open |
| I5 | Hypothetical power flow without side effects | M2 | **green** |
| I6 | Exogenous inputs carry bounds, ratings maintained | M1 | **green** |
| I7 | Both safety intervention points present | M3 | null implementation green |

The reason for the effort: promises of modularity decay silently. Six months
without a check and some reasonable shortcut has used up the extensibility
without anyone noticing.

## Working grid and scenario

`1-LV-rural1--2-sw` with the `moderate_growth` scenario. The published SimBench
scenarios produce no voltage band violation in any of the six low-voltage grids,
so the scenario raises the point of common coupling and adds moderate growth in
PV, heat pumps and charge points. Under it, 15 % of bus-weeks fail EN 50160 and
12.6 % of steps are thermally overloaded -- a control problem that is neither
trivially satisfied nor hopeless. See section 4.4 of the architecture document,
and reproduce with `scripts/survey_grids.py`.

## Next steps

* **M3** minimal environment: PV curtailment only, Gymnasium API, `pq_budget`
  observation features, both constraint terms in the reward, and baselines B0-B4
  with their own parameter search.

See section 12 of the architecture document for the full roadmap.

## License

MIT, see [`LICENSE`](LICENSE).

This does not extend to the datasets; their terms differ per source. The
repository therefore contains no raw data, only acquisition scripts and
manifests, see [`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff).
