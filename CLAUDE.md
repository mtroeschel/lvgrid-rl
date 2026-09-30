# Working on this repository

Research code for a thesis: reinforcement learning agents that autonomously
control a low-voltage distribution grid. Correctness and traceability matter more
than speed — every figure in a results table has to be defensible.

## Read these first

| | |
|---|---|
| `docs/architecture.md` | the design contract. §12 is the roadmap, §13 the decision record, §14 the extensibility contract |
| `docs/results/m3.md` | what M3 measured, what it did not, and what follows for M4 |
| `CONTRIBUTING.md` | conventions, pre-commit setup, the three rules below |
| `tests/test_invariants.py` | the seven invariants; the docstrings carry the acceptance criteria |

Implementation follows the architecture document. When a change contradicts it,
update the document in the same change rather than letting the two drift — and
say so in the commit.

## Current state

M0 to M3 are complete. **M4 is in progress: the full actuator set** — battery
storage, heat pump with a buffer model, EV charge points with a session model,
plus action mode 2 (per asset type). The steps are in `docs/architecture.md`
§12; 4.0 (reward mode), the M3 corrections and 4.1 (battery model) are done. **Invariant I2 is satisfied since 4.1**: asset dynamics
are pure functions over a frozen `AssetState`. Its test in
`tests/test_invariants.py` is parametrised over asset types — every new asset
model (heat pump, charge point) adds a case to `_i2_assets()`, it does not get a
test of its own. The protocol passes `hold_min` to `to_setpoint` and `dt_min` to
`limit_to_physics` and `dynamics`; `AssetOutcome` holds energies over the step.

**The reward mode for M4 is settled: fixed weights** (`docs/results/m4-lagrangian.md`,
§13 D5). The Lagrangian mode was validated on the M3 setup against pre-registered
criteria: it holds the constraints but three of five policies were dominated by a
fixed cap. It stays in the code as the M7 comparison arm. Do not reopen it without
a new pre-registered test; the known weak point is the thermal limit of zero,
under which the multiplier can only grow.

## Three rules that come before everything else

1. **Signs.** Consumer reference direction for *all* asset types: `p_mw > 0` is
   drawing from the grid, `p_mw < 0` is feeding in. PV always has `p_mw <= 0`.
   The conversion to pandapower's convention happens only in `lvgrid_rl.grid`.
2. **Units.** Every numeric schema field carries a suffix from
   `lvgrid_rl.core.units.UNIT_SUFFIXES`; a test enforces it across all schema
   types.
3. **Invariants.** Never remove an `xfail` marker without writing the test the
   docstring describes.

## Traps that have already cost time

- **`uv run` syncs only the default groups.** Packages from an extra keep
  whatever version is installed, so a lock file change has no effect until the
  extras are named: `uv sync --locked --extra sim --extra env --extra rl`.
- **`sim_dt` is restricted to 1 and 5 minutes.** EN 50160 permits {1, 2, 5, 10},
  but the 15-minute SimBench series cannot produce 2 or 10 without a grid offset.
- **Evaluation must use `EpisodeMode.EVALUATE`** — complete weeks, fixed order,
  zero budget. Sampled episodes differ per seed and cannot be aggregated; that
  bug once made deterministic baselines differ between runs.
- **Never compare against untuned baselines.** `train.py` refuses to start
  without `configs/baseline/tuned.json`, and the droop search is capped at
  `v_max <= 1.10` because a characteristic acting only beyond the band cannot
  hold it.
- **Training budget is 600,000 steps.** At 200,000 a run is still on the steep
  part of the learning curve, and the spread across seeds is budget noise rather
  than seed variance.
- **`configs/agent/ppo.yaml` is what trains.** It holds the Stable-Baselines3
  defaults that M3 and M4.0 ran on (`n_steps` 2,048 per worker, 16,384 per
  rollout with eight workers, 64×64 network). Until the M3 corrections the file
  was not read at all; changing a value in it now is a new experiment.
- **Result files from M3 and M4.0 are KPI schema 1**: their overload is three
  times the integral. The scripts refuse to mix them with current (schema 2)
  files; convert by dividing overload by 3, or recompute.
- **A standard-conforming evaluation costs about 20 minutes per controller**
  over the nine test weeks. Use `--policy-only` and `--baselines-from` for
  additional seeds.
- **No data or run artefacts in the repository.** `.gitignore` patterns are
  anchored with a leading slash on purpose: an unanchored `results/` once
  matched `docs/results/` and silently excluded a document.

## Before committing

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```

Work on a branch per milestone step, open a pull request even when working alone,
and describe in it what was measured and what was decided — for this project the
pull request is the audit trail, not bureaucracy. Declare AI assistance as
`CONTRIBUTING.md` sets out.
