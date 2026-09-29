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

M0 to M3 are complete. **M4 is next: the full actuator set** — battery storage,
heat pump with a buffer model, EV charge points with a session model, plus action
mode 2 (per asset type). **Invariant I2 falls due**: asset dynamics must be pure
functions over a copyable `AssetState`, because a later predictive safety filter
rolls states forward hypothetically. Its `xfail(strict=True)` test will break the
build the moment `components/bess.py` exists — that is intended, and the test is
then written out rather than the marker removed.

**One decision is open and should be settled before the first asset model**
(`docs/results/m3.md`, consequence 2): keep the fixed reward weights and report
the Pareto frontier, or switch to the Lagrangian mode where the limits are
normative (`d = 0.05` is the EN 50160 criterion itself) and the agent minimises
curtailment subject to them. The mode exists since M3 and has never been used. It
decides how every M4 result is to be read. Ask rather than assume.

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
