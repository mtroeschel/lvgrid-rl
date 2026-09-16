# Contributing

## Setup

```bash
git clone https://github.com/mtroeschel/lvgrid-rl.git
cd lvgrid-rl
uv sync --extra sim
uv run pytest
```

`uv.lock` is version-controlled and part of the reproducibility chain. Anyone
changing a dependency commits the updated lock file as well -- CI runs
`uv sync --locked` and will otherwise fail.

The `sim` extra pulls in pandas, pyarrow, pandapower and simbench. Without it,
`pytest` silently skips the data-layer tests: 57 instead of 96 passing tests and
no error.

### If `uv run pytest` reports a missing module

Then a `pytest` from the system PATH is running instead of the one in `.venv`:
the `pytest` script carries its own interpreter in its shebang line, and that
interpreter does not see the environment's packages. Diagnosis:

```bash
uv run which pytest      # should point at .venv/bin/pytest
uv run python -c "import sys; print(sys.executable)"
```

If `which` points outside `.venv`, pytest is missing from the environment. The
usual cause is tooling declared under `[project.optional-dependencies]` instead
of `[dependency-groups]` -- uv does not install extras by default, but it does
install dependency groups.

## Language

Code, comments, docstrings, tests and repository documentation are written in
**English**. This includes error messages, because tests assert on them.

The switch from German happened after M1; files not yet converted are being
migrated file by file. When you touch a file, translate it as part of the same
change rather than leaving it half-converted.

## Three rules that come before everything else

These are the places where a new contribution does damage fastest, and all three
are guarded by tests.

**1. Signs.** Consumer reference direction throughout, for *all* asset types:
`p_mw > 0` means drawing from the grid, `p_mw < 0` means feeding in. A PV system
therefore always has `p_mw <= 0`. Conversion to the pandapower convention
(`sgen` counts the other way) happens exclusively in the grid adapter
(`lvgrid_rl.grid`) and nowhere else. See `lvgrid_rl.core.units.SIGN_CONVENTION`.

**2. Units.** Every numeric field of a schema type carries a unit suffix from
`lvgrid_rl.core.units.UNIT_SUFFIXES`, and a test enforces this across all schema
types. New fields without a unit must be added deliberately to `UNITLESS_FIELDS`
in `tests/test_core.py` -- that should be a decision, not an oversight.

**3. Invariants.** `tests/test_invariants.py` guards the seven invariants of the
extensibility contract (section 13 in `docs/architektur.md`). Open invariants are
marked `xfail(strict=True)`. When a component is implemented and its test turns
green unexpectedly, **the build breaks** -- at which point the marker is removed
and the acceptance criteria named in the docstring are written out. A marker
must never be removed without actually writing the test.

## Workflow

One branch per milestone, for example `m1-data-layer`, and a pull request even
when working alone. The PR is where the CI result, the decisions and their
reasoning are recorded -- for an academic project that is the audit trail, not
bureaucracy.

## Pre-commit hooks

Set up once per clone:

```bash
uv sync --extra sim
uv run pre-commit install
```

After that, every commit runs the same checks as CI. The concrete reason for
these hooks: in M0 the CI was red because `ruff` had never been run locally --
the tests were green and the linter had 62 violations.

To check the whole tree, for instance after cloning or after an `autoupdate`:

```bash
uv run pre-commit run --all-files
```

The test hook invokes `uv run --frozen python -m pytest` and therefore requires
`uv` on the PATH. Git hooks run with the PATH of the invoking terminal, not
inside the activated `.venv`. Calling `pytest` directly would hit a script from
the system PATH, and calling `python -m pytest` fails on systems that only have
`python3`, with "Executable `python` not found". If you work without uv, skip
the hook individually:

```bash
SKIP=pytest git commit -m "..."
```

and rely on CI. The ruff hooks are unaffected; pre-commit manages their
environments itself.

If the test suite becomes noticeably slower with pandapower and real training
runs, the hook belongs under `stages: [pre-push]`. In M1 the whole hook chain
takes a little over a second.

Two further points. First, `pre-commit` does not replace CI: the hooks only see
staged files and run in the local environment, whereas CI sets up the full tree
from scratch. Second, the ruff version in `.pre-commit-config.yaml` and in the
`dev` group of `pyproject.toml` is deliberately pinned to the same value -- if
hook and CI drift apart, one reports what the other lets through, and then
neither is trusted.

Without hooks, before pushing at least:

```bash
uv run ruff check src tests
uv run ruff format --check src tests scripts
uv run pytest -q
```

## No data in the repository

Raw data and derived caches are not version-controlled, see `data/README.md`.
Reproducibility runs through configuration, seeds and the data manifest.

## Declaring AI assistance

Parts of this project are produced with the help of an AI assistant. Such
contributions are declared via a `Co-authored-by` trailer in the commit or in
the pull request description. For academic publications, the disclosure rules of
the respective journal and institution apply in addition.
