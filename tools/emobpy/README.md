# emobpy tool (decision D17)

Generates one EV-year per charge point with emobpy and writes it to
`data/raw/emobpy/<run>/`. A tool, not part of the package: emobpy 0.6.2 is
unmaintained and runs only in this frozen environment (Python 3.9, numpy < 1.24,
pandas < 2). The project never imports emobpy; it reads the Parquet files and
checks them against the SHA-256 in `manifest.json`.

```bash
uv run python scripts/list_charge_points.py   # in the project root, project env
cd tools/emobpy
uv sync --locked
uv run python generate.py                     # about 2.5 hours on 12 workers
```

What it generates, which shares and seeds it uses, and the three workarounds
for emobpy's defects are documented at the top of `generate.py` and in
`docs/architecture.md` §5 (D17).
