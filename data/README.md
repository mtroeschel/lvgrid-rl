# Data

This directory contains **no** data. Raw data and derived caches are excluded in
`.gitignore`, because the terms of use differ per source and because versioned
binary data makes a repository history unusable.

Reproducibility instead rests on three things: the acquisition scripts in
`scripts/prepare_data.py`, the preparation configuration in `configs/data/`, and
a `manifest.json` holding a SHA-256 per prepared frame. Whoever obtains the same
raw data gets the same manifest hash and therefore the same processing state.

## Sources

| Source | Resolution | Use | Reference |
|---|---|---|---|
| SimBench | 15 min, 1 year | primary source M1–M4; load, PV, storage, heat pumps, charge points | https://simbench.de/en/download/datasets/ |
| WPuQ (ISFH) | 10 s / 1 / 15 / 60 min | from M5; household and heat pump measured separately, plus the substation | DOI 10.5281/zenodo.5642902 |
| HTW Berlin | 1 s | from M5; high-resolution household load, phase-resolved | https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/ |
| emobpy | 15 min, configurable | EV availability and charging demand | https://emobpy.readthedocs.io |
| ElaadNL | distributions | calibration of the EV session generator | https://elaad.nl/en/open-datasets/ |
| when2heat | 1 h | COP characteristics and annual heat demand shape | https://data.open-power-system-data.org/when2heat/ |
| DWD Open Data | 10 min | global irradiance, temperature | https://opendata.dwd.de/climate_environment/CDC/ |

## Notes on the SimBench data

Two properties cost time if discovered late, so they are recorded here.

**The time axis is German local time including daylight saving.** Read naively
the index is neither monotonic nor unique: four quarter-hours are missing on
2016-03-27 and four occur twice on 2016-10-30. `lvgrid_rl.data.timebase` converts
to UTC and verifies the result.

**The ZIP load model columns are undefined.** simbench 1.6.1 leaves
`const_z_p_percent` and its three siblings as `NaN`, which makes the pandapower
3.x Jacobian singular so that no low-voltage grid converges. See
`lvgrid_rl.grid.loader.fix_zip_load_model`.

## Notes on the when2heat data

Version 2023-07-27, single-index CSV (about 330 MB), licence CC-BY 4.0; cite
Ruhnau, Hirth, Praktiknjo (2019), Scientific Data 6:189. Obtain it with
`uv run python scripts/fetch_when2heat.py`, which stores it under
`data/raw/when2heat/` and checks its SHA-256
(`f1f71790158d1de08403eea32dea7a2732050870c499938135606d9d7faac0fa`). Only the
COP columns are used (M4, decision D15).

**Semicolons and decimal commas.** Read without `decimal=","`, every value is a
string.

**The published `utc_timestamp` is off by the UTC offset** — one hour early in
winter, two in summer. The values are on a UTC clock but were written onto the
local wall clock as if it were UTC and then converted. Shown for the COP columns
by cross-correlation with hourly DWD air temperature (six stations, UTC;
`scripts/check_when2heat_timing.py`): on the published axis the COP leads by
1 h in every winter and 2–3 h in every summer 2010–2022, on the corrected axis
by 0 h in winter (r = 0.97) and 0–1 h in summer (r = 0.3–0.45, a weak diurnal
signal). The visible symptom is a missing UTC hour 01:00 on every autumn change;
on the corrected axis the autumn is complete and 02:00 of every spring change is
missing instead — the true value of the local hour that does not exist.
`lvgrid_rl.data.sources.when2heat.read_cop` reads `cet_cest_timestamp` with the
offset dropped as UTC, fills exactly the fifteen spring hours, and refuses any
other gap. The heat demand columns are not used and not claimed either way.

DWD data for that check: Open Data of the Deutscher Wetterdienst, Datenlizenz
Deutschland – Namensnennung – Version 2.0, stored under `data/raw/dwd/`.

## Notes on the emobpy data (decision D17)

Generated, not downloaded: `tools/emobpy/generate.py` runs emobpy 0.6.2 in a
frozen environment of its own and writes `data/raw/emobpy/<run>/` — one Parquet
file per charge point and a `manifest.json` with versions, seeds, driver types,
vehicle models, every retry, and the SHA-256 of each file. The project reads
these files only and verifies them against the manifest.

    uv run python scripts/list_charge_points.py      # project env: configs/ev/charge_points.json
    cd tools/emobpy && uv sync --locked && uv run python generate.py

A full run (7 charge points, 53 weeks) takes about two hours on 12 workers.
emobpy (MIT licence) uses the mobility statistics of *Mobilität in Deutschland*
2017 and ERA5 weather; cite Gaete-Morales et al. (2021), Scientific Data 8:152.

Two defects of emobpy are worked around and recorded in the tool: its
`set_seed` does not seed the generator the mobility model uses, and its tour
sampling has no iteration limit (`docs/architecture.md`, §5, D17).

ElaadNL, which D16 had relied on, publishes no distribution file — only a
profile generator — and is not used.

## To clarify before publication

Whether the derived cache may be redistributed differs per source. Until that is
checked per dataset the rule is: third parties obtain the raw data themselves,
and the manifest hash shows that it is the same data.
