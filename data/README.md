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

## To clarify before publication

Whether the derived cache may be redistributed differs per source. Until that is
checked per dataset the rule is: third parties obtain the raw data themselves,
and the manifest hash shows that it is the same data.
