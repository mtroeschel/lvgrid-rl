#!/usr/bin/env python3
"""Check the time axis of when2heat's COP against measured air temperature.

    uv run python scripts/check_when2heat_timing.py

The COP is a function of the momentary ambient temperature, so its diurnal cycle
must line up with measured temperature at a lag of zero, in winter and summer
alike. This script cross-correlates the diurnal component of the German
air-source COP (floor heating) with the mean of six DWD stations' hourly air
temperature (UTC; Berlin-Tempelhof, Frankfurt/Main, Hamburg-Fuhlsbüttel,
Hannover, München-Stadt, Würzburg) and prints the lag of best correlation per
season and year -- once on when2heat's published ``utc_timestamp``, once on its
local wall clock with the offset dropped, which is how
``lvgrid_rl.data.sources.when2heat.read_cop`` reads it.

The result that decided the adapter (2026-10-01): published axis, lead of 1 h in
every winter and 2-3 h in every summer 2010-2022; corrected axis, 0 h in winter
(r = 0.97) and 0 to 1 h in summer (r = 0.3-0.45).

DWD data: Open Data of the Deutscher Wetterdienst, Datenlizenz Deutschland --
Namensnennung -- Version 2.0. Stored under ``data/raw/dwd/``.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from lvgrid_rl.data.sources.when2heat import WHEN2HEAT_VERSION

DWD = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/hourly/air_temperature/historical/"
)
STATIONS = (
    "stundenwerte_TU_00433_19510101_20251231_hist.zip",
    "stundenwerte_TU_01420_19810101_20251231_hist.zip",
    "stundenwerte_TU_01975_19490101_20251231_hist.zip",
    "stundenwerte_TU_02014_19490101_20251231_hist.zip",
    "stundenwerte_TU_03379_19970701_20251231_hist.zip",
    "stundenwerte_TU_05705_19480101_20251231_hist.zip",
)


def _station(path: Path) -> pd.Series:
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.startswith("produkt"))
        frame = pd.read_csv(
            io.BytesIO(archive.read(name)), sep=";", skipinitialspace=True
        )
    frame.columns = [c.strip() for c in frame.columns]
    index = pd.to_datetime(frame["MESS_DATUM"].astype(str), format="%Y%m%d%H", utc=True)
    return pd.Series(frame["TT_TU"].to_numpy(), index=index).where(lambda x: x > -999)


def _diurnal(series: pd.Series) -> pd.Series:
    return series - series.rolling(24, center=True).mean()


def _hourly(series: pd.Series) -> pd.Series:
    full = pd.date_range(series.index[0], series.index[-1], freq="1h", tz="UTC")
    return series[~series.index.duplicated()].reindex(full).interpolate()


def main() -> None:
    """Download the stations if needed and print the lag table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwd-dir", type=Path, default=Path("data/raw/dwd"))
    parser.add_argument(
        "--when2heat",
        type=Path,
        default=Path(f"data/raw/when2heat/when2heat-{WHEN2HEAT_VERSION}.csv"),
    )
    args = parser.parse_args()

    args.dwd_dir.mkdir(parents=True, exist_ok=True)
    stations = []
    for name in STATIONS:
        path = args.dwd_dir / name
        if not path.exists():
            urllib.request.urlretrieve(DWD + name, path)  # noqa: S310 - fixed https URL
        stations.append(_station(path).loc["2009":"2022"])
    temperature = _diurnal(pd.concat(stations, axis=1).mean(axis=1))

    raw = pd.read_csv(
        args.when2heat,
        sep=";",
        decimal=",",
        usecols=["utc_timestamp", "cet_cest_timestamp", "DE_COP_ASHP_floor"],
    )
    values = raw["DE_COP_ASHP_floor"].to_numpy()
    published = pd.Series(values, index=pd.to_datetime(raw["utc_timestamp"], utc=True))
    wall_clock = pd.to_datetime(
        raw["cet_cest_timestamp"].str.slice(0, 19)
    ).dt.tz_localize("UTC")
    corrected = pd.Series(values, index=pd.DatetimeIndex(wall_clock))
    axes = {
        "published": _diurnal(_hourly(published)),
        "corrected": _diurnal(_hourly(corrected)),
    }

    def best_lag(cop: pd.Series, start: str, end: str) -> str:
        t, c = temperature[start:end], cop[start:end]
        scores = {lag: t.corr(c.shift(-lag)) for lag in range(-4, 5)}
        lag = max(scores, key=scores.get)
        return f"{lag:+d} h (r={scores[lag]:.2f})"

    print(
        "lag of best correlation, COP relative to DWD temperature (negative: COP early)"
    )
    print(
        f"{'year':6s}{'published winter':>20s}{'published summer':>20s}"
        f"{'corrected winter':>20s}{'corrected summer':>20s}"
    )
    for year in range(2010, 2023):
        winter, summer = (
            (f"{year}-01-05", f"{year}-02-28"),
            (f"{year}-06-01", f"{year}-07-31"),
        )
        cells = [best_lag(axes[a], *season) for a in axes for season in (winter, summer)]
        print(f"{year:<6d}" + "".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()
