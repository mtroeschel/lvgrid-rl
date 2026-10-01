"""when2heat: coefficients of performance of heat pumps, hourly, per country.

Ruhnau, Hirth, Praktiknjo (2019), *Time series of heat demand and heat pump
efficiency for energy system modeling*, Scientific Data 6:189; data package
version 2023-07-27, Open Power System Data, licence CC-BY 4.0.

**What it is used for (M4, 4.3).** SimBench gives the heat pumps of a grid as
*electrical* load profiles. A controllable heat pump needs the *thermal* demand
it serves, so that it can serve it at a different time. Decision for 4.3: the
thermal demand is the SimBench electrical profile times the when2heat COP of the
matching heat source -- air-source for ``Air_*`` profiles, ground-source for
``Soil_*`` -- in the same year, 2016. That keeps the energy, the seasonality and
the daily pattern of the grid dataset and takes only the efficiency from
when2heat. A heat pump that ran exactly as SimBench says would then draw exactly
the SimBench profile; shifting it in time changes the electricity needed only
through the COP of the hours it moves between.

**Simplification.** The COP is a time series per heat source and heat sink, not
a function of the buffer temperature. In stage 1 the buffer is an energy
reservoir (architecture section 5), and its temperature does not feed back into
the efficiency. The sink defaults to floor heating.

**A defect in the data.** In every year the UTC hour 01:00 of the autumn
daylight-saving night is missing -- the repeated local hour 02:00, presumably
dropped when the series was built in local time. :func:`read_cop` fills exactly
that hour by linear interpolation and refuses any other gap. It is the same
class of trap as the SimBench time axis (``data/README.md``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "WHEN2HEAT_VERSION",
    "WHEN2HEAT_SHA256",
    "WHEN2HEAT_URL",
    "SINKS",
    "CopSeries",
    "heat_source_of",
    "read_cop",
    "cop_on_index",
    "thermal_demand_mw",
]

WHEN2HEAT_VERSION = "2023-07-27"
WHEN2HEAT_URL = (
    f"https://data.open-power-system-data.org/when2heat/{WHEN2HEAT_VERSION}/when2heat.csv"
)
WHEN2HEAT_SHA256 = "f1f71790158d1de08403eea32dea7a2732050870c499938135606d9d7faac0fa"
"""Of the single-index CSV of the version above, as downloaded on 2026-10-01."""

SINKS = ("floor", "radiator", "water")
_SOURCES = ("ASHP", "GSHP")


def heat_source_of(profile: str) -> str:
    """when2heat heat source for a SimBench heat pump profile name.

    >>> heat_source_of("Air_Semi-Parallel_2")
    'ASHP'
    >>> heat_source_of("Soil_Alternative_2")
    'GSHP'
    """
    if profile.startswith("Air_"):
        return "ASHP"
    if profile.startswith("Soil_"):
        return "GSHP"
    raise ValueError(f"{profile!r} is not a SimBench heat pump profile")


@dataclass(frozen=True, slots=True)
class CopSeries:
    """Hourly COP per heat source and sink, on a complete UTC axis.

    Args:
        frame: Columns ``"<source>_<sink>"``, e.g. ``"GSHP_floor"``; index
            hourly in UTC without gaps.
        filled_utc: Hours that were missing in the source and filled by
            interpolation. Reported so that the repair stays visible.
        country: Country code of the columns read.
    """

    frame: pd.DataFrame
    filled_utc: tuple[pd.Timestamp, ...]
    country: str

    def column(self, source: str, sink: str = "floor") -> pd.Series:
        """One series, e.g. ``column("GSHP", "floor")``."""
        return self.frame[f"{source}_{sink}"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_autumn_dst_hour(ts: pd.Timestamp) -> bool:
    """01:00 UTC on the last Sunday of October: the repeated local hour."""
    if ts.month != 10 or ts.hour != 1 or ts.minute != 0 or ts.weekday() != 6:
        return False
    return (ts + pd.Timedelta(days=7)).month == 11


def read_cop(
    path: Path | str,
    country: str = "DE",
    verify: bool = True,
) -> CopSeries:
    """Read the COP columns of one country and repair the time axis.

    Args:
        path: The when2heat single-index CSV.
        country: Two-letter code.
        verify: Check the file against :data:`WHEN2HEAT_SHA256`. Disable only
            for test fixtures; a different file is a different dataset.

    Raises:
        ValueError: on a checksum mismatch, a missing column, a gap other than
            the autumn daylight-saving hour, a duplicate timestamp, or a COP
            outside ``[1, 10]``.
    """
    path = Path(path)
    if verify:
        digest = _sha256(path)
        if digest != WHEN2HEAT_SHA256:
            raise ValueError(
                f"{path} has SHA-256 {digest}, expected {WHEN2HEAT_SHA256} "
                f"(when2heat {WHEN2HEAT_VERSION}). Download it with "
                "scripts/fetch_when2heat.py."
            )
    wanted = {
        f"{country}_COP_{src}_{sink}": f"{src}_{sink}"
        for src in _SOURCES
        for sink in SINKS
    }
    header = pd.read_csv(path, sep=";", nrows=0).columns
    missing = sorted(set(wanted) - set(header))
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    # Semicolon-separated with decimal commas, the German spreadsheet
    # convention; read without ``decimal`` every value is a string.
    raw = pd.read_csv(path, sep=";", decimal=",", usecols=["utc_timestamp", *wanted])
    non_numeric = [c for c in wanted if not pd.api.types.is_numeric_dtype(raw[c])]
    if non_numeric:
        raise ValueError(f"{path}: non-numeric COP columns {non_numeric}")
    index = pd.DatetimeIndex(pd.to_datetime(raw.pop("utc_timestamp"), utc=True))
    if index.has_duplicates:
        raise ValueError(f"{path} has duplicate UTC timestamps")
    frame = raw.rename(columns=wanted).set_axis(index)

    full = pd.date_range(index[0], index[-1], freq="1h", tz="UTC")
    missing_hours = full.difference(index)
    unexpected = [ts for ts in missing_hours if not _is_autumn_dst_hour(ts)]
    if unexpected:
        raise ValueError(
            f"{path} has gaps other than the autumn daylight-saving hour, first "
            f"at {unexpected[0]}; refusing to interpolate over them"
        )
    frame = frame.reindex(full).interpolate("time")
    if ((frame < 1.0) | (frame > 10.0)).any().any():
        raise ValueError(f"{path}: COP outside [1, 10]")
    return CopSeries(frame=frame, filled_utc=tuple(missing_hours), country=country)


def cop_on_index(cop: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Interpolate an hourly COP series linearly onto a simulation index.

    Linear, because the COP follows the ambient temperature, a continuous
    quantity (``Policy.LINEAR`` in the resampling rules). The index must lie
    inside the hourly series; extrapolating an efficiency is not done.
    """
    if index[0] < cop.index[0] or index[-1] > cop.index[-1]:
        raise ValueError(
            f"index {index[0]} .. {index[-1]} is not covered by the COP series "
            f"{cop.index[0]} .. {cop.index[-1]}"
        )
    joined = cop.reindex(cop.index.union(index)).interpolate("time")
    return joined.reindex(index)


def thermal_demand_mw(electrical_mw: pd.Series, cop: pd.Series) -> pd.Series:
    """Thermal demand served by a heat pump profile: electrical times COP.

    Both series on the same index, consumer convention: the electrical profile
    is a draw, ``>= 0``, and so is the thermal demand.
    """
    if not electrical_mw.index.equals(cop.index):
        raise ValueError("electrical profile and COP must share one index")
    if (electrical_mw < -1e-12).any():
        raise ValueError("a heat pump profile must not feed in")
    return electrical_mw.clip(lower=0.0) * cop
