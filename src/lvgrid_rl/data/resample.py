"""Resampling of time series onto the uniform simulation step size.

The aggregation rule depends on the physical meaning of a quantity, not on its
data type. A mean over power preserves energy; a mean over a binary availability
flag is nonsense. The mapping therefore lives explicitly in the configuration
rather than as a default in the code (table in section 3.3 of the architecture
document).

When upsampling -- SimBench supplies fifteen minutes, the simulation needs five
-- the same reasoning applies in reverse: power is carried forward piecewise
constant, because that preserves the energy of the source interval. Linear
interpolation would not, and it would invent ramps that are not in the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

from lvgrid_rl.data.timebase import TimeBase

__all__ = [
    "Policy",
    "DEFAULT_POLICIES",
    "resample_series",
    "resample_frame",
    "energy_error",
]


class Policy(StrEnum):
    """Aggregation rule for a quantity.

    Attributes:
        MEAN: Power. Energy-preserving mean when downsampling, piecewise
            constant when upsampling.
        DIFF: Energy meter readings. Difference over the interval when
            downsampling.
        LAST: State quantities such as state of charge. End value of the
            interval.
        MAJORITY: Binary quantities such as "vehicle plugged in". Majority vote.
        LINEAR: Continuous ambient quantities such as temperature.
    """

    MEAN = "mean"
    DIFF = "diff"
    LAST = "last"
    MAJORITY = "majority"
    LINEAR = "linear"


DEFAULT_POLICIES: Mapping[str, Policy] = {
    "power": Policy.MEAN,
    "energy_counter": Policy.DIFF,
    "irradiance": Policy.MEAN,
    "temperature": Policy.LINEAR,
    "availability": Policy.MAJORITY,
    "state": Policy.LAST,
}
"""Mapping quantity kind -> rule, as in ``configs/data/default.yaml``."""


def _infer_source_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Determine the step size of a regular index.

    Raises:
        ValueError: on an irregular index. That is almost always an unhandled
            daylight saving transition (see
            :func:`lvgrid_rl.data.timebase.to_utc_index`) and must not be
            papered over with an estimated frequency.
    """
    diffs = pd.Series(index).diff().dropna().unique()
    if len(diffs) != 1:
        raise ValueError(
            f"Irregular time index, {len(diffs)} distinct step sizes: "
            f"{sorted(pd.to_timedelta(diffs))[:5]}. Usually an index that was "
            "not converted to UTC and still carries a daylight saving transition."
        )
    return pd.Timedelta(diffs[0])


def resample_series(s: pd.Series, target: pd.Timedelta, policy: Policy) -> pd.Series:
    """Bring a time series onto the target step size.

    Args:
        s: Time series with a regular, timezone-aware index.
        target: Target step size.
        policy: Aggregation rule.

    Returns:
        A time series at the target step size covering the same period.

    Raises:
        ValueError: if source and target step size are not in an integer ratio.
            Any conversion would then be an interpolation with a grid offset,
            and the resulting error would not be traceable.
    """
    source = _infer_source_freq(pd.DatetimeIndex(s.index))
    if source == target:
        return s.copy()

    if source > target:  # upsampling
        if source % target != pd.Timedelta(0):
            raise ValueError(
                f"Source step size {source} is not a multiple of the target step "
                f"size {target}"
            )
        # The last source value covers a full source interval, so the index has
        # to extend beyond the last original timestamp -- otherwise a partial
        # interval is missing at the end of the year.
        end = s.index[-1] + source - target
        new_index = pd.date_range(s.index[0], end, freq=target)
        if policy in (Policy.MEAN, Policy.MAJORITY, Policy.LAST, Policy.DIFF):
            # Piecewise constant. For MEAN this preserves the energy of the
            # source interval; for DIFF the meter increment is spread evenly.
            out = s.reindex(new_index, method="ffill")
            if policy == Policy.DIFF:
                out = out / (source // target)
            return out
        if policy == Policy.LINEAR:
            return (
                s.reindex(s.index.union(new_index)).interpolate("time").reindex(new_index)
            )
        raise AssertionError(f"Unhandled rule {policy}")

    # downsampling
    if target % source != pd.Timedelta(0):
        raise ValueError(
            f"Target step size {target} is not a multiple of the source step "
            f"size {source}"
        )
    grouper = s.resample(target, label="left", closed="left")
    if policy == Policy.MEAN:
        return grouper.mean()
    if policy == Policy.DIFF:
        return grouper.sum()
    if policy == Policy.LAST:
        return grouper.last()
    if policy == Policy.LINEAR:
        return grouper.mean()
    if policy == Policy.MAJORITY:
        # Majority vote over 0/1. A tie resolves in favour of "available",
        # because wrongly marking a charge point unavailable throws away
        # flexibility that was really there.
        return (grouper.mean() >= 0.5).astype(float)
    raise AssertionError(f"Unhandled rule {policy}")


def resample_frame(
    df: pd.DataFrame,
    timebase: TimeBase,
    policies: Mapping[str, Policy],
    default: Policy = Policy.MEAN,
) -> pd.DataFrame:
    """Resample all columns of a frame according to per-column rules.

    Args:
        df: Time series in wide format, index in UTC.
        timebase: Target step size.
        policies: Mapping column name -> rule.
        default: Rule for columns without an entry.

    Returns:
        A frame at the target step size.
    """
    target = timebase.freq
    cols = {
        name: resample_series(df[name], target, policies.get(name, default))
        for name in df.columns
    }
    out = pd.DataFrame(cols)
    out.index.name = df.index.name
    return out


def energy_error(original: pd.Series, resampled: pd.Series) -> float:
    """Relative energy error of a power series after resampling.

    The values are interval means of power, not point samples. Energy is
    therefore ``sum(p) * dt`` (rectangle rule); a trapezoidal rule would be
    wrong here because it half-weights the boundary intervals.

    For :attr:`Policy.MEAN` the error must be exactly zero as long as the time
    ranges match. Used as a check quantity in tests and in data validation.
    """
    dt_orig = _infer_source_freq(pd.DatetimeIndex(original.index))
    dt_new = _infer_source_freq(pd.DatetimeIndex(resampled.index))
    e_orig = float(original.sum()) * dt_orig.total_seconds()
    e_new = float(resampled.sum()) * dt_new.total_seconds()
    if e_orig == 0.0:
        return abs(e_new)
    return abs(e_new - e_orig) / abs(e_orig)
