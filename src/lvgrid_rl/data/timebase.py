"""Time base of the simulation: step sizes and time zones.

Two commitments meet here, both following from the voltage criterion
(section 6.6 of the architecture document):

**Permitted step sizes.** EN 50160 assesses ten-minute mean values. A simulation
step size must therefore divide ten minutes exactly, otherwise an assessment
window cannot be formed from simulation steps without interpolation. That rules
out the obvious fifteen minutes -- which happens to be the native resolution of
the SimBench time series.

**Time zones.** UTC applies throughout, internally. The reason is not tidiness
but a concrete trap in the source data: the SimBench time axis is German local
time *including* daylight saving time. Read naively it yields an index that is
neither monotonic nor unique -- four quarter-hours are missing on 2016-03-27 and
four occur twice on 2016-10-30. Anyone who misses that resamples on a broken
index and never finds out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

__all__ = [
    "PQ_WINDOW_MIN",
    "ALLOWED_SIM_DT_MIN",
    "DEFAULT_SOURCE_TZ",
    "WINDOWS_PER_WEEK",
    "TimeBase",
    "to_utc_index",
]

PQ_WINDOW_MIN: Final[int] = 10
"""Averaging interval of EN 50160, in minutes."""

ALLOWED_SIM_DT_MIN: Final[tuple[int, ...]] = (1, 2, 5, 10)
"""Step sizes that divide :data:`PQ_WINDOW_MIN` exactly."""

DEFAULT_SOURCE_TZ: Final[str] = "Europe/Berlin"
"""Time zone of the German source datasets (SimBench, WPuQ, HTW, DWD)."""

WINDOWS_PER_WEEK: Final[int] = 7 * 24 * 60 // PQ_WINDOW_MIN
"""1008 assessment windows per weekly interval."""


def to_utc_index(
    timestamps: pd.Series | pd.DatetimeIndex,
    source_tz: str = DEFAULT_SOURCE_TZ,
) -> pd.DatetimeIndex:
    """Convert naive local timestamps into a UTC index.

    The daylight saving transition is resolved via ``ambiguous="infer"``: pandas
    recognises the contiguous, ascending block of the repeated hour and assigns
    the correct UTC offsets. This works because the source data actually
    contains the repetition and stores it in order of occurrence.

    Args:
        timestamps: Naive timestamps in local time.
        source_tz: Time zone of the source data.

    Returns:
        A strictly increasing, unique index in UTC.

    Raises:
        ValueError: if the transition cannot be resolved, or if the result is
            not monotonic or not unique. Both point at a source file with gaps
            or disturbed ordering, and neither may be processed silently.
    """
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    try:
        localized = idx.tz_localize(source_tz, ambiguous="infer", nonexistent="raise")
    except Exception as exc:  # pragma: no cover - depends on source data
        raise ValueError(
            f"Timestamps could not be localised to {source_tz}: {exc}. The usual "
            "cause is a source file in which the repeated hour of the daylight "
            "saving transition is missing or out of order."
        ) from exc
    utc = localized.tz_convert("UTC")
    if not utc.is_monotonic_increasing:
        raise ValueError("UTC index is not monotonically increasing")
    if not utc.is_unique:
        raise ValueError("UTC index contains duplicate timestamps")
    return utc


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Step sizes of the simulation and of the controller.

    Args:
        sim_dt_min: Simulation step size, must be in
            :data:`ALLOWED_SIM_DT_MIN`.
        control_dt_min: Control step size, must be a multiple of
            ``sim_dt_min``. An offset against the ten-minute assessment grid is
            explicitly permitted and realistic (section 6.6): a fifteen-minute
            control cycle on five-minute physics is a valid case.

    Example:
        >>> tb = TimeBase(sim_dt_min=5, control_dt_min=15)
        >>> tb.sim_steps_per_pq_window
        2
        >>> tb.sim_steps_per_control_step
        3
        >>> TimeBase(sim_dt_min=15, control_dt_min=15)
        Traceback (most recent call last):
            ...
        ValueError: sim_dt_min=15 is not permitted...
    """

    sim_dt_min: int
    control_dt_min: int

    def __post_init__(self) -> None:
        if self.sim_dt_min not in ALLOWED_SIM_DT_MIN:
            raise ValueError(
                f"sim_dt_min={self.sim_dt_min} is not permitted; allowed are "
                f"{ALLOWED_SIM_DT_MIN}. The step size must divide the "
                f"{PQ_WINDOW_MIN}-minute assessment interval of EN 50160 exactly."
            )
        if self.control_dt_min % self.sim_dt_min != 0:
            raise ValueError(
                f"control_dt_min={self.control_dt_min} is not a multiple of "
                f"sim_dt_min={self.sim_dt_min}"
            )

    @property
    def sim_steps_per_pq_window(self) -> int:
        """Simulation steps per ten-minute assessment window."""
        return PQ_WINDOW_MIN // self.sim_dt_min

    @property
    def sim_steps_per_control_step(self) -> int:
        """Simulation steps per control decision."""
        return self.control_dt_min // self.sim_dt_min

    @property
    def control_steps_per_week(self) -> int:
        """Control decisions per weekly interval."""
        return 7 * 24 * 60 // self.control_dt_min

    def suggested_gamma(self) -> float:
        """Discount factor whose effective horizon covers one week.

        Gamma is not a free hyperparameter in this project: the voltage
        criterion refers to a weekly interval, and a habitual ``0.99`` would
        simply not see that horizon (section 6.5).

        >>> round(TimeBase(5, 15).suggested_gamma(), 5)
        0.99851
        """
        return 1.0 - 1.0 / self.control_steps_per_week

    @property
    def freq(self) -> pd.Timedelta:
        """Simulation step size as a ``Timedelta``."""
        return pd.Timedelta(minutes=self.sim_dt_min)
