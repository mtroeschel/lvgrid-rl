"""Core schemas of the simulation: state, exogenous inputs, asset ratings.

These types are the interface every later layer builds on. They are fixed first
in M0 because any component created before them would afterwards have to be
bent to fit.

Three design decisions, each justified by the extensibility contract
(section 13 of the architecture document):

* **Everything immutable** (``frozen=True``). Changes go through
  :func:`dataclasses.replace`. Asset dynamics are therefore necessarily pure
  functions (invariant I2), and hypothetical roll-outs for a later predictive
  safety filter work without rework.
* **numpy arrays are made read-only.** ``frozen=True`` protects the reference,
  not the contents. :func:`freeze_array` therefore sets ``flags.writeable =
  False``. Without it, immutability is a claim rather than a fact.
* **Exogenous inputs carry bounds, not just values** (invariant I6). In M1
  ``bounds`` is filled trivially from the asset ratings and used by nothing.
  The plumbing exists, though, and that is the point: retrofitting uncertainty
  sets later would mean touching every data path.

For units and signs see :mod:`lvgrid_rl.core.units`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "freeze_array",
    "Interval",
    "AssetRatings",
    "ExogenousInput",
    "GridState",
    "PQWindowState",
    "PQBudgetState",
    "AssetState",
    "SystemState",
]


def freeze_array(a: np.ndarray) -> np.ndarray:
    """Return a read-only view of ``a``.

    Deliberately not a copy: the schemas sit on the hot training path and a copy
    per time step would be noticeable. By calling this, the caller gives up
    ownership of the array.

    >>> arr = freeze_array(np.array([1.0, 2.0]))
    >>> arr.flags.writeable
    False
    """
    view = a.view()
    view.flags.writeable = False
    return view


# ---------------------------------------------------------------------------
# Uncertainty and equipment limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """Closed interval ``[lo, hi]``.

    Deliberately carries no unit in its field names: the unit follows from the
    field the interval is used in (for example ``p_bounds_mw``).
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo <= self.hi:
            raise ValueError(f"Empty interval: lo={self.lo} > hi={self.hi}")

    def contains(self, x: float, tol: float = 1e-9) -> bool:
        """Is ``x`` inside the interval, allowing for floating point noise?"""
        return self.lo - tol <= x <= self.hi + tol

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.hi - self.lo


@dataclass(frozen=True, slots=True)
class AssetRatings:
    """Physical limits of an asset at its grid connection point.

    Basis of the *deterministic* uncertainty sets (section 6.9, P2, of the
    architecture document). Recording these values now costs nothing; adding
    them later for all scenarios is manual work on the grid dataset.

    Signs follow the consumer reference direction: for a PV system ``p_min_mw``
    is negative (full infeed) and ``p_max_mw == 0.0``.
    """

    p_min_mw: float
    p_max_mw: float
    s_max_mva: float | None = None
    """Inverter apparent power limit, where reactive power capability exists."""
    fuse_rating_a: float | None = None
    """Rated current of the house connection fuse, if known."""
    contracted_p_mw: float | None = None
    """Contracted connection capacity, if known."""

    def __post_init__(self) -> None:
        if not self.p_min_mw <= self.p_max_mw:
            raise ValueError(f"p_min_mw={self.p_min_mw} > p_max_mw={self.p_max_mw}")
        if self.s_max_mva is not None and self.s_max_mva < 0.0:
            raise ValueError("s_max_mva must not be negative")

    @property
    def p_bounds(self) -> Interval:
        """Power limits as an interval, for uncertainty sets."""
        return Interval(self.p_min_mw, self.p_max_mw)


@dataclass(frozen=True, slots=True)
class ExogenousInput:
    """Uncontrollable input quantities for one time step.

    Holds the realised values (for the simulation) **and** bounds (for a later
    robust feasibility check). Invariant I6 requires that no data path can
    construct an instance without ``bounds``, which is why the field is not
    optional.

    Args:
        t_index: Time step index relative to the scenario window.
        series_ids: Names of the time series, in the same order as
            ``realized_mw``.
        realized_mw: Realised active power, consumer reference direction.
        bounds_mw: Array of shape ``(2, n)`` with lower and upper bounds. In M1
            filled trivially from :class:`AssetRatings`.
        ambient_temp_degc: Ambient air temperature, input to the thermal model
            and the COP characteristic.
        ghi_wm2: Global horizontal irradiance, input to the PV potential model.
    """

    t_index: int
    series_ids: tuple[str, ...]
    realized_mw: np.ndarray
    bounds_mw: np.ndarray
    ambient_temp_degc: float
    ghi_wm2: float

    def __post_init__(self) -> None:
        n = len(self.series_ids)
        if self.realized_mw.shape != (n,):
            raise ValueError(
                f"realized_mw has shape {self.realized_mw.shape}, expected ({n},)"
            )
        if self.bounds_mw.shape != (2, n):
            raise ValueError(
                f"bounds_mw has shape {self.bounds_mw.shape}, expected (2, {n}). "
                "Invariant I6: exogenous inputs must carry bounds."
            )
        lo, hi = self.bounds_mw
        if np.any(lo > hi):
            raise ValueError("bounds_mw: lower bound above upper bound")
        # The realised value must lie inside the bounds; otherwise the
        # uncertainty set is parameterised incorrectly and any later safety
        # statement resting on it would be void.
        tol = 1e-9
        outside = (self.realized_mw < lo - tol) | (self.realized_mw > hi + tol)
        if np.any(outside):
            bad = [self.series_ids[i] for i in np.flatnonzero(outside)]
            raise ValueError(f"Realisation lies outside the bounds for: {bad}")
        object.__setattr__(self, "realized_mw", freeze_array(self.realized_mw))
        object.__setattr__(self, "bounds_mw", freeze_array(self.bounds_mw))

    def bound_of(self, series_id: str) -> Interval:
        """Bounds of a single time series."""
        i = self.series_ids.index(series_id)
        return Interval(float(self.bounds_mw[0, i]), float(self.bounds_mw[1, i]))


# ---------------------------------------------------------------------------
# Grid state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridState:
    """Result of a power flow calculation, reduced to what is needed.

    Deliberately no pandas objects: this type is created once per simulation
    step, and DataFrames are too expensive for that.

    ``converged=False`` is a valid state, not an error. The environment treats
    non-convergence as a defined event (reward penalty plus ``info`` flag), not
    as an exception.
    """

    t_index: int
    vm_pu: np.ndarray
    """Voltage magnitude per bus, indexed as in the pandapower grid."""
    line_loading_percent: np.ndarray
    trafo_loading_percent: np.ndarray
    p_slack_mw: float
    losses_mw: float
    converged: bool

    def __post_init__(self) -> None:
        for name in ("vm_pu", "line_loading_percent", "trafo_loading_percent"):
            object.__setattr__(self, name, freeze_array(getattr(self, name)))


# ---------------------------------------------------------------------------
# EN 50160 state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PQWindowState:
    """Running 10-minute averaging window of an evaluated bus.

    EN 50160 assesses 10-minute mean values, not instantaneous values. This
    state holds the partial sum of the open window; a criterion-relevant value
    only appears at the end of a window.
    """

    samples_count: int
    partial_sum_pu: float

    @property
    def mean_pu(self) -> float:
        """Mean over the part of the window filled so far."""
        if self.samples_count == 0:
            return float("nan")
        return self.partial_sum_pu / self.samples_count


@dataclass(frozen=True, slots=True)
class PQBudgetState:
    """Consumed violation budget of the running weekly interval.

    EN 50160 permits 5 % of the 10-minute mean values of a week to lie outside
    +/-10 % U_n; with 1008 windows per week that is 50 windows per bus. Without
    this state in the observation the control problem is not Markovian
    (section 6.6 of the architecture document), which is why it is part of the
    system state and not merely an evaluation quantity.
    """

    windows_elapsed_count: int
    """Completed 10-minute windows since the start of the week."""
    windows_total_count: int = 1008
    """Windows per weekly interval: 7 * 24 * 6."""
    violations_k95_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Per evaluated bus: number of windows outside +/-10 % U_n."""
    violations_k100_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Per evaluated bus: number of windows outside +10 % / -15 % U_n."""
    windows: tuple[PQWindowState, ...] = ()
    """Open averaging windows, per evaluated bus."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "violations_k95_count", freeze_array(self.violations_k95_count)
        )
        object.__setattr__(
            self, "violations_k100_count", freeze_array(self.violations_k100_count)
        )

    @property
    def budget_windows_count(self) -> int:
        """Permitted number of violating windows per bus and week (K95)."""
        return int(0.05 * self.windows_total_count)

    def budget_used_frac(self) -> np.ndarray:
        """Fraction of the K95 budget consumed per bus; may exceed 1."""
        return self.violations_k95_count / max(self.budget_windows_count, 1)

    def windows_remaining_count(self) -> int:
        """Remaining 10-minute windows in the running weekly interval."""
        return max(self.windows_total_count - self.windows_elapsed_count, 0)


# ---------------------------------------------------------------------------
# Asset state and overall state
# ---------------------------------------------------------------------------


@runtime_checkable
class AssetState(Protocol):
    """Marker protocol for asset states.

    Concrete implementations arrive in M4 (``lvgrid_rl.components``) and must be
    immutable, copyable value objects -- invariant I2. A state with methods that
    mutate ``self`` makes the later predictive safety filter impossible, because
    it has to roll asset states forward hypothetically over a horizon.
    """

    asset_id: str


@dataclass(frozen=True, slots=True)
class SystemState:
    """Complete system state at one point in time.

    **Invariant I1:** this type is the single source of truth. The agent's
    observation (from M3) is a *projection* of it and never itself a carrier of
    state. Only then can a later certifier receive more information than the
    agent -- it is a separate, better instrumented component (section 6.9).

    ``exogenous`` holds the realised values *at* ``t_index``. What may be known
    at decision time is governed by
    :class:`lvgrid_rl.core.information.InformationSet`; this type is the full
    state for simulation, certification and evaluation.
    """

    t_index: int
    timestamp: datetime
    """Timestamp in UTC. Calendar features are derived from it."""
    topology_id: str
    """Identifier of the switching topology. A safety statement holds per topology."""
    grid: GridState
    assets: Mapping[str, AssetState]
    exogenous: ExogenousInput
    pq: PQBudgetState

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware (UTC). Naive timestamps cause "
                "silent errors at daylight saving transitions."
            )
