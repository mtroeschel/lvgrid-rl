"""Information ordering: what may be known at decision time.

This implements invariant I3 (section 13 of the architecture document) and is at
the same time the most unpleasant of the seven invariants, because a violation
stays invisible: a controller that accidentally reads the realisation of the
coming interval learns beautifully and is worthless in operation -- and the bug
is spread diffusely across the codebase.

The module provides two things:

* :class:`InformationSet` -- the slice of the state that action construction may
  work on. Structured such that realised future values do not fit into it.
* :class:`DecisionScope` -- a context manager that blocks access to time steps
  beyond the decision point at runtime. This makes the invariant testable rather
  than merely documented.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from lvgrid_rl.core.schemas import AssetState, Interval, PQBudgetState, freeze_array

__all__ = [
    "InformationSet",
    "DecisionScope",
    "ClairvoyanceError",
    "current_decision_horizon",
    "assert_readable",
    "unrestricted",
]


class ClairvoyanceError(RuntimeError):
    """Access to information that is not available at decision time.

    See :func:`assert_readable` and :class:`DecisionScope`.
    """


# The decision horizon is thread-local so that parallel environments in
# ``SubprocVecEnv``/``DummyVecEnv`` do not interfere with each other.
_local = threading.local()


def current_decision_horizon() -> int | None:
    """Currently valid decision point.

    Returns ``None`` when no :class:`DecisionScope` is active.
    """
    return getattr(_local, "horizon", None)


def assert_readable(t_index: int, what: str = "time series value") -> None:
    """Check whether ``t_index`` may be read at the current decision point.

    Called by the data accessors of the scenario layer. Outside a
    :class:`DecisionScope` every access is permitted and the function returns
    immediately -- simulation, evaluation and certification may see the full
    state, and the hot path should cost nothing.

    Inside a scope the access is additionally recorded, so that tests can check
    *which* time steps were read, not merely that no exception was raised.

    Raises:
        ClairvoyanceError: if a later time step is accessed inside a decision
            window.
    """
    horizon = current_decision_horizon()
    if horizon is None:
        return
    recorder = getattr(_local, "recorder", None)
    if recorder is not None:
        recorder.append(t_index)
    if t_index > horizon:
        raise ClairvoyanceError(
            f"{what} requested for t={t_index}, at most t={horizon} is allowed. "
            "Invariant I3: action construction may only use forecasts, never the "
            "realisation of the coming interval."
        )


class DecisionScope:
    """Context manager that enforces the information ordering at runtime.

    In the environment it wraps steps 1 and 2 of the step sequence
    (section 6.1): action mapping and setpoint construction. Inside the block,
    any access to a time step ``> t_decision`` raises
    :class:`ClairvoyanceError`.

    Accesses are also recorded, so that tests can assert *which* time steps were
    read.

    Example:
        >>> scope = DecisionScope(t_decision=10)
        >>> with scope:
        ...     assert_readable(10)
        ...     assert_readable(8)
        >>> scope.accessed
        (10, 8)
    """

    __slots__ = ("t_decision", "_accessed", "_previous")

    def __init__(self, t_decision: int) -> None:
        self.t_decision = t_decision
        self._accessed: list[int] = []
        self._previous: int | None = None

    def __enter__(self) -> DecisionScope:
        self._previous = current_decision_horizon()
        _local.horizon = self.t_decision
        _local.recorder = self._accessed
        return self

    def __exit__(self, *exc_info: object) -> None:
        _local.horizon = self._previous
        _local.recorder = None

    @property
    def accessed(self) -> tuple[int, ...]:
        """All time steps requested inside the block."""
        return tuple(self._accessed)


@contextmanager
def unrestricted() -> Iterator[None]:
    """Temporarily lift the information ordering.

    Only for components that legitimately need the full state: the simulation
    itself, reference methods with perfect foresight (``mpc_oracle``) and
    evaluation. Any use on an agent path is a bug and should stand out in
    review -- hence the deliberately conspicuous name.
    """
    previous = current_decision_horizon()
    _local.horizon = None
    try:
        yield
    finally:
        _local.horizon = previous


@dataclass(frozen=True, slots=True)
class InformationSet:
    """What the controller may know at decision point ``t_index``.

    Structured so that clairvoyance is not expressible: there is no field for
    realised values of the coming interval. Instead of the realisation there are
    bounds (``exogenous_bounds_mw``) and forecasts (``forecast``).

    The difference from the agent's observation: the ``InformationSet`` is the
    *maximum* of what may legitimately be known. The ``ObservationBuilder`` (M3)
    selects from it according to ``sensor_config`` and may pass on considerably
    less. A later certifier, by contrast, may use the full ``InformationSet``.

    Args:
        t_index: Decision point.
        timestamp: Timestamp in UTC.
        measurements: Measured values according to the configured sensor set,
            keyed as ``"vm_pu/bus_17"`` or ``"trafo_loading_percent/0"``.
        asset_states: Internal states of the own assets.
        series_ids: Names for the columns of ``exogenous_bounds_mw``.
        exogenous_bounds_mw: Bounds on the uncontrollable injections during
            ``[t, t + control_dt)``, shape ``(2, n)``.
        forecast: Forecasts per quantity, each an array of length ``horizon``.
            Produced by the forecast error model (section 6.7); in mode
            ``perfect`` it contains the realisation, and that is then a declared
            special case rather than a leak.
        pq: Consumed EN 50160 budget. Without this field the control problem is
            not Markovian.
    """

    t_index: int
    timestamp: datetime
    measurements: Mapping[str, float]
    asset_states: Mapping[str, AssetState]
    series_ids: tuple[str, ...]
    exogenous_bounds_mw: np.ndarray
    forecast: Mapping[str, np.ndarray]
    pq: PQBudgetState

    def __post_init__(self) -> None:
        n = len(self.series_ids)
        if self.exogenous_bounds_mw.shape != (2, n):
            raise ValueError(
                f"exogenous_bounds_mw has shape {self.exogenous_bounds_mw.shape}, "
                f"expected (2, {n})"
            )
        object.__setattr__(
            self, "exogenous_bounds_mw", freeze_array(self.exogenous_bounds_mw)
        )

    def bound_of(self, series_id: str) -> Interval:
        """Bounds of a single time series over the coming interval."""
        i = self.series_ids.index(series_id)
        return Interval(
            float(self.exogenous_bounds_mw[0, i]),
            float(self.exogenous_bounds_mw[1, i]),
        )
