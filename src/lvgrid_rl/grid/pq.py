"""EN 50160 assessment: ten-minute averaging and the weekly percentile criterion.

The voltage criterion of this project is not an instantaneous limit. EN 50160
assesses **ten-minute mean values** of the RMS voltage over a **weekly
interval**, with two nested conditions:

* **K95** -- 95 % of the ten-minute means of each week within +/-10 % U_n,
* **K100** -- all ten-minute means within +10 % / -15 % U_n (low voltage).

With 1008 windows per week, K95 permits up to **50 violating windows per bus and
week**. That budget is the reason the control problem is not Markovian without
extra state: whether an excursion "counts" depends on the distribution of the
whole week, so the agent has to know how much tolerance is left (section 6.6 of
the architecture document).

Two consequences are implemented here. :class:`PQAggregator` forms the windows
and maintains the budget; :class:`EN50160Evaluator` turns completed weeks into
the reported KPI. Everything is assessed per connection point, never per
arbitrary bus -- EN 50160 applies at the point of common coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lvgrid_rl.core.schemas import PQBudgetState, PQWindowState

__all__ = [
    "K95_BAND_PU",
    "K100_BAND_PU",
    "WINDOWS_PER_WEEK",
    "K95_BUDGET_FRACTION",
    "PQAggregator",
    "WeekResult",
    "EN50160Evaluator",
]

K95_BAND_PU: tuple[float, float] = (0.90, 1.10)
"""Band for the 95 % criterion, symmetric."""

K100_BAND_PU: tuple[float, float] = (0.85, 1.10)
"""Band that all windows must respect. Asymmetric: -15 % but only +10 %."""

WINDOWS_PER_WEEK: int = 1008
"""7 * 24 * 6 ten-minute windows."""

K95_BUDGET_FRACTION: float = 0.05
"""Share of windows per week that may leave the K95 band."""


class PQAggregator:
    """Forms ten-minute mean values and tracks the weekly violation budget.

    Args:
        n_buses: Number of assessed connection points.
        samples_per_window: Simulation steps per ten-minute window. Follows from
            ``sim_dt``; at 5 min it is 2, at 1 min it is 10. The restriction of
            ``sim_dt`` to divisors of ten minutes exists precisely so that this
            number is an integer.

    The aggregator holds only the running partial sum, not the samples
    themselves. Over a year at one-minute resolution that is the difference
    between a few hundred bytes and several hundred megabytes.
    """

    def __init__(self, n_buses: int, samples_per_window: int) -> None:
        if samples_per_window < 1:
            raise ValueError("samples_per_window must be at least 1")
        self.n_buses = n_buses
        self.samples_per_window = samples_per_window
        self._partial = np.zeros(n_buses, dtype=float)
        self._count = 0
        self._k95 = np.zeros(n_buses, dtype=np.int32)
        self._k100 = np.zeros(n_buses, dtype=np.int32)
        self._windows_elapsed = 0

    def reset_week(self, k95: np.ndarray | None = None) -> None:
        """Start a new weekly interval.

        Args:
            k95: Optional pre-consumed budget per bus. Used by the episode
                sampler to randomise the initial budget, so that an agent sees
                all budget regimes without episodes having to span a whole week
                (section 6.5).
        """
        self._k95 = (
            np.zeros(self.n_buses, dtype=np.int32)
            if k95 is None
            else np.asarray(k95, dtype=np.int32).copy()
        )
        self._k100 = np.zeros(self.n_buses, dtype=np.int32)
        self._windows_elapsed = 0
        self._partial[:] = 0.0
        self._count = 0

    def add_sample(self, vm_pu: np.ndarray) -> np.ndarray | None:
        """Add one simulation step.

        Args:
            vm_pu: Voltage magnitudes at the assessed buses.

        Returns:
            The completed window's mean values when the window closes with this
            sample, otherwise ``None``. Criterion-relevant events only occur at
            window boundaries, not at every step.
        """
        self._partial += vm_pu
        self._count += 1
        if self._count < self.samples_per_window:
            return None

        mean = self._partial / self._count
        self._partial[:] = 0.0
        self._count = 0
        self._windows_elapsed += 1

        lo95, hi95 = K95_BAND_PU
        lo100, hi100 = K100_BAND_PU
        self._k95 += ((mean < lo95) | (mean > hi95)).astype(np.int32)
        self._k100 += ((mean < lo100) | (mean > hi100)).astype(np.int32)
        return mean

    @property
    def budget_windows(self) -> int:
        """Permitted violating windows per bus and week (50 at 1008 windows)."""
        return int(K95_BUDGET_FRACTION * WINDOWS_PER_WEEK)

    def state(self) -> PQBudgetState:
        """Current budget state, for observation and system state."""
        return PQBudgetState(
            windows_elapsed_count=self._windows_elapsed,
            windows_total_count=WINDOWS_PER_WEEK,
            violations_k95_count=self._k95.copy(),
            violations_k100_count=self._k100.copy(),
            windows=(PQWindowState(self._count, float(self._partial.sum())),),
        )

    def week_complete(self) -> bool:
        """Has a full weekly interval elapsed?"""
        return self._windows_elapsed >= WINDOWS_PER_WEEK


@dataclass(frozen=True, slots=True)
class WeekResult:
    """Assessment of one completed weekly interval.

    Args:
        week_index: Running index of the weekly interval.
        windows: Number of windows assessed. Below
            :data:`WINDOWS_PER_WEEK` the week is partial and ``passed`` is
            reported but flagged via :attr:`complete`.
        k95_violations: Violating windows per bus.
        k100_violations: Windows outside the absolute band, per bus.
        p95_vm_pu: 95th percentile of the ten-minute means per bus.
        complete: Whether the week was assessed in full.
    """

    week_index: int
    windows: int
    k95_violations: np.ndarray
    k100_violations: np.ndarray
    p95_vm_pu: np.ndarray
    complete: bool = True

    @property
    def budget_windows(self) -> int:
        """Permitted violating windows per bus."""
        return int(K95_BUDGET_FRACTION * WINDOWS_PER_WEEK)

    def passed_per_bus(self) -> np.ndarray:
        """Per bus: does it satisfy K95 *and* K100?"""
        return (self.k95_violations <= self.budget_windows) & (self.k100_violations == 0)

    def passed(self) -> bool:
        """Does every assessed bus pass? The worst bus decides."""
        return bool(self.passed_per_bus().all())

    def budget_utilisation(self) -> np.ndarray:
        """Share of the K95 budget consumed per bus; may exceed 1.

        Reported alongside the binary pass rate on purpose: 0.98 and 1.02 are
        night and day in the pass rate but operationally almost identical, and a
        binary KPI alone hides every improvement below the threshold.
        """
        return self.k95_violations / self.budget_windows


@dataclass
class EN50160Evaluator:
    """Collects ten-minute means and assesses complete weekly intervals.

    Args:
        n_buses: Number of assessed connection points.
        samples_per_window: Simulation steps per ten-minute window.

    The evaluator keeps the ten-minute means of the running week, because the
    95th percentile cannot be computed incrementally. That is 1008 values per
    bus and week -- negligible next to the raw time series.
    """

    n_buses: int
    samples_per_window: int
    _aggregator: PQAggregator = field(init=False)
    _means: list[np.ndarray] = field(init=False, default_factory=list)
    _weeks: list[WeekResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._aggregator = PQAggregator(self.n_buses, self.samples_per_window)

    @property
    def aggregator(self) -> PQAggregator:
        """The underlying window aggregator, for observation building."""
        return self._aggregator

    def add_sample(self, vm_pu: np.ndarray) -> None:
        """Add one simulation step and close the week when it is full."""
        mean = self._aggregator.add_sample(vm_pu)
        if mean is None:
            return
        self._means.append(mean)
        if self._aggregator.week_complete():
            self._close_week()

    def _close_week(self, complete: bool = True) -> None:
        if not self._means:
            return
        stacked = np.vstack(self._means)
        state = self._aggregator.state()
        self._weeks.append(
            WeekResult(
                week_index=len(self._weeks),
                windows=len(self._means),
                k95_violations=np.asarray(state.violations_k95_count).copy(),
                k100_violations=np.asarray(state.violations_k100_count).copy(),
                p95_vm_pu=np.percentile(stacked, 95, axis=0),
                complete=complete,
            )
        )
        self._means.clear()
        self._aggregator.reset_week()

    def finalize(self) -> None:
        """Close a partially elapsed week at the end of a simulation run.

        A trailing partial week is reported, but flagged via
        :attr:`WeekResult.complete` so it cannot be mistaken for a
        standard-conforming assessment.
        """
        self._close_week(complete=False)

    @property
    def weeks(self) -> tuple[WeekResult, ...]:
        """All assessed weekly intervals."""
        return tuple(self._weeks)

    def pass_rate(self, complete_only: bool = True) -> float:
        """Share of (bus, week) pairs satisfying K95 and K100.

        This is the headline KPI. It is deliberately computed per bus and week
        rather than per week alone, because a grid in which one feeder end fails
        every week is not 'mostly compliant'.
        """
        weeks = [w for w in self._weeks if w.complete or not complete_only]
        if not weeks:
            return float("nan")
        passed = sum(int(w.passed_per_bus().sum()) for w in weeks)
        total = sum(len(w.passed_per_bus()) for w in weeks)
        return passed / total if total else float("nan")

    def worst_bus_p95(self) -> float:
        """Highest 95th percentile across all buses and weeks.

        The continuous companion to :meth:`pass_rate`: without it, improvements
        that do not yet cross the threshold are invisible.
        """
        weeks = [w for w in self._weeks if w.complete]
        if not weeks:
            return float("nan")
        return float(max(float(np.nanmax(w.p95_vm_pu)) for w in weeks))
