"""Tests for the EN 50160 assessment (M2)."""

from __future__ import annotations

import numpy as np
import pytest

from lvgrid_rl.grid.metrics import FallbackReport, violation_metrics
from lvgrid_rl.grid.pq import (
    K95_BAND_PU,
    K100_BAND_PU,
    WINDOWS_PER_WEEK,
    EN50160Evaluator,
    PQAggregator,
    WeekResult,
)


def _state(vm: np.ndarray, line=None, trafo=None, converged: bool = True):
    from lvgrid_rl.core.schemas import GridState

    return GridState(
        t_index=0,
        vm_pu=np.asarray(vm, dtype=float),
        line_loading_percent=np.asarray([50.0] if line is None else line, dtype=float),
        trafo_loading_percent=np.asarray([60.0] if trafo is None else trafo, dtype=float),
        p_slack_mw=0.01,
        losses_mw=0.001,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------


def test_window_closes_only_after_the_full_ten_minutes() -> None:
    """Criterion-relevant events occur at window boundaries, not every step."""
    agg = PQAggregator(n_buses=2, samples_per_window=2)
    assert agg.add_sample(np.array([1.0, 1.0])) is None
    mean = agg.add_sample(np.array([1.10, 1.00]))
    assert mean is not None
    assert mean == pytest.approx([1.05, 1.00])


def test_averaging_can_hide_an_excursion() -> None:
    """The heart of the criterion: EN 50160 assesses ten-minute means.

    An instantaneous value of 1.14 pu paired with 1.04 pu averages to 1.09 and
    does not count as a violation. A controller assessed against instantaneous
    limits would be strictly more conservative than the standard requires.
    """
    agg = PQAggregator(n_buses=1, samples_per_window=2)
    agg.add_sample(np.array([1.14]))
    agg.add_sample(np.array([1.04]))
    assert agg.state().violations_k95_count[0] == 0


def test_budget_is_fifty_windows_per_bus_and_week() -> None:
    agg = PQAggregator(n_buses=1, samples_per_window=1)
    assert WINDOWS_PER_WEEK == 7 * 24 * 6 == 1008
    assert agg.budget_windows == 50


def test_violations_are_counted_per_band() -> None:
    """K100 is asymmetric: -15 % but only +10 %."""
    agg = PQAggregator(n_buses=3, samples_per_window=1)
    agg.add_sample(np.array([1.12, 0.88, 1.00]))
    state = agg.state()
    # 1.12 is outside both bands, 0.88 outside K95 only, 1.00 inside both.
    assert list(state.violations_k95_count) == [1, 1, 0]
    assert list(state.violations_k100_count) == [1, 0, 0]


def test_reset_week_can_preload_a_budget() -> None:
    """A week can start with a pre-consumed budget.

    Used by the episode sampler so an agent sees all budget regimes without
    week-long episodes.
    """
    agg = PQAggregator(n_buses=2, samples_per_window=1)
    agg.reset_week(k95=np.array([40, 0]))
    assert list(agg.state().violations_k95_count) == [40, 0]


def test_samples_per_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PQAggregator(n_buses=1, samples_per_window=0)


# ---------------------------------------------------------------------------
# Weekly assessment
# ---------------------------------------------------------------------------


def _week(k95: list[int], k100: list[int]) -> WeekResult:
    n = len(k95)
    return WeekResult(
        week_index=0,
        windows=WINDOWS_PER_WEEK,
        k95_violations=np.array(k95),
        k100_violations=np.array(k100),
        p95_vm_pu=np.full(n, 1.05),
    )


def test_a_week_passes_while_inside_the_budget() -> None:
    assert _week([50, 0], [0, 0]).passed()
    assert not _week([51, 0], [0, 0]).passed()


def test_a_single_k100_violation_fails_the_week() -> None:
    """K100 admits no budget at all."""
    assert not _week([0, 0], [1, 0]).passed()


def test_the_worst_bus_decides() -> None:
    week = _week([0, 60], [0, 0])
    assert list(week.passed_per_bus()) == [True, False]
    assert not week.passed()


def test_budget_utilisation_exposes_near_misses() -> None:
    """Budget utilisation exposes near misses.

    0.98 and 1.02 are night and day in the pass rate but operationally almost
    identical, so both figures are reported.
    """
    week = _week([49, 51], [0, 0])
    assert week.budget_utilisation() == pytest.approx([0.98, 1.02])


def test_evaluator_reports_pass_rate_over_bus_week_pairs() -> None:
    evaluator = EN50160Evaluator(n_buses=2, samples_per_window=1)
    for _ in range(WINDOWS_PER_WEEK):
        evaluator.add_sample(np.array([1.00, 1.20]))
    assert len(evaluator.weeks) == 1
    # Bus 0 compliant, bus 1 violating in every window.
    assert evaluator.pass_rate() == pytest.approx(0.5)
    assert evaluator.worst_bus_p95() == pytest.approx(1.20)


def test_partial_week_is_flagged_not_silently_counted() -> None:
    evaluator = EN50160Evaluator(n_buses=1, samples_per_window=1)
    for _ in range(100):
        evaluator.add_sample(np.array([1.0]))
    evaluator.finalize()
    assert len(evaluator.weeks) == 1
    assert evaluator.weeks[0].complete is False
    assert np.isnan(evaluator.pass_rate(complete_only=True))


# ---------------------------------------------------------------------------
# Instantaneous metrics
# ---------------------------------------------------------------------------


def test_violation_metrics_sum_the_excursion_depth() -> None:
    positions = np.array([0, 1])
    m = violation_metrics(_state([1.12, 0.88]), positions)
    assert m.n_buses_outside_k95 == 2
    assert m.n_buses_outside_k100 == 1  # only 1.12 leaves the wider band
    assert m.voltage_excursion_pu == pytest.approx(0.02 + 0.02)


def test_overload_excess_counts_lines_and_transformers() -> None:
    m = violation_metrics(_state([1.0], line=[110.0, 90.0], trafo=[130.0]), np.array([0]))
    assert m.overload_excess_percent == pytest.approx(10.0 + 30.0)
    assert m.max_trafo_loading_percent == pytest.approx(130.0)


def test_non_convergence_yields_nan_metrics_not_an_exception() -> None:
    m = violation_metrics(_state([1.0], converged=False), np.array([0]))
    assert m.converged is False
    assert np.isnan(m.max_vm_pu)


def test_bands_match_the_standard() -> None:
    assert K95_BAND_PU == (0.90, 1.10)
    assert K100_BAND_PU == (0.85, 1.10)


# ---------------------------------------------------------------------------
# P4 report
# ---------------------------------------------------------------------------


def test_fallback_report_is_feasible_only_without_violations() -> None:
    ok = FallbackReport(100, 0, (0.99, 1.02), 80.0, None)
    assert ok.feasible and "feasible" in ok.summary()
    bad = FallbackReport(100, 3, (0.88, 1.02), 120.0, 17)
    assert not bad.feasible and "INFEASIBLE" in bad.summary()
