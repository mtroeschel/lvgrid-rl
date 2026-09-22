"""Tests of the cross-seed aggregation (M3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate import (  # noqa: E402
    MIN_SEEDS_FOR_CI,
    aggregate_metric,
    bootstrap_ci,
    cap_curve_overload,
    iqm,
    load_policy_rows,
)

# ---------------------------------------------------------------------------
# Interquartile mean
# ---------------------------------------------------------------------------


def test_iqm_ignores_the_outer_quartiles() -> None:
    """The outer quartiles are trimmed before averaging.

    One unlucky seed must not move the figure more than the effect being
    measured.
    """
    clean = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
    with_outlier = [*clean[:-1], 1000.0]
    assert iqm(clean) == pytest.approx(iqm(with_outlier))
    assert float(np.mean(with_outlier)) > 2 * float(np.mean(clean))


def test_iqm_falls_back_to_the_mean_for_tiny_samples() -> None:
    """Tiny samples fall back to the plain mean.

    With three seeds there is no robustness to be had, and pretending otherwise
    would be worse than saying so.
    """
    assert iqm([1.0, 2.0, 6.0]) == pytest.approx(3.0)


def test_iqm_matches_the_mean_when_the_sample_is_symmetric() -> None:
    assert iqm([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------


def test_bootstrap_interval_is_reproducible() -> None:
    """An interval that moves between report runs cannot be checked."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    values = [3.0, 5.0, 8.0, 9.0, 12.0, 14.0]
    low, high = bootstrap_ci(values, seed=0)
    assert low <= iqm(values) <= high


def test_wider_spread_gives_a_wider_interval() -> None:
    narrow = bootstrap_ci([9.0, 10.0, 10.0, 11.0, 10.0, 10.0], seed=0)
    wide = bootstrap_ci([1.0, 10.0, 30.0, 2.0, 25.0, 40.0], seed=0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_single_value_yields_no_interval() -> None:
    low, high = bootstrap_ci([5.0])
    assert np.isnan(low) and np.isnan(high)


def test_few_seeds_are_flagged() -> None:
    """Section 9.3 asks for at least five seeds, better ten."""
    assert aggregate_metric("m", [1.0, 2.0, 3.0]).indicative_only
    assert not aggregate_metric("m", [1.0, 2.0, 3.0, 4.0, 5.0]).indicative_only
    assert MIN_SEEDS_FOR_CI == 5


# ---------------------------------------------------------------------------
# Cap curve comparison
# ---------------------------------------------------------------------------


def _pareto_payload() -> dict:
    return {
        "points": [
            {"kind": "fixed_cap", "curtailed_mwh": 0.0, "overload_cost": 360.0},
            {"kind": "fixed_cap", "curtailed_mwh": 10.0, "overload_cost": 200.0},
            {"kind": "fixed_cap", "curtailed_mwh": 20.0, "overload_cost": 40.0},
            {"kind": "policy", "curtailed_mwh": 15.0, "overload_cost": 30.0},
        ]
    }


def test_cap_curve_is_interpolated_between_measured_points() -> None:
    assert cap_curve_overload(_pareto_payload(), 15.0) == pytest.approx(120.0)
    assert cap_curve_overload(_pareto_payload(), 10.0) == pytest.approx(200.0)


def test_cap_curve_refuses_to_extrapolate() -> None:
    """The curve refuses to extrapolate.

    Beyond the sweep it is unknown, and an extrapolated comparison would look
    like a measurement.
    """
    assert cap_curve_overload(_pareto_payload(), 30.0) is None
    assert cap_curve_overload(_pareto_payload(), -1.0) is None


def test_only_cap_points_define_the_curve() -> None:
    """A policy point must not be mistaken for part of the reference curve."""
    payload = _pareto_payload()
    without_policy = {"points": [p for p in payload["points"] if p["kind"] != "policy"]}
    assert cap_curve_overload(payload, 15.0) == cap_curve_overload(without_policy, 15.0)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def test_results_from_another_set_are_rejected(tmp_path) -> None:
    path = tmp_path / "val.json"
    path.write_text(json.dumps({"set": "val", "results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="not 'test'"):
        load_policy_rows([path], "test")


def test_only_policy_rows_are_collected(tmp_path) -> None:
    run = tmp_path / "run" / "eval"
    run.mkdir(parents=True)
    path = run / "test.json"
    path.write_text(
        json.dumps(
            {
                "set": "test",
                "results": [
                    {"controller": "policy", "pass_rate": 1.0},
                    {"controller": "do_nothing", "pass_rate": 0.9},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = load_policy_rows([path], "test")
    assert len(rows) == 1
    assert rows[0]["run"] == "run"
