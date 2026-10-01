"""Charging sessions from SimBench blocks and a connection-time distribution (D16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lvgrid_rl.data.sources.ev_sessions import (
    DwellDistribution,
    assign_departures,
    extract_sessions,
    read_dwell_csv,
    synthetic_dwell,
)

STEP_H = 5 / 60


def _profile(blocks, n=288, start="2016-01-04 00:00"):
    """A 5-minute curve with constant-power blocks ``(first_step, n_steps, kw)``."""
    values = np.zeros(n)
    for first, length, kw in blocks:
        values[first : first + length] = kw / 1000
    index = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.Series(values, index=index)


def _fixed(hours: float) -> DwellDistribution:
    """Every arrival stays ``hours`` (the upper edge of a single narrow bin)."""
    return DwellDistribution(
        duration_bins_h=(hours - 1e-6, hours),
        probabilities=np.tile([0.0, 1.0], (24, 1)),
        source="test",
    )


def test_each_block_is_one_session_with_its_energy() -> None:
    start, end, energy = extract_sessions(_profile([(10, 12, 3.7), (100, 6, 11.0)]))
    assert list(start) == [10, 100]
    assert list(end) == [22, 106]
    assert energy == pytest.approx([3.7e-3 * 1.0, 11e-3 * 0.5])


def test_blocks_close_together_are_one_interrupted_session() -> None:
    profile = _profile([(10, 6, 3.7), (19, 6, 3.7)])  # 15-minute gap
    start, end, energy = extract_sessions(profile, merge_gap_min=15)
    assert list(start) == [10] and list(end) == [25]
    apart = extract_sessions(profile, merge_gap_min=10)
    assert len(apart[0]) == 2


def test_a_feeding_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="must not feed in"):
        extract_sessions(-_profile([(10, 6, 3.7)]))


def test_the_departure_follows_the_drawn_connection_time() -> None:
    profile = _profile([(12, 6, 3.7)])  # arrives 01:00 UTC
    table = assign_departures(profile, 0.0037, _fixed(4.0), np.random.default_rng(0))
    assert table.departure[0] - table.arrival[0] == round(4.0 / STEP_H)
    assert table.raised_to_feasible == 0 and table.capped_at_next == 0


def test_a_departure_too_early_for_the_energy_is_moved_later_and_counted() -> None:
    """22 kW in SimBench's curve, 3.7 kW nominal: the energy needs longer."""
    profile = _profile([(12, 12, 22.0)])  # 22 kWh in one hour
    table = assign_departures(profile, 0.0037, _fixed(1.0), np.random.default_rng(0))
    needed = int(np.ceil(0.022 / 0.0037 / STEP_H))
    assert table.departure[0] - table.arrival[0] == needed
    assert table.raised_to_feasible == 1


def test_a_departure_after_the_next_arrival_is_capped_and_counted() -> None:
    profile = _profile([(12, 6, 3.7), (60, 6, 3.7)])
    table = assign_departures(profile, 0.0037, _fixed(10.0), np.random.default_rng(0))
    assert table.departure[0] == table.arrival[1]
    assert table.capped_at_next == 1


def test_every_session_can_deliver_its_energy() -> None:
    profile = _profile([(10, 30, 11.0), (50, 20, 3.7), (200, 10, 22.0)])
    table = assign_departures(profile, 0.011, synthetic_dwell(), np.random.default_rng(3))
    window_h = (table.departure - table.arrival) * STEP_H
    assert np.all(window_h * table.p_max_mw >= table.energy_mwh - 1e-12)


def test_the_draw_depends_on_the_hour_of_arrival() -> None:
    dwell = synthetic_dwell()
    rng = np.random.default_rng(0)
    evening = np.mean([dwell.draw_h(19, rng) for _ in range(2000)])
    noon = np.mean([dwell.draw_h(12, rng) for _ in range(2000)])
    assert evening > 2 * noon


def test_sessions_are_reproducible_for_a_seed() -> None:
    profile = _profile([(10, 6, 3.7), (150, 6, 3.7)])
    a = assign_departures(profile, 0.0037, synthetic_dwell(), np.random.default_rng(7))
    b = assign_departures(profile, 0.0037, synthetic_dwell(), np.random.default_rng(7))
    assert np.array_equal(a.departure, b.departure)


def test_the_synthetic_distribution_says_what_it_is() -> None:
    assert synthetic_dwell().source == "synthetic"


@pytest.mark.parametrize(
    ("bins", "probs", "match"),
    [
        ((1.0, 0.5), np.tile([0.5, 0.5], (24, 1)), "increasing"),
        ((1.0, 2.0), np.tile([0.5, 0.5], (23, 1)), "shape"),
        ((1.0, 2.0), np.tile([0.6, 0.6], (24, 1)), "sums to one"),
    ],
)
def test_inconsistent_distributions_are_refused(bins, probs, match) -> None:
    with pytest.raises(ValueError, match=match):
        DwellDistribution(duration_bins_h=bins, probabilities=probs, source="test")


def test_the_intermediate_csv_is_read_and_checked(tmp_path) -> None:
    rows = [
        {"arrival_hour": h, "duration_h": d, "probability": p}
        for h in range(24)
        for d, p in ((2.0, 0.3), (8.0, 0.7))
    ]
    path = tmp_path / "dwell.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    dwell = read_dwell_csv(path, source="elaadnl-test")
    assert dwell.duration_bins_h == (2.0, 8.0)
    assert dwell.source == "elaadnl-test"
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="every duration bin"):
        read_dwell_csv(path, source="x")
