"""Tests of the EV charge point model (M4, step 4.4b, D17)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.components.ev_charger import EvCharger, EvChargerState
from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import FlexAsset, Setpoint
from lvgrid_rl.core.schemas import (
    AssetRatings,
    EvSession,
    ExogenousInput,
    GridState,
    PQBudgetState,
)

GRID = GridState(
    t_index=0,
    vm_pu=np.full(2, 1.0),
    line_loading_percent=np.array([30.0]),
    trafo_loading_percent=np.array([40.0]),
    p_slack_mw=0.0,
    losses_mw=0.0,
    converged=True,
)
INFO = InformationSet(
    t_index=0,
    timestamp=datetime(2016, 1, 15, 18, 0, tzinfo=UTC),
    measurements={},
    asset_states={},
    series_ids=(),
    exogenous_bounds_mw=np.zeros((2, 0)),
    forecast={},
    pq=PQBudgetState(windows_elapsed_count=0),
)
DT = 5.0


def _ev(p_max_mw=0.011) -> EvCharger:
    return EvCharger(
        asset_id="load:15", bus=4, ratings=AssetRatings(p_min_mw=0.0, p_max_mw=p_max_mw)
    )


def _x(t, session=None) -> ExogenousInput:
    return ExogenousInput(
        t_index=t,
        series_ids=(),
        realized_mw=np.zeros(0),
        bounds_mw=np.zeros((2, 0)),
        ambient_temp_degc=0.0,
        ghi_wm2=0.0,
        ev_sessions={"load:15": session} if session is not None else {},
    )


def _run(ev, session, offered_mw, t0, t1, state=None):
    """Steps ``t0 .. t1 - 1`` at a constant offer; returns state and outcomes."""
    state = state or ev.initial_state(np.random.default_rng(0))
    sp = ev.to_setpoint(state, np.array([offered_mw]), INFO, 15)
    outcomes, drawn = [], []
    for t in range(t0, t1):
        under_way = session.arrival_t_index <= t < session.departure_t_index
        live = session if under_way else None
        x = _x(t, live)
        applied = ev.limit_to_physics(state, sp, x, DT)
        state, outcome = ev.dynamics(state, applied, x, GRID, DT)
        outcomes.append(outcome)
        drawn.append(applied.p_mw)
    return state, outcomes, drawn


def test_it_satisfies_the_asset_protocol() -> None:
    assert isinstance(_ev(), FlexAsset)
    assert _ev().kind == "ev"


def test_uncontrolled_charging_is_the_neutral_action() -> None:
    """Full rating offered: emobpy's immediate strategy, which the data is."""
    spec = _ev().action_spec()
    assert spec.neutral == (0.011,)
    assert spec.bounds[0].lo == 0.0


def test_only_a_request_beyond_the_rating_is_clipped() -> None:
    ev = _ev()
    s = ev.initial_state(np.random.default_rng(0))
    assert ev.to_setpoint(s, np.array([0.02]), INFO, 15).clipping_info == {
        "p_mw": pytest.approx(-0.009)
    }
    assert ev.to_setpoint(s, np.array([-0.001]), INFO, 15).p_mw == 0.0
    assert not ev.to_setpoint(s, np.array([0.005]), INFO, 15).was_clipped


def test_without_a_vehicle_nothing_is_drawn_and_nothing_clipped() -> None:
    ev = _ev()
    s = ev.initial_state(np.random.default_rng(0))
    sp = ev.to_setpoint(s, np.array([0.011]), INFO, 15)
    applied = ev.limit_to_physics(s, sp, _x(0), DT)
    assert applied.p_mw == 0.0 and not applied.was_clipped
    state, outcome = ev.dynamics(s, applied, _x(0), GRID, DT)
    assert not state.connected and outcome.unserved_energy_mwh == 0.0


def test_a_vehicle_is_charged_until_it_needs_nothing_more() -> None:
    """2 kWh at 11 kW: about 11 minutes, the third step only partly."""
    ev = _ev()
    session = EvSession(arrival_t_index=0, departure_t_index=12, energy_mwh=0.002)
    state, outcomes, drawn = _run(ev, session, 0.011, 0, 5)
    hours = DT / 60
    assert drawn[:2] == pytest.approx([0.011, 0.011])
    assert drawn[2] == pytest.approx((0.002 - 2 * 0.011 * hours) / hours)
    assert drawn[3:] == [0.0, 0.0]
    assert sum(o.throughput_energy_mwh for o in outcomes) == pytest.approx(0.002)
    assert state.connected and state.need_mwh == pytest.approx(0.0)
    assert state.remaining_min == pytest.approx(7 * DT)


def test_what_is_missing_at_departure_is_unserved() -> None:
    """3 kWh offered at 3.7 kW for 30 minutes: 1.15 kWh short."""
    ev = _ev()
    session = EvSession(arrival_t_index=2, departure_t_index=8, energy_mwh=0.003)
    state, outcomes, _ = _run(ev, session, 0.0037, 0, 10)
    charged = sum(o.throughput_energy_mwh for o in outcomes)
    unserved = sum(o.unserved_energy_mwh for o in outcomes)
    assert charged == pytest.approx(0.0037 * 0.5)
    assert unserved == pytest.approx(0.003 - charged)
    # Booked once, in the last step of the session.
    assert [o.unserved_energy_mwh > 0 for o in outcomes].index(True) == 7
    assert not state.connected


def test_the_announced_departure_counts_down() -> None:
    ev = _ev()
    session = EvSession(arrival_t_index=0, departure_t_index=36, energy_mwh=0.01)
    state, _, _ = _run(ev, session, 0.0, 0, 6)
    assert state.departure_t_index == 36
    assert state.remaining_min == pytest.approx(30 * DT)


def test_an_episode_can_start_in_the_middle_of_a_session() -> None:
    ev = _ev()
    session = EvSession(arrival_t_index=-10, departure_t_index=6, energy_mwh=0.008)
    start = ev.connected_state(session, 0, need_mwh=0.003, dt_min=DT)
    assert start.remaining_min == pytest.approx(6 * DT)
    state, outcomes, _ = _run(ev, session, 0.011, 0, 6, state=start)
    assert sum(o.throughput_energy_mwh for o in outcomes) == pytest.approx(0.003)
    assert sum(o.unserved_energy_mwh for o in outcomes) == pytest.approx(0.0)


def test_a_session_overlapping_the_last_one_is_refused() -> None:
    ev = _ev()
    s = EvChargerState(
        asset_id="load:15",
        connected=True,
        need_mwh=0.001,
        arrival_t_index=0,
        departure_t_index=20,
        remaining_min=50.0,
    )
    other = EvSession(arrival_t_index=5, departure_t_index=30, energy_mwh=0.002)
    with pytest.raises(ValueError, match="still connected"):
        ev.limit_to_physics(s, Setpoint("load:15", 0.0), _x(5, other), DT)


def test_drawing_past_the_need_without_the_physics_limit_raises() -> None:
    ev = _ev()
    session = EvSession(arrival_t_index=0, departure_t_index=12, energy_mwh=0.0001)
    s = ev.initial_state(np.random.default_rng(0))
    with pytest.raises(ValueError, match="limit_to_physics"):
        ev.dynamics(s, Setpoint("load:15", 0.011), _x(0, session), GRID, DT)
    with pytest.raises(ValueError, match="limit_to_physics"):
        ev.dynamics(s, Setpoint("load:15", 0.011), _x(0), GRID, DT)


def test_a_vehicle_that_vanishes_early_leaves_its_need_unserved() -> None:
    ev = _ev()
    s = EvChargerState(
        asset_id="load:15",
        connected=True,
        need_mwh=0.002,
        arrival_t_index=0,
        departure_t_index=20,
        remaining_min=50.0,
    )
    state, outcome = ev.dynamics(s, Setpoint("load:15", 0.0), _x(10), GRID, DT)
    assert outcome.unserved_energy_mwh == pytest.approx(0.002)
    assert not state.connected


def test_energy_is_conserved_over_a_session() -> None:
    """Delivered plus unserved is what the vehicle announced, at any offer."""
    ev = _ev()
    session = EvSession(arrival_t_index=0, departure_t_index=24, energy_mwh=0.009)
    for offered in (0.0, 0.002, 0.0037, 0.011):
        _, outcomes, _ = _run(ev, session, offered, 0, 26)
        total = sum(o.throughput_energy_mwh + o.unserved_energy_mwh for o in outcomes)
        assert total == pytest.approx(0.009)


def test_a_session_must_be_under_way_in_its_input() -> None:
    ev = _ev()
    late = EvSession(arrival_t_index=5, departure_t_index=10, energy_mwh=0.001)
    s = ev.initial_state(np.random.default_rng(0))
    with pytest.raises(ValueError, match="not under way"):
        ev.limit_to_physics(s, Setpoint("load:15", 0.0), _x(2, late), DT)
