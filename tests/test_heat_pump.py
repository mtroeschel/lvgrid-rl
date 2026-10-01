"""Tests of the heat pump model with buffer store (M4, step 4.3b, D15)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.components.heat_pump import HeatPump, HeatPumpState
from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import FlexAsset, Setpoint
from lvgrid_rl.core.schemas import (
    AssetRatings,
    ExogenousInput,
    GridState,
    Interval,
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


def _hp(**overrides) -> HeatPump:
    """4 kW electrical, buffer of 28 kWh over a 20 K band, no losses."""
    params = dict(
        asset_id="load:3",
        bus=3,
        ratings=AssetRatings(p_min_mw=0.0, p_max_mw=0.004),
        series_id="heat:load:3",
        cop_key="cop:load:3",
        capacity_mwh=0.028,
    )
    params.update(overrides)
    return HeatPump(**params)


def _x(demand_mw=0.006, cop=3.5) -> ExogenousInput:
    return ExogenousInput(
        t_index=0,
        series_ids=("heat:load:3",),
        realized_mw=np.array([demand_mw]),
        bounds_mw=np.array([[0.0], [0.05]]),
        ambient_temp_degc=0.0,
        ghi_wm2=0.0,
        realized_ratio={"cop:load:3": cop},
    )


def _info(demand_mw=0.006, cop=3.5) -> InformationSet:
    return InformationSet(
        t_index=0,
        timestamp=datetime(2016, 1, 15, 6, 0, tzinfo=UTC),
        measurements={},
        asset_states={},
        series_ids=("heat:load:3",),
        exogenous_bounds_mw=np.array([[0.0], [0.05]]),
        forecast={"heat:load:3": np.array([demand_mw]), "cop:load:3": np.array([cop])},
        pq=PQBudgetState(windows_elapsed_count=0),
    )


def _state(hp, frac, running=False, duration=60.0, last=0.0) -> HeatPumpState:
    return HeatPumpState(
        asset_id=hp.asset_id,
        energy_mwh=frac * hp.capacity_mwh,
        last_p_mw=last,
        running=running,
        state_duration_min=duration,
    )


def _setpoint(hp, s, p_mw, demand=0.006, cop=3.5, hold=15):
    return hp.to_setpoint(s, np.array([p_mw]), _info(demand, cop), hold)


def _run(hp, s, p_mw, minutes, demand=0.006, cop=3.5, dt=5):
    sp = Setpoint(asset_id=hp.asset_id, p_mw=p_mw)
    outcomes = []
    for _ in range(int(minutes // dt)):
        applied = hp.limit_to_physics(s, sp, _x(demand, cop), dt)
        s, outcome = hp.dynamics(s, applied, _x(demand, cop), GRID, dt)
        outcomes.append(outcome)
    return s, outcomes


# ---------------------------------------------------------------------------
# Protocol and parameters
# ---------------------------------------------------------------------------


def test_heat_pump_satisfies_the_asset_protocol() -> None:
    assert isinstance(_hp(), FlexAsset)
    assert _hp().kind == "hp"


def test_action_space_is_zero_to_rated_and_neutral_is_off() -> None:
    spec = _hp().action_spec()
    assert spec.bounds == (Interval(0.0, 0.004),)
    assert spec.neutral == (0.0,)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ratings": AssetRatings(p_min_mw=-0.001, p_max_mw=0.004)}, "draws power"),
        ({"capacity_mwh": 0.0}, "capacity_mwh"),
        ({"min_modulation_frac": 0.0}, "min_modulation_frac"),
        ({"max_block_min": 10.0}, "max_block_min"),
        ({"thermostat_on_frac": 0.9, "thermostat_off_frac": 0.5}, "thermostat"),
    ],
)
def test_inconsistent_parameters_are_rejected(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _hp(**overrides)


# ---------------------------------------------------------------------------
# Heat balance
# ---------------------------------------------------------------------------


def test_heat_in_is_electricity_times_cop() -> None:
    hp = _hp()
    s, _ = _run(hp, _state(hp, 0.2), 0.004, 60, demand=0.0, cop=3.5)
    assert s.energy_mwh == pytest.approx(0.2 * 0.028 + 0.004 * 3.5)


def test_the_same_heat_costs_less_electricity_at_a_higher_cop() -> None:
    """Why shifting a heat pump is worth anything beyond the grid."""
    hp = _hp()
    low, _ = _run(hp, _state(hp, 0.2), 0.002, 60, demand=0.0, cop=2.5)
    high, _ = _run(hp, _state(hp, 0.2), 0.002, 60, demand=0.0, cop=4.5)
    assert high.energy_mwh - 0.2 * 0.028 == pytest.approx(
        (low.energy_mwh - 0.2 * 0.028) * 4.5 / 2.5
    )


def test_heat_is_conserved() -> None:
    """Heat in minus demand minus losses equals the change in the buffer."""
    hp = _hp(standing_loss_mw=0.00028)
    s = _state(hp, 0.5)
    start = s.energy_mwh
    heat_in = losses = demand_total = 0.0
    for p, minutes, demand, cop in (
        (0.003, 30, 0.004, 3.2),
        (0.0, 30, 0.005, 3.6),
        (0.004, 15, 0.002, 4.0),
    ):
        s, outcomes = _run(hp, s, p, minutes, demand, cop)
        heat_in += p * cop * minutes / 60
        demand_total += demand * minutes / 60
        losses += sum(o.loss_energy_mwh for o in outcomes)
    assert heat_in - demand_total - losses == pytest.approx(s.energy_mwh - start)


def test_drawing_below_the_band_is_a_comfort_violation_in_kelvin_hours() -> None:
    """A heat deficit of a quarter band is 5 K below the band at 20 K width."""
    hp = _hp()
    s = _state(hp, 0.0)
    # 7 kW of demand for one hour with the heat pump off: 7 kWh = a quarter band.
    s, outcomes = _run(hp, s, 0.0, 60, demand=0.007)
    assert hp.shortfall_k(s.energy_mwh) == pytest.approx(5.0)
    # The shortfall grows linearly from 0 to 5 K: the integral is 2.5 Kh.
    assert sum(o.comfort_deviation_kh for o in outcomes) == pytest.approx(2.5)
    assert sum(o.unserved_energy_mwh for o in outcomes) == 0.0


def test_below_the_floor_demand_goes_unserved() -> None:
    hp = _hp()
    s = _state(hp, -0.9)
    s, outcomes = _run(hp, s, 0.0, 60, demand=0.007)
    assert s.energy_mwh == pytest.approx(hp.floor_mwh)
    assert sum(o.unserved_energy_mwh for o in outcomes) == pytest.approx(
        0.007 - 0.1 * 0.028
    )


# ---------------------------------------------------------------------------
# Operating constraints, each reported under its own key
# ---------------------------------------------------------------------------


def test_requests_below_the_minimum_modulation_are_rounded_and_reported() -> None:
    hp = _hp()  # P_min = 1.2 kW
    s = _state(hp, 0.3)
    up = _setpoint(hp, s, 0.0008)
    down = _setpoint(hp, s, 0.0004)
    assert up.p_mw == pytest.approx(0.0012) and "p_mw_modulation" in up.clipping_info
    assert down.p_mw == 0.0 and "p_mw_modulation" in down.clipping_info


def test_a_request_beyond_the_rating_is_reported_as_such() -> None:
    hp = _hp()
    sp = _setpoint(hp, _state(hp, 0.3), 0.006)
    assert sp.p_mw == pytest.approx(0.004)
    assert set(sp.clipping_info) == {"p_mw"}


def test_the_minimum_idle_time_keeps_it_off() -> None:
    hp = _hp()
    sp = _setpoint(hp, _state(hp, 0.3, running=False, duration=10.0), 0.004)
    assert sp.p_mw == 0.0
    assert set(sp.clipping_info) == {"p_mw_min_idle"}


def test_the_minimum_run_time_keeps_it_on() -> None:
    hp = _hp()
    sp = _setpoint(hp, _state(hp, 0.3, running=True, duration=5.0, last=0.003), 0.0)
    assert sp.p_mw == pytest.approx(hp.p_min_running_mw)
    assert set(sp.clipping_info) == {"p_mw_min_run"}


def test_a_full_buffer_overrides_the_minimum_run_time() -> None:
    """The high-limit cut-out wins: a buffer cannot be heated past its top."""
    hp = _hp()
    s = _state(hp, 1.0, running=True, duration=5.0, last=0.003)
    sp = _setpoint(hp, s, 0.0, demand=0.0)
    assert sp.p_mw == 0.0
    assert "p_mw_min_run" not in sp.clipping_info


def test_filling_a_nearly_full_buffer_is_reported_as_a_buffer_limit() -> None:
    hp = _hp()
    sp = _setpoint(hp, _state(hp, 0.95), 0.004, demand=0.002, cop=3.5)
    assert sp.p_mw < 0.004
    assert set(sp.clipping_info) == {"p_mw_buffer"}


def test_blocking_beyond_the_limit_switches_it_on_while_there_is_demand() -> None:
    hp = _hp()
    long_off = _state(hp, 0.5, running=False, duration=125.0)
    on = _setpoint(hp, long_off, 0.0, demand=0.004)
    idle = _setpoint(hp, long_off, 0.0, demand=0.0)
    assert on.p_mw == pytest.approx(hp.p_min_running_mw)
    assert set(on.clipping_info) == {"p_mw_max_block"}
    assert idle.p_mw == 0.0, "no demand, no reason to run"


def test_physics_cuts_off_a_setpoint_that_would_overheat_the_buffer() -> None:
    """Demand inside the hold can fall below the forecast the decision used."""
    hp = _hp()
    s = _state(hp, 0.97)
    sp = Setpoint(asset_id=hp.asset_id, p_mw=0.004)
    applied = hp.limit_to_physics(s, sp, _x(demand_mw=0.0), 5)
    assert applied.p_mw < 0.004
    assert "p_mw_buffer_cutoff" in applied.clipping_info
    assert sp.clipping_info == {}, "the input setpoint was mutated"


def test_skipping_the_physics_limit_is_an_error_not_a_silent_clamp() -> None:
    hp = _hp()
    sp = Setpoint(asset_id=hp.asset_id, p_mw=0.004)
    with pytest.raises(ValueError, match="overheats the buffer"):
        hp.dynamics(_state(hp, 0.99), sp, _x(demand_mw=0.0), GRID, 15)


def test_switching_and_state_duration_are_tracked() -> None:
    hp = _hp()
    s = _state(hp, 0.3, running=False, duration=30.0)
    s, outcomes = _run(hp, s, 0.002, 15)
    assert s.running and s.state_duration_min == pytest.approx(15.0)
    assert sum(o.switching_count for o in outcomes) == 1
    s, outcomes = _run(hp, s, 0.002, 10)
    assert s.state_duration_min == pytest.approx(25.0)
    assert sum(o.switching_count for o in outcomes) == 0


# ---------------------------------------------------------------------------
# The default controller
# ---------------------------------------------------------------------------


def test_the_thermostat_switches_with_hysteresis() -> None:
    hp = _hp()
    info = _info()
    assert hp.default_action(_state(hp, 0.3, running=False), info)[0] == pytest.approx(
        0.004
    )
    assert hp.default_action(_state(hp, 0.6, running=False), info)[0] == 0.0
    assert hp.default_action(_state(hp, 0.6, running=True), info)[0] == pytest.approx(
        0.004
    )
    assert hp.default_action(_state(hp, 0.95, running=True), info)[0] == 0.0


def test_the_thermostat_keeps_a_winter_day_in_the_band() -> None:
    """Left to itself, the heat pump serves the demand without a violation."""
    hp = _hp(standing_loss_mw=0.00028, initial_buffer_frac=Interval(0.5, 0.5))
    s = hp.initial_state(np.random.default_rng(0))
    comfort = 0.0
    for _ in range(96):  # one day of 15-minute decisions
        action = hp.default_action(s, _info(demand_mw=0.009, cop=2.8))
        sp = hp.to_setpoint(s, action, _info(demand_mw=0.009, cop=2.8), 15)
        for _ in range(3):
            applied = hp.limit_to_physics(s, sp, _x(0.009, 2.8), 5)
            s, outcome = hp.dynamics(s, applied, _x(0.009, 2.8), GRID, 5)
            comfort += outcome.comfort_deviation_kh
    assert comfort == 0.0
    assert 0.0 <= hp.buffer_frac(s) <= 1.0


def test_initial_state_is_drawn_from_the_band_and_free_to_switch() -> None:
    hp = _hp(initial_buffer_frac=Interval(0.2, 0.4))
    states = [hp.initial_state(np.random.default_rng(i)) for i in range(20)]
    assert all(0.2 <= hp.buffer_frac(s) <= 0.4 for s in states)
    assert all(not s.running and s.state_duration_min >= hp.min_idle_min for s in states)
    fixed = replace(hp, initial_buffer_frac=Interval(0.5, 0.5))
    assert hp.buffer_frac(fixed.initial_state(np.random.default_rng(7))) == pytest.approx(
        0.5
    )
