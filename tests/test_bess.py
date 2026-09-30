"""Tests of the battery storage model (M4)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.components.bess import BatteryState, BatteryStorage
from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import FlexAsset, Setpoint
from lvgrid_rl.core.schemas import (
    AssetRatings,
    ExogenousInput,
    GridState,
    Interval,
    PQBudgetState,
)

EXOGENOUS = ExogenousInput(
    t_index=0,
    series_ids=("load:0",),
    realized_mw=np.array([0.003]),
    bounds_mw=np.array([[0.0], [0.01]]),
    ambient_temp_degc=10.0,
    ghi_wm2=0.0,
)
INFO = InformationSet(
    t_index=0,
    timestamp=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
    measurements={},
    asset_states={},
    series_ids=("load:0",),
    exogenous_bounds_mw=np.array([[0.0], [0.01]]),
    forecast={},
    pq=PQBudgetState(windows_elapsed_count=0),
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


def _battery(**overrides) -> BatteryStorage:
    """10 kWh, 5 kW, 95 % each way, operating range 5 to 95 %."""
    params = dict(
        asset_id="storage:0",
        bus=1,
        ratings=AssetRatings(p_min_mw=-0.005, p_max_mw=0.005),
        capacity_mwh=0.01,
    )
    params.update(overrides)
    return BatteryStorage(**params)


def _state(battery: BatteryStorage, soc: float) -> BatteryState:
    return BatteryState(asset_id=battery.asset_id, energy_mwh=soc * battery.capacity_mwh)


def _run(battery, state, p_mw, minutes, dt_min=5):
    """Hold a setpoint for ``minutes``, limited per step as the environment does."""
    sp = Setpoint(asset_id=battery.asset_id, p_mw=p_mw)
    outcomes = []
    for _ in range(int(minutes // dt_min)):
        applied = battery.limit_to_physics(state, sp, EXOGENOUS, dt_min)
        state, outcome = battery.dynamics(state, applied, EXOGENOUS, GRID, dt_min)
        outcomes.append(outcome)
    return state, outcomes


# ---------------------------------------------------------------------------
# Protocol and parameters
# ---------------------------------------------------------------------------


def test_battery_satisfies_the_asset_protocol() -> None:
    assert isinstance(_battery(), FlexAsset)


def test_action_space_is_the_rated_box() -> None:
    """Invariant I4: rated limits, not the state-dependent feasible range."""
    spec = _battery().action_spec()
    assert spec.names == ("p_mw",)
    assert spec.bounds == (Interval(-0.005, 0.005),)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ratings": AssetRatings(p_min_mw=0.001, p_max_mw=0.005)}, "enclose zero"),
        ({"capacity_mwh": 0.0}, "capacity_mwh"),
        ({"soc_min_frac": 0.9, "soc_max_frac": 0.5}, "soc_min_frac"),
        ({"eta_charge_frac": 0.0}, "eta_charge_frac"),
        ({"eta_discharge_frac": 1.2}, "eta_discharge_frac"),
        ({"standing_loss_mw": -1e-6}, "standing_loss_mw"),
        ({"initial_soc_frac": Interval(0.5, 1.5)}, "initial_soc_frac"),
    ],
)
def test_inconsistent_parameters_are_rejected(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _battery(**overrides)


def test_initial_state_is_drawn_from_the_configured_band() -> None:
    battery = _battery(initial_soc_frac=Interval(0.4, 0.6))
    socs = [
        battery.soc_frac(battery.initial_state(np.random.default_rng(seed)))
        for seed in range(50)
    ]
    assert min(socs) >= 0.4
    assert max(socs) <= 0.6
    assert len(set(socs)) > 1


def test_a_point_band_gives_a_deterministic_start() -> None:
    """For evaluation: the same start in every run, whatever the generator."""
    battery = _battery(initial_soc_frac=Interval(0.5, 0.5))
    for seed in (0, 1, 2):
        state = battery.initial_state(np.random.default_rng(seed))
        assert battery.soc_frac(state) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Energy integration
# ---------------------------------------------------------------------------


def test_charging_stores_the_terminal_energy_times_the_efficiency() -> None:
    """Consumer convention: positive power charges."""
    battery = _battery(eta_charge_frac=0.9)
    state, outcomes = _run(battery, _state(battery, 0.2), 0.004, 60)
    # 4 kW for one hour at 90 %: 3.6 kWh into storage.
    assert state.energy_mwh == pytest.approx(0.002 + 0.0036)
    assert sum(o.throughput_energy_mwh for o in outcomes) == pytest.approx(0.004)
    assert sum(o.loss_energy_mwh for o in outcomes) == pytest.approx(0.0004)


def test_discharging_draws_the_terminal_energy_divided_by_the_efficiency() -> None:
    battery = _battery(eta_discharge_frac=0.8)
    state, outcomes = _run(battery, _state(battery, 0.8), -0.004, 60)
    # 4 kWh delivered at 80 % takes 5 kWh out of storage.
    assert state.energy_mwh == pytest.approx(0.008 - 0.005)
    assert sum(o.throughput_energy_mwh for o in outcomes) == pytest.approx(0.004)
    assert sum(o.loss_energy_mwh for o in outcomes) == pytest.approx(0.001)


def test_energy_is_conserved() -> None:
    """Energy in at the terminal equals the change in storage plus losses."""
    battery = _battery(
        eta_charge_frac=0.93, eta_discharge_frac=0.91, standing_loss_mw=3e-5
    )
    state = _state(battery, 0.5)
    start = state.energy_mwh
    terminal = 0.0
    losses = 0.0
    for p_mw, minutes in ((0.003, 45), (-0.0045, 30), (0.0, 60), (0.001, 15)):
        state, outcomes = _run(battery, state, p_mw, minutes)
        terminal += p_mw * minutes / 60.0
        losses += sum(o.loss_energy_mwh for o in outcomes)
    assert terminal == pytest.approx(state.energy_mwh - start + losses)


def test_round_trip_efficiency_is_the_product_of_both() -> None:
    battery = _battery(eta_charge_frac=0.95, eta_discharge_frac=0.9)
    state = _state(battery, 0.3)
    charged, _ = _run(battery, state, 0.004, 30)
    stored = charged.energy_mwh - state.energy_mwh
    back = stored * 0.9  # what a full discharge of the stored energy delivers
    assert back / (0.004 * 0.5) == pytest.approx(0.95 * 0.9)


def test_standing_loss_can_drain_an_idle_battery_below_its_operating_limit() -> None:
    """The limits bind the controller, not physics -- but never below empty."""
    battery = _battery(standing_loss_mw=0.001)
    state, _ = _run(battery, _state(battery, 0.06), 0.0, 60)
    assert battery.soc_frac(state) < battery.soc_min_frac
    state, outcomes = _run(battery, state, 0.0, 60)
    assert state.energy_mwh == 0.0
    # Only what was there can be lost.
    assert sum(o.loss_energy_mwh for o in outcomes) <= 0.0006 + 1e-12


def test_a_drained_battery_cannot_be_discharged() -> None:
    battery = _battery()
    empty = BatteryState(asset_id=battery.asset_id, energy_mwh=0.0)
    sp = battery.to_setpoint(empty, np.array([-0.004]), INFO, hold_min=15)
    assert sp.p_mw == 0.0
    assert "p_mw_soc" in sp.clipping_info


# ---------------------------------------------------------------------------
# Projection and reporting
# ---------------------------------------------------------------------------


def test_a_request_beyond_the_rating_is_reported_as_such() -> None:
    battery = _battery()
    sp = battery.to_setpoint(_state(battery, 0.5), np.array([0.008]), INFO, hold_min=15)
    assert sp.p_mw == pytest.approx(0.005)
    assert sp.clipping_info == {"p_mw": pytest.approx(-0.003)}


def test_charging_a_nearly_full_battery_is_reported_as_a_soc_limit() -> None:
    """A policy that keeps asking to charge a full battery must be visible."""
    battery = _battery()
    sp = battery.to_setpoint(_state(battery, 0.94), np.array([0.005]), INFO, hold_min=15)
    assert 0.0 < sp.p_mw < 0.005
    assert set(sp.clipping_info) == {"p_mw_soc"}


def test_rating_and_soc_limits_are_reported_separately() -> None:
    battery = _battery()
    sp = battery.to_setpoint(_state(battery, 0.94), np.array([0.009]), INFO, hold_min=15)
    assert set(sp.clipping_info) == {"p_mw", "p_mw_soc"}


def test_a_projected_setpoint_can_be_held_for_the_whole_step() -> None:
    """Feasible over the hold: the physics limit never has to intervene."""
    battery = _battery(standing_loss_mw=2e-5)
    for soc, request in ((0.94, 0.005), (0.06, -0.005), (0.5, 0.005), (0.5, -0.005)):
        state = _state(battery, soc)
        sp = battery.to_setpoint(state, np.array([request]), INFO, hold_min=15)
        for _ in range(3):
            applied = battery.limit_to_physics(state, sp, EXOGENOUS, 5)
            assert applied is sp, f"physics had to limit at soc={soc}"
            state, _ = battery.dynamics(state, applied, EXOGENOUS, GRID, 5)
        assert battery.energy_min_mwh - 1e-12 <= state.energy_mwh
        assert state.energy_mwh <= battery.energy_max_mwh + 1e-12


def test_the_projection_reaches_the_limit_rather_than_stopping_short() -> None:
    """Conservative, but not wasteful: without losses the limit is hit exactly."""
    battery = _battery()
    state = _state(battery, 0.9)
    sp = battery.to_setpoint(state, np.array([0.005]), INFO, hold_min=15)
    state, _ = _run(battery, state, sp.p_mw, 15)
    assert state.energy_mwh == pytest.approx(battery.energy_max_mwh)


def test_physics_limits_a_setpoint_formed_elsewhere_and_reports_it() -> None:
    """A baseline or safety mechanism may hand over an unprojected setpoint."""
    battery = _battery()
    state = _state(battery, 0.94)
    sp = Setpoint(asset_id=battery.asset_id, p_mw=0.005)
    applied = battery.limit_to_physics(state, sp, EXOGENOUS, 5)
    assert applied.p_mw < 0.005
    assert "p_mw_bms" in applied.clipping_info
    assert sp.clipping_info == {}, "the input setpoint was mutated"


def test_skipping_the_physics_limit_is_an_error_not_a_silent_clamp() -> None:
    battery = _battery()
    sp = Setpoint(asset_id=battery.asset_id, p_mw=0.005)
    with pytest.raises(ValueError, match="upper state-of-charge limit"):
        battery.dynamics(_state(battery, 0.94), sp, EXOGENOUS, GRID, 15)
    sp = Setpoint(asset_id=battery.asset_id, p_mw=-0.005)
    with pytest.raises(ValueError, match="lower state-of-charge limit"):
        battery.dynamics(_state(battery, 0.06), sp, EXOGENOUS, GRID, 15)


def test_the_feasible_range_is_a_box_that_shrinks_towards_the_limits() -> None:
    """What the certified shield will need per asset: an interval in power."""
    battery = _battery()
    middle = battery.feasible_power(_state(battery, 0.5), 15)
    near_full = battery.feasible_power(_state(battery, 0.94), 15)
    near_empty = battery.feasible_power(_state(battery, 0.06), 15)
    assert middle == Interval(-0.005, 0.005)
    assert near_full.hi < middle.hi and near_full.lo == middle.lo
    assert near_empty.lo > middle.lo and near_empty.hi == middle.hi
