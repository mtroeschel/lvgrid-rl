"""Reference methods for the flexible assets, B5 and B6 (M4, step 4.5b).

On hand-built assets at two buses, so that what each rule does can be read off
a single decision.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.baselines.flexibility import En14aDimming, GreedyLocal, en14a_cap_mw
from lvgrid_rl.baselines.methods import PUDroop
from lvgrid_rl.components.bess import BatteryState, BatteryStorage
from lvgrid_rl.components.ev_charger import EvCharger, EvChargerState
from lvgrid_rl.components.heat_pump import HeatPump, HeatPumpState
from lvgrid_rl.components.pv import PvSystem
from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.schemas import AssetRatings, PQBudgetState

pytest.importorskip("gymnasium", reason="extra 'env' not installed")

from lvgrid_rl.env.actions import ActionMapper  # noqa: E402

PV = PvSystem(
    asset_id="sgen:0",
    bus=1,
    ratings=AssetRatings(p_min_mw=-0.03, p_max_mw=0.0),
    series_id="sgen:0",
)
HP = HeatPump(
    asset_id="load:3",
    bus=1,
    ratings=AssetRatings(p_min_mw=0.0, p_max_mw=0.006),
    series_id="heat:load:3",
    cop_key="cop:load:3",
    capacity_mwh=0.04,
)
EV = EvCharger(asset_id="load:5", bus=1, ratings=AssetRatings(0.0, 0.011))
EV_NO_PV = EvCharger(asset_id="load:6", bus=2, ratings=AssetRatings(0.0, 0.022))
BESS = BatteryStorage(
    asset_id="storage:0",
    bus=1,
    ratings=AssetRatings(p_min_mw=-0.03, p_max_mw=0.03),
    capacity_mwh=0.06,
)
ASSETS = (PV, HP, EV, EV_NO_PV, BESS)
LOCAL = {1: ("load:1", "sgen:0"), 2: ("load:2",)}
MAPPER = ActionMapper.from_assets(ASSETS)
POSITIONS = (0, 0, 0, 1, 0)


def _info(
    pv_mw=-0.02,
    load_mw=0.003,
    vm=1.0,
    loading=50.0,
    slack=0.01,
    buffer_frac=0.5,
    ev=None,
    soc=0.5,
) -> InformationSet:
    ev = ev or {}
    states = {
        "sgen:0": PV.initial_state(np.random.default_rng(0)),
        "load:3": HeatPumpState("load:3", energy_mwh=buffer_frac * HP.capacity_mwh),
        "load:5": ev.get("load:5", EvChargerState("load:5")),
        "load:6": ev.get("load:6", EvChargerState("load:6")),
        "storage:0": BatteryState("storage:0", energy_mwh=soc * BESS.capacity_mwh),
    }
    series = ("sgen:0", "load:1", "load:2", "heat:load:3")
    return InformationSet(
        t_index=0,
        timestamp=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
        measurements={
            "vm_pu": np.array([vm, vm]),
            "trafo_loading_percent_max": loading,
            "line_loading_percent_max": 30.0,
            "p_slack_mw": slack,
            "losses_mw": 0.0,
        },
        asset_states=states,
        series_ids=series,
        exogenous_bounds_mw=np.zeros((2, len(series))),
        forecast={
            "sgen:0": np.array([pv_mw]),
            "load:1": np.array([load_mw]),
            "load:2": np.array([0.002]),
            "heat:load:3": np.array([0.002]),
            "cop:load:3": np.array([3.5]),
        },
        pq=PQBudgetState(windows_elapsed_count=0),
    )


def _physical(controller, info) -> dict[str, float]:
    values = MAPPER.to_physical(controller.act(info))
    return dict(zip(MAPPER.asset_ids, values, strict=True))


def _droop() -> PUDroop:
    return PUDroop(MAPPER, POSITIONS, 1.04, 1.10)


def _connected(asset_id, need_mwh, remaining_min) -> EvChargerState:
    return EvChargerState(
        asset_id=asset_id,
        connected=True,
        need_mwh=need_mwh,
        arrival_t_index=0,
        departure_t_index=1000,
        remaining_min=remaining_min,
    )


# ---------------------------------------------------------------------------
# B5
# ---------------------------------------------------------------------------


def test_the_14a_guaranteed_power() -> None:
    assert en14a_cap_mw("ev", 0.022) == pytest.approx(0.0042)
    assert en14a_cap_mw("hp", 0.006) == pytest.approx(0.0042)
    assert en14a_cap_mw("hp", 0.02) == pytest.approx(0.008)


def test_dimming_starts_under_load_congestion_and_keeps_pv_on_the_droop() -> None:
    ev = {
        "load:5": _connected("load:5", 0.01, 600),
        "load:6": _connected("load:6", 0.01, 600),
    }
    b5 = En14aDimming(MAPPER, ASSETS, _droop(), loading_on_percent=90.0)
    calm = _physical(b5, _info(loading=60.0, ev=ev, buffer_frac=0.1))
    assert calm["load:5"] == pytest.approx(0.011)
    congested = _physical(b5, _info(loading=95.0, ev=ev, buffer_frac=0.1))
    assert congested["load:5"] == pytest.approx(0.0042)
    assert congested["load:6"] == pytest.approx(0.0042)
    assert congested["load:3"] == pytest.approx(0.0042)  # thermostat on, dimmed
    assert congested["storage:0"] == pytest.approx(0.0)  # idle as without control
    droop = _physical(_droop(), _info(loading=95.0, ev=ev, buffer_frac=0.1))
    assert congested["sgen:0"] == pytest.approx(droop["sgen:0"])


def test_dimming_has_hysteresis_and_never_starts_on_back_feed() -> None:
    ev = {"load:5": _connected("load:5", 0.01, 600)}
    b5 = En14aDimming(MAPPER, ASSETS, _droop(), 90.0, hysteresis_percent=10.0)
    _physical(b5, _info(loading=95.0, ev=ev))
    assert _physical(b5, _info(loading=85.0, ev=ev))["load:5"] == pytest.approx(0.0042)
    assert _physical(b5, _info(loading=79.0, ev=ev))["load:5"] == pytest.approx(0.011)
    exporting = _info(loading=120.0, slack=-0.05, ev=ev)
    assert _physical(b5, exporting)["load:5"] == pytest.approx(0.011)


# ---------------------------------------------------------------------------
# B6
# ---------------------------------------------------------------------------


def _b6(threshold=0.0, margin_h=1.0) -> GreedyLocal:
    return GreedyLocal(MAPPER, ASSETS, _droop(), LOCAL, 0.25, threshold, margin_h)


def test_a_vehicle_waits_until_it_must_and_then_charges_at_full_power() -> None:
    """At bus 2 there is no PV: charging only as late as necessary."""
    empty = _physical(_b6(), _info())
    assert empty["load:6"] == pytest.approx(0.022), "an arrival must not wait"
    need = 0.022  # an hour at full power
    early = _physical(_b6(), _info(ev={"load:6": _connected("load:6", need, 600)}))
    assert early["load:6"] == 0.0
    # Laxity 10 h - 1 h - one step is above the margin of 1 h: still waiting.
    due = _physical(_b6(), _info(ev={"load:6": _connected("load:6", need, 135)}))
    assert due["load:6"] == pytest.approx(0.022)  # laxity 1.25 h - 0.25 h <= 1 h


def test_the_local_surplus_goes_to_heat_pump_vehicle_and_battery_in_turn() -> None:
    """At bus 1: 20 kW PV, 3 kW load -- 17 kW surplus."""
    ev = {"load:5": _connected("load:5", 0.02, 900)}
    out = _physical(_b6(), _info(ev=ev, buffer_frac=0.5))
    assert out["load:3"] == pytest.approx(0.006)  # pre-heating at its rating
    assert out["load:5"] == pytest.approx(0.011)  # the vehicle takes its rating
    assert out["storage:0"] == pytest.approx(0.017 - 0.006 - 0.011, abs=1e-12)


def test_the_battery_charges_above_the_threshold_and_covers_the_local_load() -> None:
    full_buffer = 0.95
    surplus = _physical(_b6(threshold=0.2), _info(buffer_frac=full_buffer))
    # 17 kW surplus, 6 kW threshold (0.2 x 30 kWp): 11 kW into the battery.
    assert surplus["storage:0"] == pytest.approx(0.011)
    evening = _physical(_b6(), _info(pv_mw=0.0, load_mw=0.004, buffer_frac=full_buffer))
    assert evening["storage:0"] == pytest.approx(-0.004)


def test_the_battery_stays_within_what_its_state_of_charge_allows() -> None:
    nearly_full = _physical(_b6(), _info(buffer_frac=0.95, soc=0.94))
    feasible = BESS.feasible_power(BatteryState("storage:0", 0.94 * 0.06), 15.0)
    assert nearly_full["storage:0"] == pytest.approx(feasible.hi)


def test_b6_keeps_pv_on_the_droop() -> None:
    info = _info(vm=1.08)
    assert _physical(_b6(), info)["sgen:0"] == pytest.approx(
        _physical(_droop(), info)["sgen:0"]
    )


def test_tuned_b5_and_b6_are_offered_only_with_flexible_assets() -> None:
    from types import SimpleNamespace

    from lvgrid_rl.baselines.flexibility import flex_baselines

    rule = {"pv_rule": {"p_u_droop": {"v_start": 1.04, "v_max": 1.1}}}
    results = {
        "en14a_dimming": {"params": {"loading_on_percent": 90.0}, **rule},
        "greedy_local": {
            "params": {"surplus_threshold_frac": 0.2, "margin_h": 1.0},
            **rule,
        },
    }
    flexible = flex_baselines(results, POSITIONS, SimpleNamespace(mapper=MAPPER))
    assert set(flexible) == {"en14a_dimming(90.0)", "greedy_local(0.2,1.0)"}
    pv_only = ActionMapper.from_assets((PV,))
    assert flex_baselines(results, (0,), SimpleNamespace(mapper=pv_only)) == {}
    assert flex_baselines({}, POSITIONS, SimpleNamespace(mapper=MAPPER)) == {}
