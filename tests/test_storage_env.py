"""The environment with controllable batteries at the PV buses (M4, step 4.2a)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("simbench", reason="extra 'sim' not installed")
pytest.importorskip("gymnasium", reason="extra 'env' not installed")

from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec  # noqa: E402
from lvgrid_rl.env.factory import DEFAULT_STORAGE, make_env  # noqa: E402

EVALUATE = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)


def _stress(storage):
    """The PV-strongest stress week, as in the curtailment tests."""
    env = make_env(seed=1, episode_spec=EVALUATE, set_name="stress", storage=storage)
    for _ in range(3):
        env.reset(seed=1)
    return env


@pytest.fixture(scope="module")
def env():
    return make_env(seed=1, storage=DEFAULT_STORAGE)


def _batteries(env):
    return [a for a in env.assets if a.kind == "bess"]


def _pv(env):
    return [a for a in env.assets if a.kind == "pv"]


# ---------------------------------------------------------------------------
# Placement and sizing
# ---------------------------------------------------------------------------


def test_one_battery_per_pv_system_at_its_bus(env) -> None:
    pv, batteries = _pv(env), _batteries(env)
    assert len(batteries) == len(pv) == 8
    assert [b.bus for b in batteries] == [p.bus for p in pv]


def test_sizing_is_one_kw_and_two_kwh_per_kwp(env) -> None:
    """Sized from the scenario-modified PV, so a PV scaling scales storage."""
    for pv, battery in zip(_pv(env), _batteries(env), strict=True):
        kwp = -pv.ratings.p_min_mw
        assert battery.ratings.p_max_mw == pytest.approx(kwp)
        assert battery.ratings.p_min_mw == pytest.approx(-kwp)
        assert battery.capacity_mwh == pytest.approx(2.0 * kwp)
    total_kwp = sum(-p.ratings.p_min_mw for p in _pv(env)) * 1000
    assert total_kwp == pytest.approx(702.3)


def test_the_simbench_storage_units_stay_uncontrolled(env) -> None:
    """Five profile-driven units at buses without PV, part of the background."""
    controlled = {a.asset_id for a in env.assets}
    simbench = [f"storage:{i}" for i in range(5)]
    assert not controlled & set(simbench)
    assert all(series in env.series_ids for series in simbench)


def test_batteries_are_written_to_the_grid(env) -> None:
    """Each battery has its own pandapower element, found by its id."""
    net = env.model.net
    for battery in _batteries(env):
        index = int(battery.asset_id.split(":")[1])
        assert net.storage.at[index, "bus"] == battery.bus
        assert net.storage.at[index, "type"] == "controlled_bess"


def test_evaluation_weeks_start_at_a_fixed_state_of_charge() -> None:
    """Controllers and seeds are compared from the same starting point."""
    socs = set()
    for seed in (1, 2):
        e = make_env(
            seed=seed, episode_spec=EVALUATE, set_name="test", storage=DEFAULT_STORAGE
        )
        e.reset(seed=seed)
        for battery in _batteries(e):
            socs.add(round(battery.soc_frac(e._asset_states[battery.asset_id]), 12))  # noqa: SLF001
    assert socs == {0.5}


# ---------------------------------------------------------------------------
# Idle batteries leave M3 untouched
# ---------------------------------------------------------------------------


def _rollout(env, action_of, steps=96):
    out = []
    for _ in range(steps):
        _, reward, _, _, info = env.step(action_of(env))
        out.append(
            (
                reward,
                info["curtailed_energy_mwh"],
                info["cost/thermal_overload"],
                info["k95_violations"],
            )
        )
    return out


def test_idle_batteries_leave_the_grid_as_in_m3() -> None:
    """With every battery at zero, the power flow is the M3 power flow.

    This is what keeps the M3 reference values and the tuned baseline
    parameters valid for M4: the added elements inject nothing until a
    controller uses them. Checked for no curtailment and for a cap.

    **Not bit for bit.** Elements with zero injection still change the order in
    which pandapower sums bus powers, and the results differ in the last digits
    (around 1e-13 relative). The counts that decide the EN 50160 assessment are
    integers and must agree exactly; the continuous quantities agree to within
    rounding.
    """
    from lvgrid_rl.baselines.methods import FixedCap

    for label, action_of in (
        ("neutral", lambda e: e.mapper.neutral_action()),
        ("cap 0.4", lambda e: FixedCap(e.mapper, cap=0.4).act(None)),
    ):
        plain = np.array(_rollout(_stress(None), action_of))
        with_storage = np.array(_rollout(_stress(DEFAULT_STORAGE), action_of))
        assert np.array_equal(plain[:, 3], with_storage[:, 3]), label
        assert np.allclose(plain, with_storage, rtol=1e-10, atol=1e-12), label
        assert plain[:, 2].sum() > 0.0, "the stretch must contain overload"


def test_rule_based_controllers_leave_batteries_idle(env) -> None:
    """A PV cap applied to a battery would be a permanent discharge."""
    from lvgrid_rl.baselines.methods import (
        DoNothing,
        FixedCap,
        PUDroop,
        asset_bus_positions,
    )

    env.reset(seed=1)
    info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
    positions = asset_bus_positions(env)
    mask = env.mapper.component_mask("bess", "p_mw")
    for controller in (
        DoNothing(env.mapper),
        FixedCap(env.mapper, cap=0.4),
        PUDroop(env.mapper, positions, 1.0, 1.01),  # curtails at any voltage
    ):
        physical = env.mapper.to_physical(controller.act(info))
        assert np.allclose(physical[mask], 0.0), controller.name


# ---------------------------------------------------------------------------
# Batteries in use
# ---------------------------------------------------------------------------


def _charge_at_noon(env):
    """Full PV infeed, every battery charging at rated power."""
    physical = env.mapper.neutral.copy()
    mask = env.mapper.component_mask("bess", "p_mw")
    physical[mask] = env.mapper.upper[mask]
    return env.mapper.to_normalised(physical)


def _charge_from(step: int, absorb_own_pv: bool):
    """Idle until ``step``, then charge: at rated power, or at the own PV."""
    counter = {"t": 0}

    def action_of(env):
        counter["t"] += 1
        if counter["t"] <= step:
            return env.mapper.neutral_action()
        if not absorb_own_pv:
            return _charge_at_noon(env)
        info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
        physical = env.mapper.neutral.copy()
        ids = list(env.mapper.asset_ids)
        for pv, battery in zip(_pv(env), _batteries(env), strict=True):
            available = -float(info.forecast[pv.series_id][0])
            physical[ids.index(battery.asset_id)] = min(
                battery.ratings.p_max_mw, available
            )
        return env.mapper.to_normalised(physical)

    return action_of


def test_well_timed_charging_reduces_overload_and_badly_timed_adds_to_it() -> None:
    """The point of M4: something to shift, not only something to cut.

    Timing is everything, which is what makes this a control problem. Charging
    at rated power from midnight draws 702 kW from the grid at night and leaves
    the batteries full long before the PV peak. Absorbing each system's own PV
    from ten o'clock takes the peak: on this day the overload falls by more than
    half (3.87 to 1.74; from eleven o'clock 1.59, from noon 2.11 again).
    """
    idle = _rollout(_stress(DEFAULT_STORAGE), lambda e: e.mapper.neutral_action())
    at_night = _rollout(_stress(DEFAULT_STORAGE), _charge_from(0, absorb_own_pv=False))
    at_midday = _rollout(_stress(DEFAULT_STORAGE), _charge_from(40, absorb_own_pv=True))
    overload = {
        name: sum(r[2] for r in run)
        for name, run in (("idle", idle), ("night", at_night), ("midday", at_midday))
    }
    assert overload["midday"] < 0.5 * overload["idle"]
    assert overload["night"] > overload["idle"]


def test_throughput_is_reported_and_priced() -> None:
    env = _stress(DEFAULT_STORAGE)
    _, _, _, _, info = env.step(_charge_at_noon(env))
    rated = sum(b.ratings.p_max_mw for b in _batteries(env))
    # Starting from 50 %, every battery can take rated power for 15 minutes.
    assert info["storage_throughput_mwh"] == pytest.approx(rated * 0.25)
    assert info["reward/bess_degradation"] == pytest.approx(-0.2 * rated * 0.25)
    assert info["asset_loss_mwh"] == pytest.approx(rated * 0.25 * 0.05)


def test_charging_a_full_battery_is_counted_by_cause() -> None:
    """Every request beyond the state of charge shows up under ``p_mw_soc``."""
    env = _stress(DEFAULT_STORAGE)
    counts = []
    for _ in range(12):  # three hours; 2 h at rated power fills from 50 %
        _, _, _, _, info = env.step(_charge_at_noon(env))
        counts.append(info.get("clipping/bess/p_mw_soc", 0))
    assert counts[0] == 0
    assert counts[-1] == len(_batteries(env))
    # Projected setpoints are feasible for the whole hold, so the battery
    # management cut-off never has to act.
    assert "clipping/bess/p_mw_bms" not in info


def test_state_of_charge_is_observed(env) -> None:
    names = env.obs_builder.feature_names
    soc = [n for n in names if n.endswith("/soc_frac")]
    assert soc == [f"asset/{b.asset_id}/soc_frac" for b in _batteries(env)]


def test_without_batteries_the_observation_is_the_m3_one() -> None:
    plain = make_env(seed=1)
    assert not [n for n in plain.obs_builder.feature_names if n.startswith("asset/")]


def test_neutral_action_idles_the_batteries(env) -> None:
    physical = env.mapper.to_physical(env.mapper.neutral_action())
    mask = env.mapper.component_mask("bess", "p_mw")
    assert np.allclose(physical[mask], 0.0)
    pv = env.mapper.component_mask("pv", "p_mw")
    assert np.allclose(physical[pv], env.mapper.lower[pv]), "PV at full infeed"


def test_single_partition_reproduces_the_flat_environment_with_storage() -> None:
    """Rule 4 of section 6.3, with a second asset type in the vector."""
    from lvgrid_rl.env.multiagent import PartitionedEnv

    probe = make_env(seed=11, storage=DEFAULT_STORAGE)
    actions = [
        np.full(probe.mapper.dim, value, dtype=np.float64)
        for value in (-1.0, -0.25, 0.5, 1.0, 0.0)
    ]
    flat_env = make_env(seed=11, storage=DEFAULT_STORAGE)
    flat_env.reset(seed=11)
    flat = [flat_env.step(a) for a in actions]
    partitioned = PartitionedEnv(make_env(seed=11, storage=DEFAULT_STORAGE))
    partitioned.reset(seed=11)
    grouped = [partitioned.step(partitioned.split_action(a)) for a in actions]
    for (obs, reward, _, _, _), (obs_d, reward_d, _, _, _) in zip(
        flat, grouped, strict=True
    ):
        assert np.array_equal(obs, obs_d["central"])
        assert reward == reward_d["central"]


def test_the_environment_passes_the_gymnasium_checker(env) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


def test_the_runner_reports_storage_kpis_and_clipping_by_cause() -> None:
    """Throughput, losses and limitations reach the results table.

    Also guards the controller name: an early version of the clipping tally
    reused the variable holding it, and every row came out named after a
    clipping key.
    """
    from lvgrid_rl.eval.runner import run_controller

    class AlwaysCharge:
        name = "always_charge"

        def __init__(self, env):
            self.env = env

        def reset(self, info) -> None:
            pass

        def act(self, info):
            return _charge_at_noon(self.env)

    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=1, randomise_budget=False)
    env = make_env(seed=1, episode_spec=spec, set_name="stress", storage=DEFAULT_STORAGE)
    result = run_controller(
        env, AlwaysCharge(env), "always_charge", "stress", n_episodes=1
    )
    summary = result.summary()
    assert summary["controller"] == "always_charge"
    assert summary["storage_throughput_mwh"] > 0.0
    assert summary["asset_loss_mwh"] > 0.0
    assert summary["clipping/bess/p_mw_soc"] > 0
    assert summary["clipped_steps"] > 0
