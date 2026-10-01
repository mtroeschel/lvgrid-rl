"""The environment with controllable heat pumps (M4, step 4.3b part 2, D15).

The COP comes from a synthetic cache entry, so these tests run without the
when2heat download (in CI); one test checks the real data when it is present.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("simbench", reason="extra 'sim' not installed")
pytest.importorskip("gymnasium", reason="extra 'env' not installed")

from lvgrid_rl.data.cache import CacheKey, ProfileCache  # noqa: E402
from lvgrid_rl.data.sources.when2heat import SINKS, WHEN2HEAT_VERSION  # noqa: E402
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec  # noqa: E402
from lvgrid_rl.env.factory import (  # noqa: E402
    DEFAULT_HEAT_PUMPS,
    DEFAULT_STORAGE,
    make_env,
)
from lvgrid_rl.env.lv_grid_env import EnvConfig  # noqa: E402
from lvgrid_rl.env.obs import ObservationLayoutMode, ObservationSpec  # noqa: E402

EVALUATE = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
RAW = Path(f"data/raw/when2heat/when2heat-{WHEN2HEAT_VERSION}.csv")


@pytest.fixture(scope="module")
def sizing(tmp_path_factory):
    """Heat pump sizing pointed at a synthetic COP cache: 3.5 with a daily swing."""
    root = tmp_path_factory.mktemp("cop_cache")
    index = pd.date_range("2015-12-31 20:00", "2017-01-01 04:00", freq="1h", tz="UTC")
    swing = 0.4 * np.sin(2 * np.pi * (index.hour - 9) / 24)
    frame = pd.DataFrame(
        {f"{src}_{sink}": 3.5 + swing for src in ("ASHP", "GSHP") for sink in SINKS},
        index=index,
    )
    key = CacheKey(
        source="when2heat",
        source_version=WHEN2HEAT_VERSION,
        dataset="DE-cop-wallclock",
        sim_dt_min=60,
    )
    ProfileCache(root).store(key, frame, notes="synthetic; filled: ")
    return replace(DEFAULT_HEAT_PUMPS, cache_dir=str(root), when2heat_csv="/nonexistent")


@pytest.fixture(scope="module")
def env(sizing):
    e = make_env(seed=1, storage=DEFAULT_STORAGE, heat_pumps=sizing)
    e.reset(seed=1)
    return e


def _hps(env):
    return [a for a in env.assets if a.kind == "hp"]


def _winter(sizing, **kwargs):
    """The first test week, in winter, evaluation mode."""
    e = make_env(
        seed=1, set_name="test", episode_spec=EVALUATE, heat_pumps=sizing, **kwargs
    )
    e.reset(seed=1)
    return e


def _rollout(env, controller_of, steps=96):
    infos = []
    controller = controller_of(env)
    for _ in range(steps):
        info_set = env._information_set(env._t, env._last_grid)  # noqa: SLF001
        *_, info = env.step(controller.act(info_set))
        infos.append(info)
    return infos


# ---------------------------------------------------------------------------
# Placement and data path
# ---------------------------------------------------------------------------


def test_the_grid_heat_pumps_become_controllable_at_their_load_elements(env) -> None:
    pumps = _hps(env)
    assert len(pumps) == 8
    assert all(p.asset_id.startswith("load:") for p in pumps)
    # The live grid holds the current setpoints; the ratings are in a fresh one.
    from lvgrid_rl.grid.loader import load_grid
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    fresh = load_grid("1-LV-rural1--2-sw", SCENARIO_LIBRARY["moderate_growth"]).net
    for p in pumps:
        index = int(p.asset_id.split(":")[1])
        assert fresh.load.at[index, "bus"] == p.bus
        assert p.ratings.p_max_mw == pytest.approx(fresh.load.at[index, "p_mw"])


def test_their_profiles_no_longer_drive_the_grid(env) -> None:
    """A controllable heat pump must not also be written as uncontrolled load."""
    uncontrolled = set(env._uncontrolled)  # noqa: SLF001
    for p in _hps(env):
        assert p.asset_id not in uncontrolled
        assert p.series_id not in uncontrolled, "thermal demand is not a grid element"


def test_buffers_hold_two_hours_of_rated_heat(env) -> None:
    """At the synthetic COP of 3.5 on average, capacity is 2 h x rating x 3.5."""
    for p in _hps(env):
        assert p.capacity_mwh == pytest.approx(2.0 * p.ratings.p_max_mw * 3.5, rel=0.02)
        assert p.standing_loss_mw == pytest.approx(0.005 * p.capacity_mwh)


def test_thermal_demand_and_cop_reach_the_heat_pump_by_the_information_path(env) -> None:
    """Exogenous input and forecast, never an array the model holds (I3)."""
    exogenous = env._exogenous(env._t)  # noqa: SLF001
    info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
    for p in _hps(env):
        assert p.series_id in exogenous.series_ids
        assert 2.9 < exogenous.realized_ratio[p.cop_key] < 4.0
        assert len(info.forecast[p.cop_key]) == env.config.forecast_horizon
        assert len(info.forecast[p.series_id]) == env.config.forecast_horizon


def test_evaluation_weeks_start_with_half_full_buffers(sizing) -> None:
    e = _winter(sizing)
    for p in _hps(e):
        assert p.buffer_frac(e._asset_states[p.asset_id]) == pytest.approx(0.5)  # noqa: SLF001


def test_without_the_option_heat_pumps_stay_uncontrolled_load() -> None:
    plain = make_env(seed=1)
    assert not [a for a in plain.assets if a.kind == "hp"]
    assert "load:13" in plain._uncontrolled  # noqa: SLF001


# ---------------------------------------------------------------------------
# "Do nothing" is the thermostat
# ---------------------------------------------------------------------------


def test_do_nothing_leaves_heat_pumps_on_their_thermostats(sizing) -> None:
    from lvgrid_rl.baselines.methods import DoNothing

    infos = _rollout(_winter(sizing), lambda e: DoNothing(e.mapper))
    assert sum(i["hp_comfort_kh"] for i in infos) == 0.0
    assert sum(i["hp_switches"] for i in infos) > 0, "the thermostats switched"


def test_the_neutral_action_needs_the_state_when_heat_pumps_are_present(env) -> None:
    with pytest.raises(ValueError, match="controller of its own"):
        env.mapper.neutral_action()
    info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
    physical = env.mapper.to_physical(env.mapper.neutral_action(info))
    expected = np.concatenate(
        [p.default_action(info.asset_states[p.asset_id], info) for p in _hps(env)]
    )
    assert np.allclose(physical[env.mapper.component_mask("hp", "p_mw")], expected)


def test_pv_rules_leave_heat_pumps_on_their_thermostats(env) -> None:
    from lvgrid_rl.baselines.methods import (
        DoNothing,
        FixedCap,
        PUDroop,
        asset_bus_positions,
    )

    info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
    mask = env.mapper.component_mask("hp", "p_mw")
    reference = env.mapper.to_physical(DoNothing(env.mapper).act(info))[mask]
    for controller in (
        FixedCap(env.mapper, cap=0.4),
        PUDroop(env.mapper, asset_bus_positions(env), 1.0, 1.01),
    ):
        physical = env.mapper.to_physical(controller.act(info))
        assert np.allclose(physical[mask], reference), controller.name


# ---------------------------------------------------------------------------
# Comfort, its price and its KPIs
# ---------------------------------------------------------------------------


class _AllOff:
    """Keeps every heat pump off; everything else neutral."""

    name = "all_off"

    def __init__(self, env) -> None:
        self.env = env

    def reset(self, info) -> None:
        pass

    def act(self, info):
        physical = self.env.mapper.default_physical(info)
        physical[self.env.mapper.component_mask("hp", "p_mw")] = 0.0
        return self.env.mapper.to_normalised(physical)


def test_keeping_heat_pumps_off_in_winter_costs_comfort_and_is_priced(sizing) -> None:
    infos = _rollout(_winter(sizing), _AllOff, steps=192)
    comfort = sum(i["hp_comfort_kh"] for i in infos)
    assert comfort > 0.0
    assert sum(i["reward/hp_comfort"] for i in infos) == pytest.approx(-2.0 * comfort)


def test_blocking_beyond_the_limit_is_enforced_and_counted(sizing) -> None:
    infos = _rollout(_winter(sizing), _AllOff, steps=192)
    assert sum(i.get("clipping/hp/p_mw_max_block", 0) for i in infos) > 0


def test_the_runner_reports_heat_pump_kpis(sizing) -> None:
    from lvgrid_rl.eval.runner import run_controller

    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=1, randomise_budget=False)
    e = make_env(seed=1, set_name="test", episode_spec=spec, heat_pumps=sizing)
    summary = run_controller(e, _AllOff(e), "all_off", "test", n_episodes=1).summary()
    assert summary["controller"] == "all_off"
    for key in ("hp_comfort_kh", "hp_unserved_heat_mwh", "hp_switches"):
        assert key in summary
    assert summary["clipping/hp/p_mw_max_block"] > 0


# ---------------------------------------------------------------------------
# Observation and the policies on top of it
# ---------------------------------------------------------------------------


def test_the_flat_observation_holds_buffer_running_and_cop(env) -> None:
    names = env.obs_builder.feature_names
    for p in _hps(env):
        for feature in ("buffer_frac", "running", "cop"):
            assert f"asset/{p.asset_id}/{feature}" in names
        assert any(n.startswith(f"forecast/{p.series_id}/") for n in names)


def test_per_asset_blocks_cover_three_kinds(sizing) -> None:
    config = EnvConfig(
        observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET)
    )
    e = make_env(seed=1, storage=DEFAULT_STORAGE, heat_pumps=sizing, config=config)
    obs, _ = e.reset(seed=1)
    layout = e.observation_layout
    assert set(layout.kinds) == {"pv", "bess", "hp"}
    block = layout.blocks_of("hp")[0]
    values = dict(
        zip(
            [
                n.split("/", 2)[2]
                for n in e.obs_builder.feature_names[block.obs_start : block.obs_stop]
            ],
            obs[block.obs_start : block.obs_stop],
            strict=True,
        )
    )
    pump = _hps(e)[0]
    assert values["buffer_frac"] == pytest.approx(
        pump.buffer_frac(e._asset_states[pump.asset_id]),
        abs=1e-6,  # noqa: SLF001
    )


def test_the_shared_policy_builds_for_all_three_kinds(sizing) -> None:
    pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")
    from lvgrid_rl.agents.factory import AgentSpec, make_agent
    from lvgrid_rl.agents.shared_policy import SharedAssetPolicy
    from lvgrid_rl.data.timebase import TimeBase

    config = EnvConfig(
        observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET)
    )
    e = make_env(seed=1, storage=DEFAULT_STORAGE, heat_pumps=sizing, config=config)
    spec = AgentSpec.from_yaml("configs/agent/ppo_shared.yaml")
    spec = replace(
        spec, hyperparams={**spec.hyperparams, "n_steps": 32, "batch_size": 32}
    )
    model = make_agent(spec, e, TimeBase(5, 15), seed=1)
    assert set(model.policy.kind_log_std) == {"pv", "bess", "hp"}
    assert isinstance(model.policy, SharedAssetPolicy)
    model.learn(32)


def test_the_environment_passes_the_gymnasium_checker(env) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


def test_single_partition_reproduces_the_flat_environment_with_heat_pumps(sizing) -> None:
    from lvgrid_rl.env.multiagent import PartitionedEnv

    probe = make_env(seed=11, storage=DEFAULT_STORAGE, heat_pumps=sizing)
    actions = [np.full(probe.mapper.dim, v) for v in (-1.0, 0.3, 1.0)]
    flat_env = make_env(seed=11, storage=DEFAULT_STORAGE, heat_pumps=sizing)
    flat_env.reset(seed=11)
    flat = [flat_env.step(a) for a in actions]
    partitioned = PartitionedEnv(
        make_env(seed=11, storage=DEFAULT_STORAGE, heat_pumps=sizing)
    )
    partitioned.reset(seed=11)
    grouped = [partitioned.step(partitioned.split_action(a)) for a in actions]
    for (obs, reward, *_), (obs_d, reward_d, *_) in zip(flat, grouped, strict=True):
        assert np.array_equal(obs, obs_d["central"])
        assert reward == reward_d["central"]


@pytest.mark.skipif(not RAW.exists(), reason="run scripts/fetch_when2heat.py first")
def test_with_the_real_cop_buffers_follow_the_seasonal_performance_factor(
    tmp_path,
) -> None:
    """Ground-source pumps get larger buffers than air-source of equal rating."""
    sizing = replace(DEFAULT_HEAT_PUMPS, cache_dir=str(tmp_path))
    e = make_env(seed=1, heat_pumps=sizing)
    ratio = {p.asset_id: p.capacity_mwh / (2.0 * p.ratings.p_max_mw) for p in _hps(e)}
    assert ratio["load:13"] == pytest.approx(4.63, abs=0.01)  # Soil_*
    assert ratio["load:14"] == pytest.approx(3.46, abs=0.01)  # Air_*
