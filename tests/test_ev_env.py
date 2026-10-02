"""The environment with controllable charge points (M4, step 4.4b part 2, D17).

The sessions come from a synthetic emobpy run written to a temporary directory,
so these tests run without the generated data (in CI); one test checks the
dataset of record when it is present.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("simbench", reason="extra 'sim' not installed")
pytest.importorskip("gymnasium", reason="extra 'env' not installed")

from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec  # noqa: E402
from lvgrid_rl.env.factory import DEFAULT_EV, EvSizing, make_env  # noqa: E402
from lvgrid_rl.env.lv_grid_env import EnvConfig  # noqa: E402
from lvgrid_rl.env.obs import ObservationLayoutMode, ObservationSpec  # noqa: E402

EVALUATE = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
CHARGE_POINTS = json.loads(
    Path("configs/ev/charge_points.json").read_text(encoding="utf-8")
)["charge_points"]


def _vehicle_year(kw: float) -> pd.DataFrame:
    """Home from 18:00 to 07:00 local every day, drawing ``kw`` for an hour."""
    times = pd.date_range("2015-12-28 00:00", "2017-01-01 23:45", freq="15min")
    hour = times.hour
    home = (hour >= 18) | (hour < 7)
    grid = np.where((hour >= 18) & (hour < 19), kw, 0.0)
    return pd.DataFrame(
        {
            "datetime_local": times,
            "state": np.where(home, "home", "workplace"),
            "charging_point": np.where(home, "home", "none"),
            "charge_grid": grid,
        }
    )


@pytest.fixture(scope="module")
def sizing(tmp_path_factory) -> EvSizing:
    """A synthetic run for the grid's seven charge points, own manifest."""
    root = tmp_path_factory.mktemp("emobpy_run")
    entries = []
    for point in CHARGE_POINTS:
        path = root / f"{point['asset_id'].replace(':', '_')}.parquet"
        _vehicle_year(min(point["nominal_power_kw"], 3.7)).to_parquet(path, index=False)
        entries.append(
            {
                **point,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "battery_capacity_kwh": 50.0,
                "driver": "fulltime",
                "vehicle": ["Synthetic", "Car", 2020],
                "weeks": [],
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"charge_points": entries}), encoding="utf-8"
    )
    return EvSizing(run_dir=str(root), manifest=None)


@pytest.fixture(scope="module")
def env(sizing):
    e = make_env(seed=1, ev=sizing)
    e.reset(seed=1)
    return e


def _evs(env):
    return [a for a in env.assets if a.kind == "ev"]


def _week(sizing, **kwargs):
    """The first test week in evaluation mode.

    It starts at Monday 00:00 local, in the middle of a stay at home.
    """
    e = make_env(seed=1, set_name="test", episode_spec=EVALUATE, ev=sizing, **kwargs)
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


class _AllOff:
    """Offers nothing at any charge point; everything else neutral."""

    name = "all_off"

    def __init__(self, env) -> None:
        self.env = env

    def reset(self, info) -> None:
        pass

    def act(self, info):
        physical = self.env.mapper.default_physical(info)
        physical[self.env.mapper.component_mask("ev", "p_mw")] = 0.0
        return self.env.mapper.to_normalised(physical)


# ---------------------------------------------------------------------------
# Placement and data path
# ---------------------------------------------------------------------------


def test_the_grid_charge_points_become_controllable_at_nominal_power(env) -> None:
    chargers = {a.asset_id: a for a in _evs(env)}
    assert set(chargers) == {p["asset_id"] for p in CHARGE_POINTS}
    for point in CHARGE_POINTS:
        charger = chargers[point["asset_id"]]
        assert charger.bus == point["bus"]
        assert charger.ratings.p_max_mw == pytest.approx(point["nominal_power_kw"] / 1e3)


def test_their_profiles_no_longer_drive_the_grid(env) -> None:
    uncontrolled = set(env._uncontrolled)  # noqa: SLF001
    assert not uncontrolled & {a.asset_id for a in _evs(env)}


def test_the_sessions_carry_the_simbench_energy_of_each_curve(env) -> None:
    hours = env.config.sim_dt_min / 60.0
    frame = env._profiles  # noqa: SLF001
    for charger in _evs(env):
        column = env.series_ids.index(charger.asset_id)
        target = frame[:, column].sum() * hours
        sessions = env._ev_sessions[charger.asset_id]  # noqa: SLF001
        assert sum(s.energy_mwh for s in sessions) == pytest.approx(target, rel=1e-9)


def test_sessions_reach_the_charge_point_by_the_exogenous_input(env) -> None:
    """Through ExogenousInput.ev_sessions only, never a table the model holds."""
    charger = _evs(env)[0]
    sessions = env._ev_sessions[charger.asset_id]  # noqa: SLF001
    session = sessions[len(sessions) // 2]
    inside = env._exogenous(session.arrival_t_index + 1)  # noqa: SLF001
    assert inside.ev_sessions[charger.asset_id] == session
    outside = env._exogenous(session.departure_t_index)  # noqa: SLF001
    assert charger.asset_id not in outside.ev_sessions
    # The model holds parameters only: no session data it could look ahead in.
    assert {f.name for f in fields(charger)} == {"asset_id", "bus", "ratings"}


def test_an_episode_starting_in_a_stay_connects_the_vehicle_pro_rata(sizing) -> None:
    e = _week(sizing)
    for charger in _evs(e):
        state = e._asset_states[charger.asset_id]  # noqa: SLF001
        session = e._exogenous(e._t).ev_sessions[charger.asset_id]  # noqa: SLF001
        assert state.connected
        ahead = (session.departure_t_index - e._t) / (  # noqa: SLF001
            session.departure_t_index - session.arrival_t_index
        )
        assert state.need_mwh == pytest.approx(session.energy_mwh * ahead)


def test_without_the_option_charge_points_stay_uncontrolled_load() -> None:
    plain = make_env(seed=1)
    assert not [a for a in plain.assets if a.kind == "ev"]
    assert CHARGE_POINTS[0]["asset_id"] in plain._uncontrolled  # noqa: SLF001


def test_a_missing_run_says_how_to_make_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--replay"):
        make_env(seed=1, ev=EvSizing(run_dir=str(tmp_path / "none"), manifest=None))


# ---------------------------------------------------------------------------
# Uncontrolled charging, unserved energy and its price
# ---------------------------------------------------------------------------


def test_do_nothing_charges_on_arrival_and_serves_everything(sizing) -> None:
    from lvgrid_rl.baselines.methods import DoNothing

    infos = _rollout(_week(sizing), lambda e: DoNothing(e.mapper), steps=192)
    assert sum(i["ev_charged_mwh"] for i in infos) > 0.0
    assert sum(i["ev_unserved_mwh"] for i in infos) == pytest.approx(0.0, abs=1e-12)
    assert not any(k.startswith("clipping/ev/") for i in infos for k in i)


def test_offering_nothing_leaves_energy_unserved_and_is_priced(sizing) -> None:
    infos = _rollout(_week(sizing), _AllOff, steps=192)
    unserved = sum(i["ev_unserved_mwh"] for i in infos)
    assert unserved > 0.0
    assert sum(i["ev_charged_mwh"] for i in infos) == 0.0
    assert sum(i["ev_sessions_short"] for i in infos) >= len(CHARGE_POINTS)
    assert sum(i["reward/ev_unserved"] for i in infos) == pytest.approx(-100.0 * unserved)


def test_the_runner_reports_charge_point_kpis(sizing) -> None:
    from lvgrid_rl.eval.runner import run_controller

    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=1, randomise_budget=False)
    e = make_env(seed=1, set_name="test", episode_spec=spec, ev=sizing)
    summary = run_controller(e, _AllOff(e), "all_off", "test", n_episodes=1).summary()
    for key in ("ev_unserved_mwh", "ev_charged_mwh", "ev_sessions_short"):
        assert key in summary
    assert summary["ev_unserved_mwh"] > 0.0


# ---------------------------------------------------------------------------
# Observation and the policies on top of it
# ---------------------------------------------------------------------------


def test_the_flat_observation_holds_connection_need_and_time_left(sizing) -> None:
    e = _week(sizing)
    obs, _ = e.reset(seed=1)
    names = e.obs_builder.feature_names
    charger = _evs(e)[0]
    state = e._asset_states[charger.asset_id]  # noqa: SLF001
    values = {n: obs[names.index(n)] for n in names if charger.asset_id in n}
    prefix = f"asset/{charger.asset_id}"
    assert values[f"{prefix}/connected"] == 1.0
    need_h = state.need_mwh / charger.ratings.p_max_mw
    assert values[f"{prefix}/need"] == pytest.approx(need_h / 4.0, rel=1e-5)
    assert values[f"{prefix}/time_left"] == pytest.approx(
        state.remaining_min / 60.0 / 24.0, rel=1e-5
    )


def test_per_asset_blocks_include_the_charge_points(sizing) -> None:
    config = EnvConfig(
        observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET)
    )
    e = make_env(seed=1, set_name="test", episode_spec=EVALUATE, ev=sizing, config=config)
    obs, _ = e.reset(seed=1)
    layout = e.observation_layout
    assert set(layout.kinds) == {"pv", "ev"}
    block = layout.blocks_of("ev")[0]
    names = [
        n.split("/", 2)[2]
        for n in e.obs_builder.feature_names[block.obs_start : block.obs_stop]
    ]
    values = dict(zip(names, obs[block.obs_start : block.obs_stop], strict=True))
    assert values["connected"] == 1.0
    assert values["laxity"] == pytest.approx(
        values["time_left"] - values["need"] * 4.0 / 24.0, abs=1e-6
    )


def test_the_shared_policy_builds_with_charge_points(sizing) -> None:
    pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")
    from lvgrid_rl.agents.factory import AgentSpec, make_agent
    from lvgrid_rl.data.timebase import TimeBase

    config = EnvConfig(
        observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET)
    )
    e = make_env(seed=1, ev=sizing, config=config)
    spec = AgentSpec.from_yaml("configs/agent/ppo_shared.yaml")
    spec = replace(
        spec, hyperparams={**spec.hyperparams, "n_steps": 32, "batch_size": 32}
    )
    model = make_agent(spec, e, TimeBase(5, 15), seed=1)
    assert set(model.policy.kind_log_std) == {"pv", "ev"}
    model.learn(32)


def test_the_environment_passes_the_gymnasium_checker(env) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.skipif(
    not Path(DEFAULT_EV.run_dir).exists(),
    reason="reproduce the run first: tools/emobpy/generate.py --replay",
)
def test_the_dataset_of_record_builds_and_matches_its_manifest() -> None:
    e = make_env(seed=1, ev=DEFAULT_EV)
    sessions = e._ev_sessions  # noqa: SLF001
    assert len(sessions) == 7
    assert 240 <= min(len(s) for s in sessions.values())
    assert max(len(s) for s in sessions.values()) <= 300
