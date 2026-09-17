"""Tests of the reference methods, the evaluation runner and the multi-agent view."""

from __future__ import annotations

import numpy as np
import pytest

from lvgrid_rl.baselines.methods import (
    BASELINES,
    DoNothing,
    FixedCap,
    PUDroop,
    QUDroop,
    RandomController,
)
from lvgrid_rl.core.protocols import ActionSpec
from lvgrid_rl.core.schemas import Interval, PQBudgetState
from lvgrid_rl.env.actions import ActionMapper

pytest.importorskip("gymnasium", reason="extra 'env' not installed")


def _mapper(n: int = 3, with_q: bool = False) -> ActionMapper:
    if with_q:
        spec = ActionSpec(
            names=("p_mw", "q_mvar"),
            bounds=(Interval(-0.02, 0.0), Interval(-0.01, 0.01)),
        )
    else:
        spec = ActionSpec(names=("p_mw",), bounds=(Interval(-0.02, 0.0),))
    return ActionMapper(tuple(f"sgen:{i}" for i in range(n)), (spec,) * n)


def _info(voltages: list[float]):
    from datetime import UTC, datetime

    from lvgrid_rl.core.information import InformationSet

    return InformationSet(
        t_index=0,
        timestamp=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
        measurements={"vm_pu": np.array(voltages)},
        asset_states={},
        series_ids=(),
        exogenous_bounds_mw=np.zeros((2, 0)),
        forecast={},
        pq=PQBudgetState(windows_elapsed_count=0),
    )


# ---------------------------------------------------------------------------
# B0 to B4
# ---------------------------------------------------------------------------


def test_all_methods_are_registered() -> None:
    assert set(BASELINES) == {
        "do_nothing",
        "random",
        "fixed_cap",
        "p_u_droop",
        "q_u_droop",
    }


def test_do_nothing_means_full_infeed() -> None:
    """The sign trap: for PV the upper action bound is zero infeed."""
    mapper = _mapper()
    action = DoNothing(mapper).act(_info([1.0, 1.0, 1.0]))
    assert np.allclose(mapper.to_physical(action), mapper.lower)


def test_fixed_cap_curtails_to_a_share_of_rated_power() -> None:
    mapper = _mapper()
    action = FixedCap(mapper, cap=0.5).act(_info([1.0] * 3))
    assert np.allclose(mapper.to_physical(action), 0.5 * mapper.lower)


def test_random_controller_is_reproducible() -> None:
    """A sanity check must itself be reproducible, or it cannot be compared."""
    mapper = _mapper()
    a = RandomController(mapper, seed=3)
    first = a.act(_info([1.0] * 3))
    a.reset(_info([1.0] * 3))
    assert np.allclose(a.act(_info([1.0] * 3)), first)


def test_p_u_droop_acts_only_above_the_deadband() -> None:
    mapper = _mapper()
    droop = PUDroop(mapper, asset_bus_positions=(0, 1, 2), v_start=1.05, v_max=1.10)
    physical = mapper.to_physical(droop.act(_info([1.00, 1.075, 1.20])))
    assert physical[0] == pytest.approx(mapper.lower[0]), "below deadband: no action"
    assert physical[1] == pytest.approx(0.5 * mapper.lower[1]), "half way: half power"
    assert physical[2] == pytest.approx(0.0), "far above: full curtailment"


def test_p_u_droop_uses_the_local_voltage() -> None:
    """Each system reacts to its own bus, not to the worst bus in the grid."""
    mapper = _mapper()
    droop = PUDroop(mapper, asset_bus_positions=(2, 1, 0), v_start=1.05, v_max=1.10)
    physical = mapper.to_physical(droop.act(_info([1.20, 1.00, 1.00])))
    assert physical[0] == pytest.approx(mapper.lower[0])
    assert physical[2] == pytest.approx(0.0)


def test_droop_rejects_an_inverted_characteristic() -> None:
    with pytest.raises(ValueError, match="v_start must be below v_max"):
        PUDroop(_mapper(), (0, 1, 2), v_start=1.10, v_max=1.05)


def test_q_u_droop_requires_a_reactive_action_component() -> None:
    """Otherwise B3 would silently do nothing and look like a weak baseline."""
    with pytest.raises(ValueError, match="pv_mode='pq'"):
        QUDroop(_mapper(with_q=False), (0, 1, 2))


def test_q_u_droop_absorbs_reactive_power_above_the_deadband() -> None:
    mapper = _mapper(with_q=True)
    droop = QUDroop(mapper, (0, 1, 2), v_start=1.04, v_max=1.10)
    physical = mapper.to_physical(droop.act(_info([1.00, 1.07, 1.20])))
    # Layout is (p, q) per asset; active power stays at full infeed throughout.
    assert physical[0] == pytest.approx(mapper.lower[0])
    assert physical[1] == pytest.approx(0.0), "below deadband: no reactive power"
    assert physical[3] == pytest.approx(0.005), "half way: half capability"
    assert physical[5] == pytest.approx(0.01), "far above: full capability"


# ---------------------------------------------------------------------------
# Multi-agent view -- rule 4 of section 6.3
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.env.factory import make_env

    return make_env(seed=1)


def test_single_partition_reproduces_the_flat_environment_exactly(env) -> None:
    """The regression test that keeps the two paths from drifting apart.

    A centralised single agent is a partitioned environment with one partition.
    If that stops being bit-identical, the multi-agent path has silently become
    a second implementation.
    """
    from lvgrid_rl.env.factory import make_env
    from lvgrid_rl.env.multiagent import PartitionedEnv

    actions = [
        np.full(env.mapper.dim, value, dtype=np.float64)
        for value in (-1.0, -0.25, 0.5, 1.0, 0.0)
    ]

    flat_env = make_env(seed=11)
    flat_env.reset(seed=11)
    flat = [flat_env.step(a) for a in actions]

    partitioned = PartitionedEnv(make_env(seed=11))
    partitioned.reset(seed=11)
    grouped = [partitioned.step(partitioned.split_action(a)) for a in actions]

    for (obs, reward, _, _, _), (obs_d, reward_d, _, _, _) in zip(
        flat, grouped, strict=True
    ):
        assert np.array_equal(obs, obs_d["central"])
        assert reward == reward_d["central"]


def test_per_asset_partition_covers_every_component(env) -> None:
    from lvgrid_rl.env.multiagent import PartitionedEnv, partition_by_asset

    partitioned = PartitionedEnv(env, partition_by_asset(env))
    assert partitioned.agents == env.mapper.asset_ids
    assert len(partitioned.agents) == len(env.assets)


def test_incomplete_partition_is_rejected(env) -> None:
    """A partition that leaves a component unassigned would silently freeze it."""
    from lvgrid_rl.env.multiagent import AgentPartition, PartitionedEnv

    with pytest.raises(ValueError, match="exactly once"):
        PartitionedEnv(env, [AgentPartition("a", (0,))])


def test_wrong_action_length_is_rejected(env) -> None:
    from lvgrid_rl.env.multiagent import PartitionedEnv

    partitioned = PartitionedEnv(env)
    partitioned.reset(seed=1)
    with pytest.raises(ValueError, match="supplied"):
        partitioned.step({"central": np.zeros(2)})


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def test_runner_reports_physical_kpis(env) -> None:
    """The comparison against reference methods runs on KPIs, never on reward."""
    from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
    from lvgrid_rl.env.factory import make_env
    from lvgrid_rl.eval.runner import run_controller

    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=1, randomise_budget=False)
    local = make_env(seed=1, episode_spec=spec, set_name="stress")
    result = run_controller(
        local, DoNothing(local.mapper), "do_nothing", "stress", n_episodes=1, seed=1
    )
    summary = result.summary()
    assert summary["controller"] == "do_nothing"
    assert summary["episodes"] == 1
    assert summary["curtailed_mwh"] >= 0.0
    assert summary["decision_time_ms"] > 0.0
    assert result.as_records()[0]["set"] == "stress"


def test_every_controller_sees_the_same_episodes(env) -> None:
    """Every controller sees the same episodes.

    A difference in KPIs is then a difference in control rather than in luck.
    """
    from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
    from lvgrid_rl.env.factory import make_env
    from lvgrid_rl.eval.runner import run_controller

    spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
    weeks = []
    for controller_cls in (DoNothing, lambda m: FixedCap(m, cap=0.6)):
        local = make_env(seed=1, episode_spec=spec, set_name="stress")
        local.sampler._weeks = local.sampler._weeks[:1]
        result = run_controller(
            local, controller_cls(local.mapper), "c", "stress", n_episodes=1, seed=1
        )
        weeks.append(result.episodes[0].week)
    assert weeks[0] == weeks[1]
