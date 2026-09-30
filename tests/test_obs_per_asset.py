"""The per-asset observation layout (M4, step 4.2b, first part).

What a policy with weights shared per asset kind needs from the observation: a
global block, then one block per asset of a fixed width per kind, in action
order. The properties tested here are the ones such a policy silently relies on.
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("simbench", reason="extra 'sim' not installed")
pytest.importorskip("gymnasium", reason="extra 'env' not installed")

from lvgrid_rl.env.factory import DEFAULT_STORAGE, make_env  # noqa: E402
from lvgrid_rl.env.lv_grid_env import EnvConfig  # noqa: E402
from lvgrid_rl.env.obs import (  # noqa: E402
    AssetFeatureSpec,
    ObservationBuilder,
    ObservationLayoutMode,
    ObservationSpec,
    SensorConfig,
    observation_layout,
)

PER_ASSET = EnvConfig(observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET))


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=1, storage=DEFAULT_STORAGE, config=PER_ASSET)
    e.reset(seed=1)
    return e


def _state_and_info(env):
    grid = env._last_grid  # noqa: SLF001
    return env._system_state(env._t, grid), env._information_set(env._t, grid)  # noqa: SLF001


def _block(env, obs, asset_id):
    block = next(b for b in env.observation_layout.blocks if b.asset_id == asset_id)
    names = env.obs_builder.feature_names[block.obs_start : block.obs_stop]
    values = obs[block.obs_start : block.obs_stop]
    return {
        name.split("/", 2)[2]: float(v) for name, v in zip(names, values, strict=True)
    }


# ---------------------------------------------------------------------------
# The flat layout stays what it was
# ---------------------------------------------------------------------------


def test_flat_is_the_default_and_has_no_asset_blocks() -> None:
    plain = make_env(seed=1, storage=DEFAULT_STORAGE)
    assert plain.obs_builder.spec.layout is ObservationLayoutMode.FLAT
    assert not [n for n in plain.obs_builder.feature_names if n.endswith("/vm_local")]
    with pytest.raises(ValueError, match="flat layout"):
        plain.obs_builder.layout()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_blocks_tile_the_vector_after_the_global_part(env) -> None:
    layout = env.observation_layout
    cursor = layout.global_dim
    for block in layout.blocks:
        assert block.obs_start == cursor
        cursor = block.obs_stop
    assert cursor == env.observation_space.shape[0]


def test_blocks_are_in_action_order_and_cover_the_action(env) -> None:
    """Block ``i`` must produce the action of asset ``i``."""
    layout = env.observation_layout
    assert tuple(b.asset_id for b in layout.blocks) == env.mapper.asset_ids
    cursor = 0
    for block in layout.blocks:
        assert block.action_start == cursor
        cursor = block.action_stop
    assert cursor == env.mapper.dim


def test_every_block_of_a_kind_has_the_same_width(env) -> None:
    """The precondition for sharing one network across the assets of a kind."""
    layout = env.observation_layout
    assert set(layout.kinds) == {"pv", "bess"}
    for kind in layout.kinds:
        widths = {b.obs_stop - b.obs_start for b in layout.blocks_of(kind)}
        assert len(widths) == 1, kind


def test_forecasts_and_soc_move_into_the_blocks(env) -> None:
    global_names = env.obs_builder.feature_names[: env.observation_layout.global_dim]
    assert not [n for n in global_names if n.startswith(("forecast/", "asset/"))]


def test_the_global_width_does_not_depend_on_the_assets(env) -> None:
    """Invariant to the number of assets -- not to the number of buses."""
    spec = env.obs_builder.spec
    fewer = ObservationBuilder(
        replace(spec, assets=spec.assets[:3]), env.obs_builder.n_evaluated_buses
    )
    assert fewer.layout().global_dim == env.observation_layout.global_dim


def test_the_layout_pickles_with_a_policy(env) -> None:
    layout = env.observation_layout
    assert pickle.loads(pickle.dumps(layout)) == layout


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_local_voltage_is_the_measurement_at_the_own_bus(env) -> None:
    state, info = _state_and_info(env)
    obs = env.obs_builder.build(state, info)
    order = list(env.model.connection_point_buses)
    vm = np.asarray(info.measurements["vm_pu"])
    for asset in env.assets:
        expected = (vm[order.index(asset.bus)] - 1.0) / 0.10
        assert _block(env, obs, asset.asset_id)["vm_local"] == pytest.approx(
            expected, abs=1e-6
        )


def test_pv_block_holds_its_forecast_relative_to_its_rating(env) -> None:
    state, info = _state_and_info(env)
    obs = env.obs_builder.build(state, info)
    for asset in (a for a in env.assets if a.kind == "pv"):
        block = _block(env, obs, asset.asset_id)
        rated = -asset.ratings.p_min_mw
        for h in range(4):
            expected = info.forecast[asset.series_id][h] / rated
            assert block[f"forecast/{h}"] == pytest.approx(expected, abs=1e-6)
        assert block["rated_p"] == pytest.approx(rated / 0.1, rel=1e-6)


def test_battery_block_holds_soc_and_the_colocated_pv(env) -> None:
    state, info = _state_and_info(env)
    obs = env.obs_builder.build(state, info)
    pv_by_bus = {a.bus: a for a in env.assets if a.kind == "pv"}
    for battery in (a for a in env.assets if a.kind == "bess"):
        block = _block(env, obs, battery.asset_id)
        assert block["soc_frac"] == pytest.approx(
            battery.soc_frac(state.assets[battery.asset_id]), abs=1e-6
        )
        assert block["energy_to_power"] == pytest.approx(2.0 / 24.0, rel=1e-6)
        pv = pv_by_bus[battery.bus]
        expected = info.forecast[pv.series_id][0] / battery.ratings.p_max_mw
        assert block["colocated_pv_forecast/0"] == pytest.approx(expected, abs=1e-6)


def test_last_setpoint_is_the_executed_one_relative_to_the_rating() -> None:
    """What was executed, not what was asked for.

    The request is half of rated charging power; a battery that starts nearly
    full is limited below that, and the block must show the limited value.
    """
    e = make_env(seed=1, storage=DEFAULT_STORAGE, config=PER_ASSET)
    e.reset(seed=1)
    physical = e.mapper.neutral.copy()
    mask = e.mapper.component_mask("bess", "p_mw")
    physical[mask] = 0.5 * e.mapper.upper[mask]
    obs, *_ = e.step(e.mapper.to_normalised(physical))
    observed = []
    for battery in (a for a in e.assets if a.kind == "bess"):
        executed = e._asset_states[battery.asset_id].last_p_mw  # noqa: SLF001
        value = _block(e, obs, battery.asset_id)["last_p"]
        assert value == pytest.approx(executed / battery.ratings.p_max_mw, abs=1e-6)
        observed.append(value)
    assert max(observed) == pytest.approx(0.5, abs=1e-6), "unlimited ones at the request"


# ---------------------------------------------------------------------------
# The properties a shared network relies on
# ---------------------------------------------------------------------------


def test_reordering_the_assets_permutes_the_blocks_and_nothing_else(env) -> None:
    """Equivariance: the global part is the same, the blocks move with the assets.

    A shared network is then equivariant too, and an asset's action depends on
    its own block rather than on where it stands in the list.
    """
    spec = env.obs_builder.spec
    reordered = ObservationBuilder(
        replace(spec, assets=tuple(reversed(spec.assets))),
        env.obs_builder.n_evaluated_buses,
    )
    state, info = _state_and_info(env)
    original = env.obs_builder.build(state, info)
    permuted = reordered.build(state, info)
    g = env.observation_layout.global_dim
    assert np.array_equal(original[:g], permuted[:g])
    for block in env.observation_layout.blocks:
        other = next(b for b in reordered.layout().blocks if b.asset_id == block.asset_id)
        assert np.array_equal(
            original[block.obs_start : block.obs_stop],
            permuted[other.obs_start : other.obs_stop],
        )


def test_blocks_out_of_action_order_are_refused(env) -> None:
    spec = env.obs_builder.spec
    reordered = ObservationBuilder(
        replace(spec, assets=tuple(reversed(spec.assets))),
        env.obs_builder.n_evaluated_buses,
    )
    with pytest.raises(ValueError, match="action order"):
        observation_layout(reordered, env.mapper)


def test_blocks_are_the_same_under_realistic_sensing() -> None:
    """The local voltage is a house connection value, available in both."""
    realistic = EnvConfig(
        observation=ObservationSpec(
            layout=ObservationLayoutMode.PER_ASSET,
            sensor_config=SensorConfig.REALISTIC,
            measured_buses=(0, 12),
        )
    )
    full = make_env(seed=1, storage=DEFAULT_STORAGE, config=PER_ASSET)
    limited = make_env(seed=1, storage=DEFAULT_STORAGE, config=realistic)
    obs_full, _ = full.reset(seed=1)
    obs_limited, _ = limited.reset(seed=1)
    assert limited.observation_layout.global_dim < full.observation_layout.global_dim
    for a, b in zip(
        full.observation_layout.blocks, limited.observation_layout.blocks, strict=True
    ):
        assert np.array_equal(
            obs_full[a.obs_start : a.obs_stop], obs_limited[b.obs_start : b.obs_stop]
        )


def test_an_asset_kind_without_features_is_refused() -> None:
    with pytest.raises(ValueError, match="no per-asset features"):
        AssetFeatureSpec(
            asset_id="x:0", kind="heat_pump", bus_position=0, rated_p_mw=0.01
        )


def test_the_environment_passes_the_gymnasium_checker(env) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


def test_single_partition_reproduces_the_flat_environment_per_asset() -> None:
    """Rule 4 of section 6.3 holds in the per-asset layout too."""
    from lvgrid_rl.env.multiagent import PartitionedEnv

    probe = make_env(seed=11, storage=DEFAULT_STORAGE, config=PER_ASSET)
    actions = [np.full(probe.mapper.dim, v) for v in (-1.0, 0.3, 1.0)]
    flat_env = make_env(seed=11, storage=DEFAULT_STORAGE, config=PER_ASSET)
    flat_env.reset(seed=11)
    flat = [flat_env.step(a) for a in actions]
    partitioned = PartitionedEnv(
        make_env(seed=11, storage=DEFAULT_STORAGE, config=PER_ASSET)
    )
    partitioned.reset(seed=11)
    grouped = [partitioned.step(partitioned.split_action(a)) for a in actions]
    for (obs, reward, *_), (obs_d, reward_d, *_) in zip(flat, grouped, strict=True):
        assert np.array_equal(obs, obs_d["central"])
        assert reward == reward_d["central"]
