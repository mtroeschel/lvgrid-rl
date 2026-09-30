"""The policy of action mode 2: weights shared per asset kind (decision D14)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")
pytest.importorskip("simbench", reason="extra 'sim' not installed")

import torch as th  # noqa: E402

from lvgrid_rl.agents.factory import AgentSpec, make_agent  # noqa: E402
from lvgrid_rl.agents.shared_policy import SharedAssetPolicy  # noqa: E402
from lvgrid_rl.data.timebase import TimeBase  # noqa: E402
from lvgrid_rl.env.factory import DEFAULT_STORAGE, make_env  # noqa: E402
from lvgrid_rl.env.lv_grid_env import EnvConfig  # noqa: E402
from lvgrid_rl.env.obs import (  # noqa: E402
    AssetBlock,
    ObservationLayout,
    ObservationLayoutMode,
    ObservationSpec,
)

TIMEBASE = TimeBase(sim_dt_min=5, control_dt_min=15)
PER_ASSET = EnvConfig(observation=ObservationSpec(layout=ObservationLayoutMode.PER_ASSET))


def _spec() -> AgentSpec:
    spec = AgentSpec.from_yaml("configs/agent/ppo_shared.yaml")
    return replace(
        spec, hyperparams={**spec.hyperparams, "n_steps": 32, "batch_size": 32}
    )


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=1, storage=DEFAULT_STORAGE, config=PER_ASSET)
    e.reset(seed=1)
    return e


@pytest.fixture(scope="module")
def model(env):
    return make_agent(_spec(), env, TIMEBASE, seed=1)


def _mean_actions(model, obs: np.ndarray) -> np.ndarray:
    tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    with th.no_grad():
        latent = model.policy.mlp_extractor.forward_actor(tensor)
    return latent.squeeze(0).numpy()


def _blocks(layout, kind):
    return layout.blocks_of(kind)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_the_factory_builds_the_shared_policy_from_the_config(model) -> None:
    assert isinstance(model.policy, SharedAssetPolicy)
    assert model.policy.layout == model.get_env().get_attr("observation_layout")[0]


def test_the_flat_layout_is_refused_with_an_explanation() -> None:
    flat = make_env(seed=1, storage=DEFAULT_STORAGE)
    with pytest.raises(ValueError, match="flat layout"):
        make_agent(_spec(), flat, TIMEBASE, seed=1)


def test_the_policy_refuses_to_run_without_a_layout(env) -> None:
    with pytest.raises(ValueError, match="per-asset observation layout"):
        SharedAssetPolicy(
            env.observation_space, env.action_space, lambda _: 3e-4, layout=None
        )


def test_the_action_net_does_not_mix_assets(model) -> None:
    """A dense layer over all means would undo the sharing."""
    assert isinstance(model.policy.action_net, th.nn.Identity)


def test_exploration_noise_is_shared_per_kind(model) -> None:
    assert set(model.policy.kind_log_std) == {"pv", "bess"}
    assert not hasattr(model.policy, "log_std")
    dist = model.policy.get_distribution(
        th.zeros((1, model.observation_space.shape[0]), dtype=th.float32)
    )
    std = dist.distribution.stddev.squeeze(0).detach().numpy()
    assert std.shape == (16,)
    assert np.allclose(std, std[0]), "equal at initialisation, and tied per kind"


def test_initial_actions_are_near_the_centre(model, env) -> None:
    """SB3's small output gain, kept for the per-kind heads."""
    obs, _ = env.reset(seed=1)
    assert np.abs(_mean_actions(model, obs)).max() < 0.2


# ---------------------------------------------------------------------------
# Equivariance and size independence
# ---------------------------------------------------------------------------


def test_swapping_two_assets_swaps_their_actions(model, env) -> None:
    """The defining property: an action follows its asset, not its position."""
    obs, _ = env.reset(seed=1)
    obs = obs + np.random.default_rng(0).normal(0, 0.3, obs.shape).astype(np.float32)
    layout = env.observation_layout
    for kind in ("pv", "bess"):
        a, b = _blocks(layout, kind)[0], _blocks(layout, kind)[3]
        swapped = obs.copy()
        swapped[a.obs_start : a.obs_stop] = obs[b.obs_start : b.obs_stop]
        swapped[b.obs_start : b.obs_stop] = obs[a.obs_start : a.obs_stop]
        before = _mean_actions(model, obs)
        after = _mean_actions(model, swapped)
        assert after[a.action_start] == pytest.approx(before[b.action_start], abs=1e-6)
        assert after[b.action_start] == pytest.approx(before[a.action_start], abs=1e-6)
        others = [
            blk.action_start
            for blk in layout.blocks
            if blk.asset_id not in (a.asset_id, b.asset_id)
        ]
        assert np.allclose(after[others], before[others], atol=1e-6), kind


def _sub_layout(layout: ObservationLayout, per_kind: int) -> ObservationLayout:
    """The same global block with only the first ``per_kind`` assets of each kind."""
    blocks, obs_cursor, act_cursor = [], layout.global_dim, 0
    for kind in layout.kinds:
        for block in layout.blocks_of(kind)[:per_kind]:
            width = block.obs_stop - block.obs_start
            act = block.action_stop - block.action_start
            blocks.append(
                AssetBlock(
                    block.asset_id,
                    kind,
                    obs_cursor,
                    obs_cursor + width,
                    act_cursor,
                    act_cursor + act,
                )
            )
            obs_cursor += width
            act_cursor += act
    return ObservationLayout(global_dim=layout.global_dim, blocks=tuple(blocks))


def test_the_parameters_do_not_depend_on_the_number_of_assets(model, env) -> None:
    """Weights trained on eight batteries load into a grid with three."""
    import gymnasium as gym

    small = _sub_layout(env.observation_layout, 3)
    n_obs = max(b.obs_stop for b in small.blocks)
    n_act = max(b.action_stop for b in small.blocks)
    other = SharedAssetPolicy(
        gym.spaces.Box(-10, 10, (n_obs,), np.float32),
        gym.spaces.Box(-1, 1, (n_act,), np.float32),
        lambda _: 3e-4,
        layout=small,
    )
    assert SharedAssetPolicy.parameter_count(other) == SharedAssetPolicy.parameter_count(
        model.policy
    )
    other.load_state_dict(model.policy.state_dict())  # raises on any shape mismatch


def test_mean_pooling_makes_duplicated_fleets_indistinguishable(model, env) -> None:
    """Twice the same fleet gives the same context, and so the same actions.

    What mean pooling buys: the context describes the fleet's typical state,
    not its size.
    """
    import gymnasium as gym

    layout = env.observation_layout
    obs, _ = env.reset(seed=1)
    base = _sub_layout(layout, 2)
    doubled_blocks, obs_cursor, act_cursor = [], layout.global_dim, 0
    for kind in layout.kinds:
        for _ in range(2):
            for block in base.blocks_of(kind):
                width = block.obs_stop - block.obs_start
                doubled_blocks.append(
                    AssetBlock(
                        block.asset_id,
                        kind,
                        obs_cursor,
                        obs_cursor + width,
                        act_cursor,
                        act_cursor + 1,
                    )
                )
                obs_cursor += width
                act_cursor += 1
    doubled = ObservationLayout(
        global_dim=layout.global_dim, blocks=tuple(doubled_blocks)
    )

    def build(sub):
        n_obs = max(b.obs_stop for b in sub.blocks)
        n_act = max(b.action_stop for b in sub.blocks)
        p = SharedAssetPolicy(
            gym.spaces.Box(-10, 10, (n_obs,), np.float32),
            gym.spaces.Box(-1, 1, (n_act,), np.float32),
            lambda _: 3e-4,
            layout=sub,
        )
        p.load_state_dict(model.policy.state_dict())
        return p

    source = {
        kind: [obs[b.obs_start : b.obs_stop] for b in layout.blocks_of(kind)[:2]]
        for kind in layout.kinds
    }
    base_obs = np.concatenate(
        [obs[: layout.global_dim]] + [x for k in layout.kinds for x in source[k]]
    )
    doubled_obs = np.concatenate(
        [obs[: layout.global_dim]] + [x for k in layout.kinds for x in source[k] * 2]
    )
    with th.no_grad():
        a = build(base).mlp_extractor.forward_actor(th.as_tensor(base_obs).unsqueeze(0))[
            0
        ]
        b = build(doubled).mlp_extractor.forward_actor(
            th.as_tensor(doubled_obs).unsqueeze(0)
        )[0]
    # base: pv0 pv1 bess0 bess1; doubled: pv0 pv1 pv0 pv1 bess0 bess1 bess0 bess1
    assert np.allclose(b[[0, 1]].numpy(), a[[0, 1]].numpy(), atol=1e-6)
    assert np.allclose(b[[4, 5]].numpy(), a[[2, 3]].numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Training and persistence
# ---------------------------------------------------------------------------


def test_training_updates_the_shared_weights(env) -> None:
    model = make_agent(_spec(), env, TIMEBASE, seed=2)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    model.learn(64)
    after = model.policy.state_dict()
    changed = [k for k in before if not th.equal(before[k], after[k])]
    assert any(k.startswith("mlp_extractor.actor_heads.bess") for k in changed)
    assert any(k.startswith("mlp_extractor.actor_heads.pv") for k in changed)
    assert any(k.startswith("kind_log_std") for k in changed)


def test_a_saved_policy_loads_and_acts_the_same(model, env, tmp_path) -> None:
    from stable_baselines3 import PPO

    path = tmp_path / "shared"
    model.save(path)
    loaded = PPO.load(path, device="cpu")
    assert isinstance(loaded.policy, SharedAssetPolicy)
    obs, _ = env.reset(seed=1)
    a, _ = model.predict(obs, deterministic=True)
    b, _ = loaded.predict(obs, deterministic=True)
    assert np.array_equal(a, b)
