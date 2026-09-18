"""Tests of the agent factory and the policy adapter (M3)."""

from __future__ import annotations

import numpy as np
import pytest

from lvgrid_rl.data.timebase import TimeBase

pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")

from lvgrid_rl.agents.factory import (  # noqa: E402
    SUPPORTED_ALGOS,
    AgentSpec,
    make_agent,
    resolve_gamma,
)
from lvgrid_rl.agents.policy_controller import PolicyController  # noqa: E402

TIMEBASE = TimeBase(sim_dt_min=5, control_dt_min=15)


# ---------------------------------------------------------------------------
# Gamma is derived, not chosen
# ---------------------------------------------------------------------------


def test_gamma_is_derived_from_the_control_step() -> None:
    """The criterion spans a week, so the effective horizon must too."""
    gamma = resolve_gamma(TIMEBASE)
    horizon = 1.0 / (1.0 - gamma)
    assert horizon == pytest.approx(TIMEBASE.control_steps_per_week)


def test_a_habitual_gamma_is_rejected() -> None:
    """``0.99`` is an effective horizon of about 25 hours at a 15-minute cycle.

    With it the agent optimises a problem in which the weekly budget does not
    exist: the policy looks trained but cannot trade excursions off against
    remaining time.
    """
    with pytest.raises(ValueError, match="EN 50160 criterion spans"):
        resolve_gamma(TIMEBASE, requested=0.99)


def test_a_consistent_gamma_is_accepted() -> None:
    assert resolve_gamma(TIMEBASE, requested=0.999) == 0.999


def test_gamma_may_not_hide_in_hyperparams() -> None:
    """Otherwise a sweep could silently override the derivation."""
    with pytest.raises(ValueError, match="belongs in the 'gamma' field"):
        AgentSpec(hyperparams={"gamma": 0.99})


# ---------------------------------------------------------------------------
# Agent specification
# ---------------------------------------------------------------------------


def test_unknown_algorithm_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        AgentSpec(algo="not_an_algo")


def test_registry_covers_the_planned_comparison() -> None:
    """The agent comparison in M7 needs on-policy, off-policy and recurrent."""
    assert {"ppo", "sac", "tqc", "recurrent_ppo"} <= set(SUPPORTED_ALGOS)


def test_factory_builds_a_model_with_the_derived_gamma() -> None:
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    model = make_agent(
        AgentSpec(algo="ppo", hyperparams={"n_steps": 16}), env, TIMEBASE, seed=1
    )
    assert model.gamma == pytest.approx(TIMEBASE.suggested_gamma())
    env.close()


# ---------------------------------------------------------------------------
# Policy adapter
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in; the adapter must not depend on SB3 internals."""

    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.seen: list[np.ndarray] = []

    def predict(self, observation, deterministic: bool = True):
        self.seen.append(np.asarray(observation))
        return self.action, None


def test_policy_controller_uses_the_handed_observation() -> None:
    model = _StubModel(np.array([0.25]))
    controller = PolicyController(model)
    controller.set_observation(np.array([1.0, 2.0]))
    action = controller.act(None)  # type: ignore[arg-type]
    assert action == pytest.approx([0.25])
    assert model.seen[0] == pytest.approx([1.0, 2.0])


def test_policy_controller_fails_loudly_without_an_observation() -> None:
    """Missing observations must fail loudly.

    Silently predicting from a stale one would corrupt an evaluation in a way
    that looks like a weak policy.
    """
    controller = PolicyController(_StubModel(np.zeros(1)))
    with pytest.raises(RuntimeError, match="No observation received"):
        controller.act(None)  # type: ignore[arg-type]


def test_evaluation_is_deterministic_by_default() -> None:
    """Otherwise the reported KPIs mix policy quality with sampling noise."""
    assert PolicyController(_StubModel(np.zeros(1))).deterministic is True


def test_runner_hands_observations_to_controllers_that_want_them() -> None:
    """The seam that lets a policy and a droop rule share one runner."""
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
    from lvgrid_rl.env.factory import make_env
    from lvgrid_rl.eval.runner import run_controller

    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=1, randomise_budget=False)
    env = make_env(seed=1, episode_spec=spec, set_name="stress")
    model = _StubModel(np.zeros(env.mapper.dim))
    controller = PolicyController(model, name="stub")
    run_controller(env, controller, "stub", "stress", n_episodes=1, seed=1)
    assert len(model.seen) > 1
    assert model.seen[0].shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# Baseline parameters for the comparison
# ---------------------------------------------------------------------------


def test_missing_tuned_parameters_stop_the_run() -> None:
    """Falling back to defaults would be the worst of both worlds.

    The run would look like a comparison against tuned references and would not
    be one -- which is exactly how an agent wins against a deliberately weak
    baseline (architecture section 6.6, consequence 5).
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import load_baseline_params

    with pytest.raises(FileNotFoundError, match="tune_baselines"):
        load_baseline_params(Path("does/not/exist.json"), allow_untuned=False)


def test_untuned_comparison_must_be_asked_for_explicitly() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import load_baseline_params

    params, is_tuned = load_baseline_params(
        Path("does/not/exist.json"), allow_untuned=True
    )
    assert is_tuned is False
    assert "fixed_cap" in params


def test_tuned_parameters_are_used_when_present(tmp_path) -> None:
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import load_baseline_params

    path = tmp_path / "tuned.json"
    path.write_text(
        json.dumps({"results": {"fixed_cap": {"params": {"cap": 0.42}}}}),
        encoding="utf-8",
    )
    params, is_tuned = load_baseline_params(path, allow_untuned=False)
    assert is_tuned is True
    assert params["fixed_cap"]["params"]["cap"] == 0.42
