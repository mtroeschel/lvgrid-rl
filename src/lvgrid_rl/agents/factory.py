"""Building a Stable-Baselines3 agent from a declarative specification.

Every experiment is described by configuration, not by editing source (§1.2,
principle 2). The factory is the single place where a YAML block becomes a
trainable model, so adding an algorithm is a registry entry rather than a new
training script.

**Gamma is derived, not chosen.** The EN 50160 criterion refers to a weekly
interval, so the effective horizon has to cover one week. A habitual ``0.99``
would be an effective horizon of about 25 hours at a 15-minute control cycle and
would simply not see the criterion (§6.5). The factory therefore derives gamma
from the time base and refuses a value that is inconsistent with it, rather than
accepting whatever a hyperparameter sweep proposes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lvgrid_rl.data.timebase import TimeBase

__all__ = [
    "AgentSpec",
    "CUSTOM_POLICIES",
    "SUPPORTED_ALGOS",
    "make_agent",
    "resolve_gamma",
]

SUPPORTED_ALGOS: Mapping[str, str] = {
    "ppo": "stable_baselines3.PPO",
    "a2c": "stable_baselines3.A2C",
    "sac": "stable_baselines3.SAC",
    "td3": "stable_baselines3.TD3",
    "ddpg": "stable_baselines3.DDPG",
    "tqc": "sb3_contrib.TQC",
    "trpo": "sb3_contrib.TRPO",
    "crossq": "sb3_contrib.CrossQ",
    "recurrent_ppo": "sb3_contrib.RecurrentPPO",
}
"""Algorithm name -> import path. Adding one is a line, not a script."""

CUSTOM_POLICIES: Mapping[str, str] = {
    "SharedAssetPolicy": "lvgrid_rl.agents.shared_policy.SharedAssetPolicy",
}
"""Policies of this project, by the name a configuration file uses."""


def resolve_gamma(timebase: TimeBase, requested: float | None = None) -> float:
    """Derive the discount factor, or validate a requested one.

    Args:
        timebase: Simulation and control step sizes.
        requested: A value from the configuration, or ``None`` to derive.

    Returns:
        The discount factor to use.

    Raises:
        ValueError: if the requested value implies an effective horizon shorter
            than half the criterion horizon. That is not pedantry: with such a
            gamma the agent optimises a problem in which the weekly budget does
            not exist, and the resulting policy looks trained but cannot trade
            excursions off against time.
    """
    derived = timebase.suggested_gamma()
    if requested is None:
        return derived
    horizon = 1.0 / max(1.0 - requested, 1e-12)
    if horizon < 0.5 * timebase.control_steps_per_week:
        raise ValueError(
            f"gamma={requested} gives an effective horizon of {horizon:.0f} "
            f"control steps, but the EN 50160 criterion spans "
            f"{timebase.control_steps_per_week}. Derived value: {derived:.5f}."
        )
    return requested


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Declarative agent description, mirroring ``configs/agent/*.yaml``.

    Args:
        algo: Key from :data:`SUPPORTED_ALGOS`.
        policy: Stable-Baselines3 policy class name.
        hyperparams: Algorithm keyword arguments. ``gamma`` is handled
            separately and must not appear here.
        policy_kwargs: Passed through to the policy.
        gamma: Optional explicit discount factor; validated against the time
            base. Leave unset to derive it.
    """

    algo: str = "ppo"
    policy: str = "MlpPolicy"
    hyperparams: Mapping[str, Any] = field(default_factory=dict)
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    gamma: float | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> AgentSpec:
        """Read a specification from ``configs/agent/*.yaml``.

        Unknown top-level keys are rejected rather than ignored: a misspelt
        ``hyperparam`` would otherwise leave the run on library defaults while
        the file claims otherwise -- the defect M3 had with this very file.
        """
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {"algo", "policy", "hyperparams", "policy_kwargs", "gamma"}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        return cls(**payload)

    def as_config(self) -> dict[str, Any]:
        """Plain dictionary for the run manifest and its configuration hash."""
        return {
            "algo": self.algo,
            "policy": self.policy,
            "hyperparams": dict(self.hyperparams),
            "policy_kwargs": dict(self.policy_kwargs),
            "gamma": self.gamma,
        }

    def __post_init__(self) -> None:
        if self.algo not in SUPPORTED_ALGOS:
            raise ValueError(
                f"Unknown algorithm {self.algo!r}; known: {sorted(SUPPORTED_ALGOS)}"
            )
        if "gamma" in self.hyperparams:
            raise ValueError(
                "gamma belongs in the 'gamma' field, not in hyperparams: it is "
                "derived from the control step size and validated, not swept."
            )


def _import(path: str):
    module_name, _, attribute = path.rpartition(".")
    module = __import__(module_name, fromlist=[attribute])
    return getattr(module, attribute)


def make_agent(
    spec: AgentSpec,
    env,
    timebase: TimeBase,
    seed: int,
    tensorboard_log: str | None = None,
):
    """Instantiate the agent described by ``spec``.

    Args:
        spec: The agent description.
        env: A (vectorised) Gymnasium environment.
        timebase: Used to derive or validate gamma.
        seed: Training seed.
        tensorboard_log: Directory for TensorBoard events, or ``None``.
    """
    algo_class = _import(SUPPORTED_ALGOS[spec.algo])
    gamma = resolve_gamma(timebase, spec.gamma)
    policy: Any = spec.policy
    policy_kwargs = dict(spec.policy_kwargs)
    if spec.policy in CUSTOM_POLICIES:
        policy = _import(CUSTOM_POLICIES[spec.policy])
        # The layout describes the grid, not the policy, so it comes from the
        # environment rather than from the configuration file.
        policy_kwargs["layout"] = _observation_layout(env)
    return algo_class(
        policy,
        env,
        gamma=gamma,
        seed=seed,
        verbose=0,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs or None,
        **dict(spec.hyperparams),
    )


def _observation_layout(env):
    """The per-asset layout of a plain or vectorised environment."""
    if hasattr(env, "get_attr"):  # a VecEnv, possibly in worker processes
        return env.get_attr("observation_layout")[0]
    return env.unwrapped.observation_layout
