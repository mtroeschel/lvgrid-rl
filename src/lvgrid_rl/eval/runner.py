"""Running a controller over an evaluation set and collecting KPIs.

The runner knows only :class:`~lvgrid_rl.core.protocols.Controller`, so a
rule-based method, an OPF, an MPC and a trained policy all go through the same
path. That is what makes the comparison structurally fair rather than a matter of
discipline.

KPIs are reported **per set** -- test, stress weeks, held-out month -- and never
aggregated across them. A blended pass rate of 0.85 cannot be read: it could mean
failure on the one extreme week or a little failure everywhere (§6.5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from lvgrid_rl.env.lv_grid_env import LVGridEnv

__all__ = ["EpisodeResult", "RunResult", "run_controller"]


class _Controller(Protocol):
    def reset(self, info) -> None: ...
    def act(self, info) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """KPIs of one episode.

    The quantities are physical and unweighted. The comparison against reference
    methods runs on these, never on the reward -- otherwise the policy whose
    objective one defined oneself wins trivially.
    """

    week: tuple[int, int]
    steps: int
    reward: float
    curtailed_energy_mwh: float
    k95_violating_windows: int
    k100_violating_windows: int
    overload_cost: float
    diverged_steps: int
    clipped_steps: int
    decision_time_ms: float


@dataclass(frozen=True, slots=True)
class RunResult:
    """Aggregate over the episodes of one set."""

    controller: str
    set_name: str
    episodes: tuple[EpisodeResult, ...]

    @property
    def n_episodes(self) -> int:
        """Number of episodes run."""
        return len(self.episodes)

    def total(self, field: str) -> float:
        """Sum of one field over all episodes."""
        return float(sum(getattr(e, field) for e in self.episodes))

    def mean(self, field: str) -> float:
        """Mean of one field over all episodes."""
        return self.total(field) / max(self.n_episodes, 1)

    def summary(self) -> dict[str, float | str]:
        """Flat summary for a results table."""
        return {
            "controller": self.controller,
            "set": self.set_name,
            "episodes": self.n_episodes,
            "reward": self.total("reward"),
            "curtailed_mwh": self.total("curtailed_energy_mwh"),
            "k95_windows": self.total("k95_violating_windows"),
            "k100_windows": self.total("k100_violating_windows"),
            "overload_cost": self.total("overload_cost"),
            "diverged_steps": self.total("diverged_steps"),
            "decision_time_ms": self.mean("decision_time_ms"),
        }

    def as_records(self) -> list[dict]:
        """One record per episode, for writing to Parquet."""
        return [
            {"controller": self.controller, "set": self.set_name, **asdict(e)}
            for e in self.episodes
        ]


def run_controller(
    env: LVGridEnv,
    controller: _Controller,
    name: str,
    set_name: str,
    n_episodes: int | None = None,
    seed: int = 0,
) -> RunResult:
    """Run a controller over the episodes of one set.

    Args:
        env: An environment whose sampler is bound to the desired set. In
            evaluation mode the sampler returns complete weeks in a fixed order,
            so every controller sees exactly the same episodes.
        controller: Anything implementing the controller protocol.
        name: Label for the results table.
        set_name: The evaluation set, for the results table.
        n_episodes: How many episodes to run; defaults to one per week in the
            set, which is the standard-conforming choice.
        seed: Seed handed to ``reset``.
    """
    import time

    env.sampler.reset_cursor()
    episodes: list[EpisodeResult] = []
    total = n_episodes if n_episodes is not None else env.sampler.n_weeks

    for index in range(total):
        _, info = env.reset(seed=seed + index)
        state = env._information_set(env._t, env._last_grid)  # noqa: SLF001
        controller.reset(state)

        reward = 0.0
        curtailed = 0.0
        k95 = 0
        k100 = 0
        overload = 0.0
        diverged = 0
        clipped = 0
        decision_ns = 0
        steps = 0
        week = info["week"]

        while True:
            state = env._information_set(env._t, env._last_grid)  # noqa: SLF001
            started = time.perf_counter_ns()
            action = controller.act(state)
            decision_ns += time.perf_counter_ns() - started

            _, step_reward, terminated, truncated, step_info = env.step(action)
            reward += step_reward
            curtailed += step_info["curtailed_energy_mwh"]
            k95 += step_info["k95_violations"]
            k100 += step_info["k100_violations"]
            overload += step_info.get("cost/thermal_overload", 0.0)
            diverged += int(step_info["pf_diverged"])
            clipped += int(step_info["action_clipped"])
            steps += 1
            if terminated or truncated:
                break

        episodes.append(
            EpisodeResult(
                week=week,
                steps=steps,
                reward=reward,
                curtailed_energy_mwh=curtailed,
                k95_violating_windows=k95,
                k100_violating_windows=k100,
                overload_cost=overload,
                diverged_steps=diverged,
                clipped_steps=clipped,
                decision_time_ms=decision_ns / 1e6 / max(steps, 1),
            )
        )

    return RunResult(controller=name, set_name=set_name, episodes=tuple(episodes))
