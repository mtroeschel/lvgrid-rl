"""Partitioned view of the environment, for the later multi-agent path.

Rule 4 of §6.3: a multi-agent view exists from M3, even unused, together with a
regression test requiring that the single-agent environment and a multi-agent
environment with one partition produce bit-identical trajectories given identical
actions. That test is the only reliable protection against the two paths drifting
apart over months.

**Deviation from the document, stated plainly.** The architecture names a
``PettingZooAdapter``. This is a dependency-free partitioned view with the same
shape as a PettingZoo ``ParallelEnv`` -- dictionary actions and observations keyed
by agent -- but without importing PettingZoo, because adding a dependency for an
interface nothing uses yet buys nothing. The typed PettingZoo shell is a thin
wrapper over this class and belongs in M10, when multi-agent work actually
starts. What matters now is the property the rule protects, and that property is
tested here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lvgrid_rl.env.lv_grid_env import LVGridEnv

__all__ = ["AgentPartition", "PartitionedEnv"]


@dataclass(frozen=True, slots=True)
class AgentPartition:
    """One agent's slice of the flat action vector.

    Args:
        name: Agent identifier.
        component_indices: Positions within the flat action vector this agent
            controls. Derived from the asset order, which is stable by
            construction.
    """

    name: str
    component_indices: tuple[int, ...]


def single_partition(env: LVGridEnv, name: str = "central") -> tuple[AgentPartition, ...]:
    """The degenerate partition containing every actuator.

    This is what makes the single agent a special case rather than a special
    path: the centralised controller is a partitioned environment with one
    partition.
    """
    return (AgentPartition(name=name, component_indices=tuple(range(env.mapper.dim))),)


def partition_by_asset(env: LVGridEnv) -> tuple[AgentPartition, ...]:
    """One agent per actuator."""
    partitions: list[AgentPartition] = []
    cursor = 0
    for asset_id, spec in zip(env.mapper.asset_ids, env.mapper.specs, strict=True):
        partitions.append(
            AgentPartition(
                name=asset_id,
                component_indices=tuple(range(cursor, cursor + spec.dim)),
            )
        )
        cursor += spec.dim
    return tuple(partitions)


class PartitionedEnv:
    """Dictionary-keyed view over a :class:`LVGridEnv`.

    Args:
        env: The underlying environment. It is not copied: the partitioned view
            is a *view*, and the physics remains in one place.
        partitions: How the flat action vector is divided among agents.

    The observation is shared for now -- every agent sees the same vector. That
    is deliberate: restricting observations per agent is a research question
    (§6.2, ``sensor_config``) and not a property of the partitioning.
    """

    def __init__(
        self, env: LVGridEnv, partitions: Sequence[AgentPartition] | None = None
    ) -> None:
        self.env = env
        self.partitions = tuple(partitions or single_partition(env))
        seen = [i for p in self.partitions for i in p.component_indices]
        if sorted(seen) != list(range(env.mapper.dim)):
            raise ValueError(
                "Partitions must cover every action component exactly once; "
                f"got {len(seen)} assignments for {env.mapper.dim} components."
            )

    @property
    def agents(self) -> tuple[str, ...]:
        """Agent names."""
        return tuple(p.name for p in self.partitions)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset and return per-agent observations."""
        observation, info = self.env.reset(seed=seed, options=options)
        return (
            {name: observation for name in self.agents},
            {name: info for name in self.agents},
        )

    def step(
        self, actions: Mapping[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Reassemble the flat action, step once, and fan the result out.

        The reward is shared rather than split. Credit assignment across agents
        is a research question of its own, and inventing a split here would bake
        an answer into the plumbing.
        """
        flat = np.zeros(self.env.mapper.dim, dtype=np.float64)
        for partition in self.partitions:
            values = np.asarray(actions[partition.name], dtype=np.float64).reshape(-1)
            if values.shape != (len(partition.component_indices),):
                raise ValueError(
                    f"Agent {partition.name!r} supplied {values.shape[0]} values "
                    f"for {len(partition.component_indices)} components"
                )
            flat[list(partition.component_indices)] = values

        observation, reward, terminated, truncated, info = self.env.step(flat)
        return (
            {name: observation for name in self.agents},
            dict.fromkeys(self.agents, float(reward)),
            dict.fromkeys(self.agents, terminated),
            dict.fromkeys(self.agents, truncated),
            {name: info for name in self.agents},
        )

    def split_action(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat action into the per-agent dictionary."""
        array = np.asarray(flat, dtype=np.float64).reshape(-1)
        return {p.name: array[list(p.component_indices)] for p in self.partitions}
