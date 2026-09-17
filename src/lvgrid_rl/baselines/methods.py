"""Reference methods B0 to B4.

These matter as much as the agents, because they define the scale on which RL
results are read. "RL beats doing nothing" is not a defensible statement; "RL
reaches X % of the oracle at Y % of the compute time" is.

All of them implement :class:`~lvgrid_rl.core.protocols.Controller`, so they run
through exactly the same evaluation path as a trained policy. That is what makes
the comparison structurally fair rather than a matter of care.

**On parameterisation.** The deadbands and slopes of the droop methods are
implicitly designed for a hard ±10 % criterion. Under the EN 50160 percentile
criterion the optimum is different, because tolerance is permitted: a controller
may spend up to 50 ten-minute windows per bus and week outside the band. Every
baseline therefore gets its own documented parameter search against the same
target KPI (``scripts/tune_baselines.py``). Skipping that means the RL agent wins
against a deliberately badly tuned reference, which is the most attackable point
in this literature (architecture §6.6, consequence 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.env.actions import ActionMapper

__all__ = [
    "DoNothing",
    "RandomController",
    "FixedCap",
    "PUDroop",
    "QUDroop",
    "BASELINES",
]


@dataclass
class DoNothing:
    """B0 -- no curtailment at all.

    For PV in the consumer sign convention this is the *lower* action bound:
    full infeed. The sign trap is real, so the neutral action comes from the
    mapper rather than from a literal.
    """

    mapper: ActionMapper
    name: str = "do_nothing"

    def reset(self, info: InformationSet) -> None:
        """Nothing to reset."""

    def act(self, info: InformationSet) -> np.ndarray:
        """Always full infeed."""
        return self.mapper.neutral_action()


@dataclass
class RandomController:
    """B1 -- uniformly random actions.

    A sanity check, not a competitor: if a trained policy does not clearly beat
    this, something is wrong with the training rather than with the problem.
    """

    mapper: ActionMapper
    seed: int = 0
    name: str = "random"
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, info: InformationSet) -> None:
        """Restart the stream, so an evaluation is reproducible."""
        self._rng = np.random.default_rng(self.seed)

    def act(self, info: InformationSet) -> np.ndarray:
        """Draw a uniform action."""
        return self._rng.uniform(-1.0, 1.0, size=self.mapper.dim)


@dataclass
class FixedCap:
    """B2 -- rigid infeed limit, the historical 70 % / 60 % rule.

    Curtails every system to a fixed share of its rated power regardless of the
    grid state. Cheap, requires no measurement, and wastes yield on every day
    the grid would have coped.

    Args:
        cap: Admissible share of rated power, e.g. 0.7.
    """

    mapper: ActionMapper
    cap: float = 0.7
    name: str = "fixed_cap"

    def reset(self, info: InformationSet) -> None:
        """Nothing to reset."""

    def act(self, info: InformationSet) -> np.ndarray:
        """Cap every component at ``cap`` times its rated power."""
        physical = self.mapper.lower * self.cap
        return self.mapper.to_normalised(physical)


@dataclass
class PUDroop:
    """B4 -- voltage-dependent active power curtailment, P(U).

    Each system reduces its infeed once the voltage at its own bus exceeds a
    deadband, linearly up to full curtailment at ``v_max``. Local, needs only a
    local measurement, and is the closest rule-based competitor to what an RL
    agent can do with P alone.

    Args:
        v_start: Voltage at which curtailment begins.
        v_max: Voltage at which curtailment is complete.
        asset_bus_positions: Position of each asset's bus within the assessed
            connection points, in mapper order.
    """

    mapper: ActionMapper
    asset_bus_positions: tuple[int, ...]
    v_start: float = 1.06
    v_max: float = 1.10
    name: str = "p_u_droop"

    def __post_init__(self) -> None:
        if not self.v_start < self.v_max:
            raise ValueError("v_start must be below v_max")
        if len(self.asset_bus_positions) != len(self.mapper.asset_ids):
            raise ValueError("one bus position per asset is required")

    def reset(self, info: InformationSet) -> None:
        """Nothing to reset; the rule is memoryless."""

    def act(self, info: InformationSet) -> np.ndarray:
        """Curtail proportionally to the local voltage excess."""
        vm = np.asarray(info.measurements["vm_pu"])
        local = vm[list(self.asset_bus_positions)]
        share = np.clip((local - self.v_start) / (self.v_max - self.v_start), 0.0, 1.0)
        # share = 0 means full infeed (lower bound), share = 1 means zero infeed.
        physical = self.mapper.lower * (1.0 - share)
        return self.mapper.to_normalised(physical)


@dataclass
class QUDroop:
    """B3 -- reactive power characteristic Q(U) per VDE-AR-N 4105.

    Requires the environment to be built with ``PvMode.PQ``, because the action
    space must contain a reactive component. In a low-voltage grid the effect is
    limited by the high R/X ratio, which is exactly why this is a reference case
    rather than the main study (decision D4).

    Args:
        v_start: Voltage at which reactive power absorption begins.
        v_max: Voltage at which the full reactive capability is used.
    """

    mapper: ActionMapper
    asset_bus_positions: tuple[int, ...]
    v_start: float = 1.04
    v_max: float = 1.10
    name: str = "q_u_droop"

    def __post_init__(self) -> None:
        if not self.v_start < self.v_max:
            raise ValueError("v_start must be below v_max")
        names = self.mapper.component_names
        if not any(name.endswith("/q_mvar") for name in names):
            raise ValueError(
                "Q(U) needs a reactive component in the action space; build the "
                "environment with pv_mode='pq'."
            )

    def reset(self, info: InformationSet) -> None:
        """Nothing to reset."""

    def act(self, info: InformationSet) -> np.ndarray:
        """Absorb reactive power proportionally to the local voltage excess."""
        vm = np.asarray(info.measurements["vm_pu"])
        physical = self.mapper.lower.copy()
        cursor = 0
        for position, spec in zip(
            self.asset_bus_positions, self.mapper.specs, strict=True
        ):
            share = float(
                np.clip(
                    (vm[position] - self.v_start) / (self.v_max - self.v_start),
                    0.0,
                    1.0,
                )
            )
            for offset, name in enumerate(spec.names):
                bound = spec.bounds[offset]
                if name == "p_mw":
                    physical[cursor + offset] = bound.lo  # full infeed
                else:
                    # Consumer convention: absorbing reactive power is positive.
                    physical[cursor + offset] = share * bound.hi
            cursor += spec.dim
        return self.mapper.to_normalised(physical)


BASELINES = {
    "do_nothing": DoNothing,
    "random": RandomController,
    "fixed_cap": FixedCap,
    "p_u_droop": PUDroop,
    "q_u_droop": QUDroop,
}
"""Registry by name, for configuration-driven evaluation."""


def asset_bus_positions(env) -> tuple[int, ...]:
    """Position of each asset's bus within the assessed connection points.

    The local droop rules need the voltage at their own bus, and the observation
    exposes voltages indexed by connection point rather than by pandapower bus.
    Resolving that once here keeps the mapping out of the control loop and out of
    every individual rule.
    """
    order = list(env.model.connection_point_buses)
    return tuple(order.index(asset.bus) for asset in env.assets)
