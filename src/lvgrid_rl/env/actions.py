"""Mapping between the agent's action space and physical setpoints.

**Invariant I4.** Asset actions are physical power quantities; the normalisation
to the ``[-1, 1]`` box that RL algorithms expect happens here and is **affine and
invertible**. Two consequences follow, and both are the reason this is a separate
module rather than three lines inside the environment:

* A later certifier formulates the feasible set in injections. Projecting onto it
  requires action coordinates that are box-shaped and related to injections by an
  invertible map. Semantics such as "state-of-charge target" or "fraction of
  remaining energy" break that.
* The inverse is needed in practice, not only in theory: a safety mechanism that
  modifies an executed setpoint has to report the corresponding action back to
  the learning algorithm (`store_executed` coupling, §7.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lvgrid_rl.core.protocols import ActionSpec

__all__ = ["ActionMapper"]


@dataclass(frozen=True, slots=True)
class ActionMapper:
    """Flat concatenation of per-asset action spaces, mode 1 of §6.3.

    Args:
        asset_ids: Assets in a stable order. The order is fixed at construction
            and never derived from a dictionary iteration, because an action
            vector whose meaning depends on insertion order is untraceable.
        specs: Action specification per asset, same order.

    Example:
        >>> from lvgrid_rl.core.protocols import ActionSpec
        >>> from lvgrid_rl.core.schemas import Interval
        >>> spec = ActionSpec(names=("p_mw",), bounds=(Interval(-0.02, 0.0),))
        >>> mapper = ActionMapper(("sgen:0",), (spec,))
        >>> mapper.to_physical(np.array([-1.0]))
        array([-0.02])
        >>> mapper.to_physical(np.array([1.0]))
        array([0.])
    """

    asset_ids: tuple[str, ...]
    specs: tuple[ActionSpec, ...]

    def __post_init__(self) -> None:
        if len(self.asset_ids) != len(self.specs):
            raise ValueError("asset_ids and specs must have equal length")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique")

    @classmethod
    def from_assets(cls, assets: Sequence) -> ActionMapper:
        """Build the mapper from a sequence of assets, preserving their order."""
        return cls(
            asset_ids=tuple(a.asset_id for a in assets),
            specs=tuple(a.action_spec() for a in assets),
        )

    @property
    def dim(self) -> int:
        """Dimension of the flat action vector."""
        return sum(spec.dim for spec in self.specs)

    @property
    def lower(self) -> np.ndarray:
        """Lower physical bounds, concatenated."""
        return np.array(
            [b.lo for spec in self.specs for b in spec.bounds], dtype=np.float64
        )

    @property
    def upper(self) -> np.ndarray:
        """Upper physical bounds, concatenated."""
        return np.array(
            [b.hi for spec in self.specs for b in spec.bounds], dtype=np.float64
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Human-readable name per action component, for logging and debugging."""
        return tuple(
            f"{asset_id}/{name}"
            for asset_id, spec in zip(self.asset_ids, self.specs, strict=True)
            for name in spec.names
        )

    def to_physical(self, normalised: np.ndarray) -> np.ndarray:
        """Map a normalised action in ``[-1, 1]`` onto physical units.

        Degenerate components, where lower and upper bound coincide, map to that
        constant. That happens for a PV system whose rated power is zero and must
        not produce a division by zero.
        """
        a = np.asarray(normalised, dtype=np.float64).reshape(-1)
        if a.shape != (self.dim,):
            raise ValueError(f"Action has shape {a.shape}, expected ({self.dim},)")
        span = self.upper - self.lower
        return self.lower + 0.5 * (np.clip(a, -1.0, 1.0) + 1.0) * span

    def to_normalised(self, physical: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_physical`.

        Degenerate components map to zero, which is the canonical representative
        of a one-point interval.
        """
        p = np.asarray(physical, dtype=np.float64).reshape(-1)
        if p.shape != (self.dim,):
            raise ValueError(f"Setpoint has shape {p.shape}, expected ({self.dim},)")
        span = self.upper - self.lower
        out = np.zeros_like(p)
        nondegenerate = span > 0.0
        out[nondegenerate] = (
            2.0 * (p[nondegenerate] - self.lower[nondegenerate]) / span[nondegenerate]
            - 1.0
        )
        return out

    def split(self, physical: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat physical vector into per-asset slices."""
        out: dict[str, np.ndarray] = {}
        cursor = 0
        for asset_id, spec in zip(self.asset_ids, self.specs, strict=True):
            out[asset_id] = physical[cursor : cursor + spec.dim]
            cursor += spec.dim
        return out

    def neutral_action(self) -> np.ndarray:
        """Normalised action that curtails nothing.

        For a PV system the upper bound is zero infeed, so ``+1`` would mean full
        curtailment and ``-1`` full infeed. This helper avoids that sign trap in
        baselines and tests.
        """
        return self.to_normalised(self.lower)
