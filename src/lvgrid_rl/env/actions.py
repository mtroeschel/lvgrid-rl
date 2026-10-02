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

__all__ = ["ActionMapper", "PV_NORMALISATIONS"]

PV_NORMALISATIONS = ("rated", "available")
"""How a PV system's active power action is normalised.

``rated``: on its rated limits, the mapping since M3. ``available``: on the
power the forecast says is available at the decision, ``-1`` full infeed and
``+1`` none -- still a power, affine and invertible, but its scale depends on
the information set. The candidate remedy for the PV dead zone (architecture
§6.3), compared against ``rated`` before the M4 acceptance runs."""


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
    kinds: tuple[str, ...] | None = None
    """Asset type per asset, same order; ``None`` for hand-built mappers."""
    assets: tuple | None = None
    """The assets themselves, for those whose idle behaviour is a controller of
    their own (``default_action``); ``None`` for hand-built mappers."""
    pv_normalisation: str = "rated"
    """See :data:`PV_NORMALISATIONS`."""

    def __post_init__(self) -> None:
        if self.pv_normalisation not in PV_NORMALISATIONS:
            raise ValueError(
                f"pv_normalisation must be one of {PV_NORMALISATIONS}, "
                f"not {self.pv_normalisation!r}"
            )
        if self.pv_normalisation == "available" and self.assets is None:
            raise ValueError("pv_normalisation 'available' needs the assets")
        if len(self.asset_ids) != len(self.specs):
            raise ValueError("asset_ids and specs must have equal length")
        if self.kinds is not None and len(self.kinds) != len(self.asset_ids):
            raise ValueError("kinds must match asset_ids")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique")

    @classmethod
    def from_assets(
        cls, assets: Sequence, pv_normalisation: str = "rated"
    ) -> ActionMapper:
        """Build the mapper from a sequence of assets, preserving their order."""
        return cls(
            asset_ids=tuple(a.asset_id for a in assets),
            specs=tuple(a.action_spec() for a in assets),
            kinds=tuple(a.kind for a in assets),
            assets=tuple(assets),
            pv_normalisation=pv_normalisation,
        )

    def default_physical(self, info=None) -> np.ndarray:
        """What every asset does when nobody intervenes, in physical units.

        The static :attr:`neutral` for assets whose idle behaviour is a fixed
        value; the asset's own controller (``default_action``) for the others --
        a heat pump left alone follows its thermostat, it does not switch off.

        Raises:
            ValueError: if an asset has a controller of its own and no
                information set is given, since its default depends on state.
        """
        out = self.neutral.copy()
        if self.assets is None:
            return out
        cursor = 0
        for asset, spec in zip(self.assets, self.specs, strict=True):
            if hasattr(asset, "default_action"):
                if info is None:
                    raise ValueError(
                        f"{asset.asset_id} has a controller of its own; its default "
                        "depends on state, so pass the information set"
                    )
                state = info.asset_states[asset.asset_id]
                out[cursor : cursor + spec.dim] = asset.default_action(state, info)
            cursor += spec.dim
        return out

    def component_mask(self, kind: str, name: str) -> np.ndarray:
        """Boolean mask over the flat vector: components ``name`` of ``kind``.

        What a rule-based controller acts on. A cap or droop rule for PV must
        leave a battery alone, and it can only do that if it can tell the two
        apart.
        """
        if self.kinds is None:
            raise ValueError("this mapper carries no asset kinds")
        return np.array(
            [
                k == kind and n == name
                for k, spec in zip(self.kinds, self.specs, strict=True)
                for n in spec.names
            ],
            dtype=bool,
        )

    @property
    def neutral(self) -> np.ndarray:
        """Physical "no intervention" value per component, concatenated."""
        return np.array(
            [
                value
                for spec in self.specs
                for value in (
                    spec.neutral
                    if spec.neutral is not None
                    else tuple(b.lo for b in spec.bounds)
                )
            ],
            dtype=np.float64,
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

    def _scale(self, info) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper ends of the normalisation, per component.

        The rated limits, except under ``available`` normalisation, where a PV
        system's active power runs from its forecast available power (full
        infeed) to zero.
        """
        lower, upper = self.lower, self.upper
        if self.pv_normalisation == "rated":
            return lower, upper
        if info is None:
            raise ValueError(
                "pv_normalisation 'available' needs the information set: the "
                "scale of a PV action is the power available at the decision"
            )
        lower = lower.copy()
        cursor = 0
        for asset, spec in zip(self.assets, self.specs, strict=True):
            if asset.kind == "pv":
                for offset, name in enumerate(spec.names):
                    if name == "p_mw":
                        available = float(info.forecast[asset.series_id][0])
                        bound = spec.bounds[offset]
                        lower[cursor + offset] = min(max(available, bound.lo), bound.hi)
            cursor += spec.dim
        return lower, upper

    def to_physical(self, normalised: np.ndarray, info=None) -> np.ndarray:
        """Map a normalised action in ``[-1, 1]`` onto physical units.

        Degenerate components, where lower and upper bound coincide, map to that
        constant. That happens for a PV system whose rated power is zero and must
        not produce a division by zero.
        """
        a = np.asarray(normalised, dtype=np.float64).reshape(-1)
        if a.shape != (self.dim,):
            raise ValueError(f"Action has shape {a.shape}, expected ({self.dim},)")
        lower, upper = self._scale(info)
        return lower + 0.5 * (np.clip(a, -1.0, 1.0) + 1.0) * (upper - lower)

    def to_normalised(self, physical: np.ndarray, info=None) -> np.ndarray:
        """Inverse of :meth:`to_physical`.

        Degenerate components map to zero, which is the canonical representative
        of a one-point interval -- a PV system at night under ``available``
        normalisation among them. Under ``available`` a setpoint asking for more
        infeed than is available lies outside the scale and maps to ``-1``,
        full infeed, which is what the PV system would deliver anyway.
        """
        p = np.asarray(physical, dtype=np.float64).reshape(-1)
        if p.shape != (self.dim,):
            raise ValueError(f"Setpoint has shape {p.shape}, expected ({self.dim},)")
        lower, upper = self._scale(info)
        span = upper - lower
        out = np.zeros_like(p)
        nondegenerate = span > 0.0
        out[nondegenerate] = (
            2.0 * (p[nondegenerate] - lower[nondegenerate]) / span[nondegenerate] - 1.0
        )
        if self.pv_normalisation == "available":
            out = np.clip(out, -1.0, 1.0)
        return out

    def split(self, physical: np.ndarray) -> dict[str, np.ndarray]:
        """Split a flat physical vector into per-asset slices."""
        out: dict[str, np.ndarray] = {}
        cursor = 0
        for asset_id, spec in zip(self.asset_ids, self.specs, strict=True):
            out[asset_id] = physical[cursor : cursor + spec.dim]
            cursor += spec.dim
        return out

    def neutral_action(self, info=None) -> np.ndarray:
        """Normalised action that intervenes nowhere.

        For a PV system the upper bound is zero infeed, so ``+1`` would mean full
        curtailment and ``-1`` full infeed; for a battery ``-1`` is full
        discharge and the neutral action is the middle. The values come from
        each asset's :attr:`ActionSpec.neutral`, not from a bound: until M4 this
        returned the lower bound for every component, which was right only for
        PV active power. Assets with a controller of their own contribute its
        action, which needs ``info`` (:meth:`default_physical`).
        """
        return self.to_normalised(self.default_physical(info), info)
