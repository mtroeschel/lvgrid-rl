"""Protocols of the interchangeable components.

This module contains interfaces only, no implementations. Its purpose is to fix
the contracts that the extensibility contract (section 13 of the architecture
document) protects -- in particular those whose retrofitting would be expensive
to impossible:

* :class:`FlexAsset` with pure dynamics (I2) and physical action limits (I4),
* :class:`PowerFlowEngine` with a hypothetical call (I5),
* :class:`SafetyComponent` and :class:`SafetyCertifier` as intervention points
  that exist from M0 with a null implementation (I7).

Concrete implementations follow from M2 onwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.schemas import (
    AssetRatings,
    AssetState,
    ExogenousInput,
    GridState,
    Interval,
    SystemState,
)

__all__ = [
    "ActionSpec",
    "Setpoint",
    "AssetOutcome",
    "FlexAsset",
    "PowerFlowEngine",
    "Verdict",
    "UncertaintySet",
    "SafetyCertifier",
    "InterventionInfo",
    "SafetyComponent",
    "NullSafetyComponent",
    "Controller",
]


# ---------------------------------------------------------------------------
# Actions and setpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action space of an asset, in **physical** units.

    **Invariant I4.** Actions are power quantities, not normalised fractions and
    not target values. Semantics such as "state-of-charge target" or "fraction
    of remaining energy" produce a non-box feasible set in action coordinates,
    whereas the certified feasible set is formulated in injections. Normalisation
    for the agent happens in the ``ActionMapper`` and is affine and invertible
    there.

    Args:
        names: Descriptive names of the components, e.g. ``("p_mw", "q_mvar")``.
        bounds: Limits per component.
        discrete_levels: For discretised variants, the number of levels per
            component; otherwise ``None``. Needed for the masking arm of the
            safe-RL comparison (section 6.8).
        neutral: The physical value per component that means "no
            intervention": full infeed for PV active power, zero for reactive
            power, zero for a battery. ``None`` falls back to the lower bound,
            which is right for PV active power and wrong for everything else --
            the reason every asset model states it explicitly.
    """

    names: tuple[str, ...]
    bounds: tuple[Interval, ...]
    discrete_levels: tuple[int, ...] | None = None
    neutral: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.names) != len(self.bounds):
            raise ValueError("names and bounds must have equal length")
        if self.neutral is not None:
            if len(self.neutral) != len(self.names):
                raise ValueError("neutral must match names")
            for value, bound in zip(self.neutral, self.bounds, strict=True):
                if not bound.contains(value):
                    raise ValueError(f"neutral value {value} outside {bound}")
        if self.discrete_levels is not None and len(self.discrete_levels) != len(
            self.names
        ):
            raise ValueError("discrete_levels must match names")

    @property
    def dim(self) -> int:
        """Dimension of this asset's action space."""
        return len(self.names)


@dataclass(frozen=True, slots=True)
class Setpoint:
    """Executed operating point of an asset, consumer reference direction.

    ``clipping_info`` is not meant to be optional: **every** limitation an asset
    applies to the proposed action is reported here and not performed silently
    (invariant I4). Only then can one later distinguish whether a policy
    systematically proposes infeasible actions or whether a safety mechanism
    intervened.
    """

    asset_id: str
    p_mw: float
    q_mvar: float = 0.0
    clipping_info: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.clipping_info is None:
            object.__setattr__(self, "clipping_info", {})

    @property
    def was_clipped(self) -> bool:
        """Was the proposed action limited?"""
        return bool(self.clipping_info)


@dataclass(frozen=True, slots=True)
class AssetOutcome:
    """Consequences of one simulation step that matter for reward and KPIs.

    The quantities are physical and unweighted. Weighting happens in the
    ``RewardComposer``; the comparison against reference methods is made on
    these raw quantities and never on the reward (section 6.4).

    Energies are energies **over the step** passed to
    :meth:`FlexAsset.dynamics`, not powers. Until M4 the PV model returned a
    power in ``curtailed_energy_mwh`` and left the multiplication to the caller,
    because the protocol did not pass a step duration; the suffix promised MWh
    and the value was MW.
    """

    curtailed_energy_mwh: float = 0.0
    unserved_energy_mwh: float = 0.0
    comfort_deviation_kh: float = 0.0
    throughput_energy_mwh: float = 0.0
    """Energy through the asset's grid terminal, charging and discharging alike.
    The basis of the degradation cost term."""
    loss_energy_mwh: float = 0.0
    """Conversion and standing losses inside the asset."""
    switching_count: int = 0


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


@runtime_checkable
class FlexAsset(Protocol):
    """Controllable asset connected to the grid.

    **Invariant I2:** :meth:`to_setpoint`, :meth:`limit_to_physics` and
    :meth:`dynamics` are pure functions. The same input yields the same output,
    no input is mutated, and the asset object itself holds parameters only,
    never state. A predictive safety filter has to roll asset states forward
    hypothetically over a horizon; with state-mutating methods that could only be
    retrofitted by rewriting every asset model.

    **Two time scales.** A decision is made once per control step and held for
    ``hold_min``; the physics runs in simulation steps of ``dt_min``.
    :meth:`to_setpoint` projects the action onto what is feasible for the whole
    hold and reports every limitation -- that is where a policy proposing the
    infeasible becomes visible. :meth:`limit_to_physics` then applies what can
    only be known inside the hold (irradiance, a state reached earlier than
    foreseen) once per simulation step, before the power flow.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings
    kind: str
    """Asset type, e.g. ``"pv"`` or ``"bess"``. Action mode 2 shares parameters
    per kind, and rule-based controllers act on the kinds they are meant for."""

    def action_spec(self) -> ActionSpec:
        """Action space in physical units."""
        ...

    def initial_state(self, rng: np.random.Generator) -> AssetState:
        """Initial state, drawn with the given generator."""
        ...

    def to_setpoint(
        self, s: AssetState, action: np.ndarray, info: InformationSet, hold_min: float
    ) -> Setpoint:
        """Map an action onto an operating point held for ``hold_min``.

        Limiting to the feasible range is allowed but must be reported in
        ``Setpoint.clipping_info``.
        """
        ...

    def limit_to_physics(
        self, s: AssetState, sp: Setpoint, x: ExogenousInput, dt_min: float
    ) -> Setpoint:
        """Limit a held setpoint to what is physically possible in this step.

        Applied once per simulation step, before the power flow. Returns ``sp``
        itself when nothing is limited.
        """
        ...

    def dynamics(
        self,
        s: AssetState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
        dt_min: float,
    ) -> tuple[AssetState, AssetOutcome]:
        """Advance the state by ``dt_min`` as a pure function."""
        ...


# ---------------------------------------------------------------------------
# Grid physics
# ---------------------------------------------------------------------------


@runtime_checkable
class PowerFlowEngine(Protocol):
    """Power flow calculation with a hypothetical call.

    **Invariant I5:** :meth:`run_hypothetical` must not alter the state of the
    live grid. Backup trajectories of a predictive safety filter and the
    falsification search of the verification step need exactly that.
    """

    def run(self, setpoints: Mapping[str, Setpoint], t_index: int) -> GridState:
        """Run the power flow on the live grid and advance its state."""
        ...

    def run_hypothetical(
        self, setpoints: Mapping[str, Setpoint], t_index: int
    ) -> GridState:
        """Run the power flow without altering the live grid."""
        ...


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Result of a feasibility check -- deliberately asymmetric.

    There is no ``UNSAFE``. ``NOT_CERTIFIED`` means "not provably feasible", and
    that is sufficient to trigger the fallback. A certifier may be conservative
    but never optimistic -- this asymmetry is the basis of any later guarantee
    (section 6.9).
    """

    CERTIFIED_SAFE = "certified_safe"
    NOT_CERTIFIED = "not_certified"


@dataclass(frozen=True, slots=True)
class UncertaintySet:
    """Set of possible realisations of the uncontrollable injections.

    Args:
        series_ids: Names of the affected time series.
        bounds_mw: Shape ``(2, n)``, lower and upper bounds.
        confidence: ``None`` for deterministic bounds derived from asset
            ratings -- the statement then holds without residual probability.
            For data-driven sets, the confidence ``1 - delta``, which must then
            be stated alongside every guarantee (section 6.9, building block 2).
    """

    series_ids: tuple[str, ...]
    bounds_mw: np.ndarray
    confidence: float | None = None


@runtime_checkable
class SafetyCertifier(Protocol):
    """Checks whether an action is provably feasible.

    Implementations from M7b onwards: robust convex restriction, alternatively
    linearisation with a rigorous remainder bound. Plain linearisation without a
    remainder bound explicitly does **not** satisfy this protocol -- it belongs
    to the heuristic mechanisms (section 6.8).
    """

    def certify(
        self, state: SystemState, action: np.ndarray, uncertainty: UncertaintySet
    ) -> Verdict:
        """Is the action feasible for all realisations of the uncertainty set?"""
        ...


@dataclass(frozen=True, slots=True)
class InterventionInfo:
    """Record of a safety intervention, basis of the safety KPIs (section 8.3)."""

    intervened: bool
    magnitude: float = 0.0
    """Norm of the action change, ``||a' - a||``."""
    reason: str = ""


@runtime_checkable
class SafetyComponent(Protocol):
    """Safety intervention *inside* the environment.

    **Invariant I7.** Deliberately not a Gym wrapper: a wrapper only sees the
    observation, whereas a certifier needs the full ``SystemState``. The second
    intervention point, for masking, lies outside the environment and is
    provided via ``info["action_mask"]``.
    """

    def transform(
        self, action: np.ndarray, state: SystemState, info: InformationSet
    ) -> tuple[np.ndarray, InterventionInfo]:
        """Alter the action if necessary."""
        ...

    def action_mask(self, state: SystemState, info: InformationSet) -> np.ndarray | None:
        """Mask of admissible discrete actions, or ``None``."""
        ...


class NullSafetyComponent:
    """Default implementation: never intervenes, permits everything.

    Exists from M0 so that the intervention point is part of the step sequence
    from the start and later only has to be swapped in, not built in.
    """

    def transform(
        self, action: np.ndarray, state: SystemState, info: InformationSet
    ) -> tuple[np.ndarray, InterventionInfo]:
        """Return the action unchanged."""
        return action, InterventionInfo(intervened=False)

    def action_mask(self, state: SystemState, info: InformationSet) -> np.ndarray | None:
        """Permit all actions."""
        return None


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@runtime_checkable
class Controller(Protocol):
    """Common interface of RL policies and reference methods.

    The evaluation path knows only this protocol. Rule-based methods, OPF, MPC
    and trained policies therefore run through exactly the same evaluation,
    which makes the comparison structurally fair (section 7.1).

    Methods that need more than the ``InformationSet`` -- such as ``mpc_oracle``
    with perfect foresight -- receive that privileged access explicitly and are
    flagged accordingly in the results report.
    """

    def reset(self, info: InformationSet) -> None:
        """Reset internal state at the start of an episode."""
        ...

    def act(self, info: InformationSet) -> np.ndarray:
        """Action for the current decision point."""
        ...
