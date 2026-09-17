"""Reward composition with separated objective and constraint terms.

The split into ``objective`` (costs to minimise) and ``constraint`` (limits to
respect) is what lets fixed weights and the Lagrangian variant run from the same
configuration (§6.4). In ``fixed_weights`` mode everything is scalarised; in
``lagrangian`` mode only the objective terms form the reward and the constraint
terms are reported as costs, with multipliers updated outside.

Both constraint terms are active from M3. The M2 survey showed that thermal
overload is the dominant binding constraint in the reference scenarios and
remains significant under ``moderate_growth`` (12.6 % of steps), so an
environment with only a voltage term would optimise against a criterion that
barely binds.

Every term is reported individually. Without that decomposition it is impossible
to reconstruct later which term dominated learning -- and the comparison against
reference methods runs on physical KPIs, never on the reward.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from lvgrid_rl.grid.metrics import ViolationMetrics
from lvgrid_rl.grid.pq import K95_BUDGET_FRACTION

__all__ = ["RewardMode", "TermSpec", "RewardConfig", "RewardComposer", "RewardBreakdown"]


class RewardMode(StrEnum):
    """How objective and constraint terms are combined."""

    FIXED_WEIGHTS = "fixed_weights"
    LAGRANGIAN = "lagrangian"


@dataclass(frozen=True, slots=True)
class TermSpec:
    """One reward term.

    Args:
        weight: Multiplier in ``fixed_weights`` mode. Negative for costs.
        limit: Admissible value in ``lagrangian`` mode, in the term's own unit.
            For ``en50160_k95`` this is 0.05 -- exactly the standard's criterion,
            so no penalty weight needs guessing.
    """

    weight: float
    limit: float = 0.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward configuration, mirroring ``configs/env/*.yaml``."""

    mode: RewardMode = RewardMode.FIXED_WEIGHTS
    objective: Mapping[str, TermSpec] = field(
        default_factory=lambda: {
            "pv_curtailment": TermSpec(weight=-1.0),
            "action_smoothness": TermSpec(weight=-0.05),
            "grid_losses": TermSpec(weight=-0.1),
        }
    )
    constraint: Mapping[str, TermSpec] = field(
        default_factory=lambda: {
            "en50160_k95": TermSpec(weight=-10.0, limit=K95_BUDGET_FRACTION),
            "en50160_k100": TermSpec(weight=-50.0, limit=0.0),
            "thermal_overload": TermSpec(weight=-10.0, limit=0.0),
        }
    )
    potential_scale: float = 1.0
    """Weight of the potential-based shaping on the K95 budget."""
    divergence_penalty: float = -100.0
    """Applied when the power flow does not converge."""

    def __post_init__(self) -> None:
        overlap = set(self.objective) & set(self.constraint)
        if overlap:
            raise ValueError(f"Term appears in both categories: {sorted(overlap)}")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Result of one reward evaluation.

    ``costs`` carries the constraint terms in their own units, which is what the
    Lagrangian callback consumes and what the KPI engine records unweighted.
    """

    total: float
    terms: Mapping[str, float]
    costs: Mapping[str, float]

    def as_info(self) -> dict[str, float]:
        """Flat dictionary for the environment's ``info``."""
        out = {f"reward/{k}": v for k, v in self.terms.items()}
        out.update({f"cost/{k}": v for k, v in self.costs.items()})
        out["reward/total"] = self.total
        return out


class RewardComposer:
    """Turns physical outcomes into a scalar reward plus a breakdown.

    Args:
        config: Term weights and mode.
        budget_windows: Permitted violating windows per bus and week, used to
            normalise the K95 cost onto ``[0, ~1]``.
    """

    def __init__(self, config: RewardConfig, budget_windows: int) -> None:
        self.config = config
        self.budget_windows = max(budget_windows, 1)
        self._multipliers: dict[str, float] = dict.fromkeys(config.constraint, 0.0)

    @property
    def multipliers(self) -> Mapping[str, float]:
        """Current Lagrange multipliers, one per constraint term."""
        return dict(self._multipliers)

    def set_multipliers(self, values: Mapping[str, float]) -> None:
        """Update the multipliers from outside.

        The environment does not update them itself: a changing multiplier makes
        the reward non-stationary, and who is allowed to change it and when is a
        property of the training loop, not of the physics.
        """
        unknown = set(values) - set(self._multipliers)
        if unknown:
            raise KeyError(f"Unknown constraint terms: {sorted(unknown)}")
        self._multipliers.update(values)

    def potential(self, budget_used_max: float, week_progress: float) -> float:
        """Potential for shaping on the K95 budget.

        The true cost is terminal -- whether a week passes is only known at its
        end. Potential-based shaping supplies a dense signal without changing the
        optimal policy (Ng et al.): the shaped reward adds ``gamma * phi(s') -
        phi(s)``, which telescopes over any trajectory.

        The potential is negative and grows in magnitude as the budget is
        consumed faster than the week elapses. Spending budget early is therefore
        penalised more than spending it late, which is the correct incentive:
        early consumption removes options.
        """
        headroom = budget_used_max - week_progress
        return -self.config.potential_scale * max(headroom, 0.0) ** 2

    def compute(
        self,
        metrics: ViolationMetrics,
        curtailed_energy_mwh: float,
        setpoint_change_mw: float,
        k95_violations_this_step: int,
        k100_violations_this_step: int,
        n_buses: int,
        dt_hours: float,
    ) -> RewardBreakdown:
        """Evaluate the reward for one control step.

        Args:
            metrics: Instantaneous violation metrics of the step.
            curtailed_energy_mwh: Energy not fed in because of curtailment.
            setpoint_change_mw: Summed magnitude of the setpoint change against
                the previous control step, for the smoothness term. A magnitude
                rather than a switch count: a controller that moves a setpoint
                by a kilowatt is not doing the same thing as one that swings it
                by the full rated power, and a binary counter cannot tell them
                apart.
            k95_violations_this_step: Assessment windows that closed outside the
                K95 band during this step, summed over buses.
            k100_violations_this_step: Same for the absolute band.
            n_buses: Number of assessed connection points, for normalisation.
            dt_hours: Duration of the control step.
        """
        if not metrics.converged:
            # A diverged power flow is a defined event, not an exception. It
            # carries a large penalty because the state is uninformative, but it
            # must not abort the run.
            return RewardBreakdown(
                total=self.config.divergence_penalty,
                terms={"divergence": self.config.divergence_penalty},
                costs=dict.fromkeys(self.config.constraint, float("nan")),
            )

        scale = max(n_buses, 1) * self.budget_windows
        costs = {
            "en50160_k95": k95_violations_this_step / scale,
            "en50160_k100": k100_violations_this_step / max(n_buses, 1),
            "thermal_overload": metrics.overload_excess_percent / 100.0 * dt_hours,
        }
        objective = {
            "pv_curtailment": curtailed_energy_mwh,
            "action_smoothness": float(setpoint_change_mw),
            "grid_losses": metrics.losses_mw * dt_hours,
        }

        terms: dict[str, float] = {}
        for name, value in objective.items():
            spec = self.config.objective.get(name)
            if spec is not None:
                terms[name] = spec.weight * value

        if self.config.mode is RewardMode.FIXED_WEIGHTS:
            for name, value in costs.items():
                spec = self.config.constraint.get(name)
                if spec is not None:
                    terms[name] = spec.weight * max(value - spec.limit, 0.0)
        else:
            for name, value in costs.items():
                if name in self.config.constraint:
                    terms[name] = -self._multipliers[name] * value

        return RewardBreakdown(
            total=float(np.sum(list(terms.values()))), terms=terms, costs=costs
        )
