"""Instantaneous grid metrics and the fallback feasibility check.

Two things live here that look unrelated but share the same question -- is this
operating point admissible:

* :func:`violation_metrics` reduces a :class:`GridState` to the quantities the
  reward and the KPIs need, per simulation step.
* :func:`check_fallback_feasible` answers precondition P4 of the certification
  plan (section 6.9): does a feasible fallback action exist at all? If the base
  load alone already violates limits, no controller can guarantee anything and
  the answer is grid reinforcement, not control. Running this early is cheap and
  says whether a hard guarantee is reachable in the planned scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lvgrid_rl.core.schemas import GridState
from lvgrid_rl.grid.pq import K95_BAND_PU, K100_BAND_PU

__all__ = ["ViolationMetrics", "violation_metrics", "FallbackReport"]


@dataclass(frozen=True, slots=True)
class ViolationMetrics:
    """Instantaneous violation quantities of one simulation step.

    These are the raw physical quantities, unweighted. Weighting happens in the
    reward composer; the comparison against reference methods runs on these.

    Note that the voltage quantities here are *instantaneous*. They are a
    secondary indicator, kept for comparability with the literature, which
    commonly uses a hard band. The criterion this project is assessed against
    works on ten-minute means and weekly intervals, see
    :mod:`lvgrid_rl.grid.pq`.
    """

    max_vm_pu: float
    min_vm_pu: float
    voltage_excursion_pu: float
    """Sum over assessed buses of the distance outside the K95 band."""
    n_buses_outside_k95: int
    n_buses_outside_k100: int
    max_line_loading_percent: float
    max_trafo_loading_percent: float
    overload_excess_percent: float
    """Sum of loading above 100 % across all lines and transformers."""
    losses_mw: float
    converged: bool


def violation_metrics(
    state: GridState, evaluated_positions: np.ndarray
) -> ViolationMetrics:
    """Reduce a grid state to its violation quantities.

    Args:
        state: Result of one power flow.
        evaluated_positions: Positional indices of the assessed connection
            points within ``state.vm_pu``.
    """
    if not state.converged:
        return ViolationMetrics(
            max_vm_pu=float("nan"),
            min_vm_pu=float("nan"),
            voltage_excursion_pu=float("nan"),
            n_buses_outside_k95=0,
            n_buses_outside_k100=0,
            max_line_loading_percent=float("nan"),
            max_trafo_loading_percent=float("nan"),
            overload_excess_percent=float("nan"),
            losses_mw=float("nan"),
            converged=False,
        )

    v = state.vm_pu[evaluated_positions]
    lo95, hi95 = K95_BAND_PU
    lo100, hi100 = K100_BAND_PU
    excursion = np.maximum(v - hi95, 0.0) + np.maximum(lo95 - v, 0.0)

    loadings = np.concatenate([state.line_loading_percent, state.trafo_loading_percent])
    excess = np.maximum(loadings - 100.0, 0.0)

    return ViolationMetrics(
        max_vm_pu=float(np.nanmax(v)),
        min_vm_pu=float(np.nanmin(v)),
        voltage_excursion_pu=float(np.nansum(excursion)),
        n_buses_outside_k95=int(np.sum((v < lo95) | (v > hi95))),
        n_buses_outside_k100=int(np.sum((v < lo100) | (v > hi100))),
        max_line_loading_percent=float(np.nanmax(state.line_loading_percent)),
        max_trafo_loading_percent=float(np.nanmax(state.trafo_loading_percent)),
        overload_excess_percent=float(np.nansum(excess)),
        losses_mw=float(state.losses_mw),
        converged=True,
    )


@dataclass(frozen=True, slots=True)
class FallbackReport:
    """Result of the P4 pre-check.

    Args:
        steps: Number of time steps examined.
        infeasible_steps: Steps in which the fallback action itself violates a
            limit.
        worst_vm_pu: Extreme voltages under the fallback action.
        worst_loading_percent: Worst equipment loading under the fallback.
        first_infeasible_step: Index of the first offending step, for
            inspection.
    """

    steps: int
    infeasible_steps: int
    worst_vm_pu: tuple[float, float]
    worst_loading_percent: float
    first_infeasible_step: int | None

    @property
    def feasible(self) -> bool:
        """Does a feasible fallback exist at every examined step?

        If this is ``False``, precondition P4 of the certification plan fails.
        That is not a setback but a finding: in such a scenario no control
        method can replace grid reinforcement, and it should be known before
        effort goes into a certified shield.
        """
        return self.infeasible_steps == 0

    def summary(self) -> str:
        """One-line summary for logs and scripts."""
        verdict = "feasible" if self.feasible else "INFEASIBLE"
        return (
            f"P4 {verdict}: {self.infeasible_steps}/{self.steps} steps violate, "
            f"vm in [{self.worst_vm_pu[0]:.4f}, {self.worst_vm_pu[1]:.4f}], "
            f"worst loading {self.worst_loading_percent:.1f} %"
        )
