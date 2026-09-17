"""PV system: the first actuator, and the only one in M3.

That M3 restricts itself to PV curtailment is not a didactic simplification. The
M2 measurements show that under ``moderate_growth`` the fallback action brings
transformer loading from 280 % down to 55.6 % and voltage to 1.051 pu, so
curtailing PV alone resolves both the voltage and the thermal problem. M3 is
therefore a complete control problem with a genuine cost trade-off: every
kilowatt-hour curtailed is a kilowatt-hour of lost yield.

Reactive power is implemented as an option because reference method B3 (Q(U)
characteristic per VDE-AR-N 4105) needs it, but the main study is P-dominated
(decision D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import ActionSpec, AssetOutcome, Setpoint
from lvgrid_rl.core.schemas import AssetRatings, ExogenousInput, GridState, Interval

__all__ = ["PvMode", "PvState", "PvSystem"]


class PvMode(StrEnum):
    """Which quantities the agent may set on a PV inverter."""

    P_ONLY = "p_only"
    Q_ONLY = "q_only"
    PQ = "pq"


@dataclass(frozen=True, slots=True)
class PvState:
    """State of a PV system.

    A PV system carries no dynamics of its own -- the available power follows
    from irradiance, and curtailment has no memory. The type exists anyway, for
    two reasons: it keeps the asset protocol uniform, and ``last_p_mw`` is needed
    for the action smoothness term and for detecting switching behaviour.
    """

    asset_id: str
    last_p_mw: float = 0.0
    last_q_mvar: float = 0.0


@dataclass(frozen=True, slots=True)
class PvSystem:
    """A curtailable PV system.

    Args:
        asset_id: Identifier of the form ``"sgen:<index>"``.
        bus: Connection bus.
        ratings: Limits in the consumer sign convention, so ``p_min_mw`` is the
            negative rated power and ``p_max_mw`` is zero.
        mode: Which quantities are controllable.
        series_id: Name of the profile column supplying the available power.

    **Sign convention.** Everything here is consumer reference: infeed is
    negative. ``p_available_mw`` is therefore ``<= 0``, and curtailment moves the
    setpoint *towards* zero.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings
    series_id: str
    mode: PvMode = PvMode.P_ONLY

    def action_spec(self) -> ActionSpec:
        """Action space in physical units (invariant I4).

        The bounds are the rated limits, not the currently available power. A
        state-dependent action space would make the mapping non-affine and the
        feasible set non-box, which is exactly what invariant I4 forbids: the
        certified feasible set is formulated in injections. Limiting to what is
        currently available happens in :meth:`to_setpoint` and is reported.
        """
        names: list[str] = []
        bounds: list[Interval] = []
        if self.mode in (PvMode.P_ONLY, PvMode.PQ):
            names.append("p_mw")
            bounds.append(Interval(self.ratings.p_min_mw, self.ratings.p_max_mw))
        if self.mode in (PvMode.Q_ONLY, PvMode.PQ):
            q_max = self.ratings.s_max_mva or abs(self.ratings.p_min_mw)
            names.append("q_mvar")
            bounds.append(Interval(-q_max, q_max))
        return ActionSpec(names=tuple(names), bounds=tuple(bounds))

    def initial_state(self, rng: np.random.Generator) -> PvState:
        """Initial state. Deterministic; the generator is unused here."""
        return PvState(asset_id=self.asset_id)

    def to_setpoint(
        self, s: PvState, action: np.ndarray, info: InformationSet
    ) -> Setpoint:
        """Map an action onto an operating point.

        The available power is taken from the forecast entry for this asset's
        series, which at decision time is what a real controller would have. Any
        limitation is reported in ``clipping_info`` rather than applied
        silently, so that a policy systematically proposing infeasible actions
        stays visible.
        """
        spec = self.action_spec()
        available = float(info.forecast[self.series_id][0])
        clipping: dict[str, float] = {}

        p_mw = 0.0
        q_mvar = 0.0
        for value, name, bound in zip(action, spec.names, spec.bounds, strict=True):
            requested = float(value)
            limited = min(max(requested, bound.lo), bound.hi)
            if name == "p_mw":
                # Cannot feed in more than is available: available is <= 0, so
                # the setpoint must not be below it.
                limited = max(limited, available)
                p_mw = limited
            else:
                q_mvar = limited
            if abs(limited - requested) > 1e-12:
                clipping[name] = limited - requested

        if self.mode is PvMode.PQ and self.ratings.s_max_mva is not None:
            apparent = float(np.hypot(p_mw, q_mvar))
            if apparent > self.ratings.s_max_mva:
                # Inverter capability diagram: active power has priority,
                # reactive power yields. Reported like any other limitation.
                room = self.ratings.s_max_mva**2 - p_mw**2
                allowed = float(np.sqrt(max(room, 0.0)))
                clipping["q_mvar_s_max"] = np.sign(q_mvar) * allowed - q_mvar
                q_mvar = float(np.sign(q_mvar) * allowed)

        return Setpoint(
            asset_id=self.asset_id,
            p_mw=p_mw,
            q_mvar=q_mvar,
            clipping_info=clipping,
        )

    def apply_availability(self, sp: Setpoint, x: ExogenousInput) -> Setpoint:
        """Limit a setpoint to the power actually available at this instant.

        This is physics, not control, and therefore lives outside
        :meth:`to_setpoint`. The distinction matters once ``control_dt`` spans
        several simulation steps: the controller decides once, from what it knows
        at the decision point, and that decision is then held for the whole
        control step. Irradiance does not hold still, so a setpoint formed at the
        start of the step can exceed what the array can deliver three minutes
        later -- and writing it unchanged would inject power that does not exist.

        The limitation is not reported as clipping: it is not the controller
        proposing something inadmissible, it is the weather.
        """
        available = float(x.realized_mw[x.series_ids.index(self.series_id)])
        # Both are negative in the consumer convention, so the physical limit is
        # the larger (less negative) of the two.
        limited = max(sp.p_mw, available)
        if limited == sp.p_mw:
            return sp
        return Setpoint(
            asset_id=sp.asset_id,
            p_mw=limited,
            q_mvar=sp.q_mvar,
            clipping_info=sp.clipping_info,
        )

    def dynamics(
        self,
        s: PvState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
    ) -> tuple[PvState, AssetOutcome]:
        """Advance the state as a pure function (invariant I2).

        The curtailed energy is the gap between what was available and what was
        fed in, over one simulation step. Both quantities are negative in the
        consumer convention, so the difference is taken on magnitudes.
        """
        available = float(x.realized_mw[x.series_ids.index(self.series_id)])
        # Power, not energy: the caller multiplies by the step duration, because
        # only it knows the step size.
        curtailed_mw = max(abs(available) - abs(sp.p_mw), 0.0)
        switched = int(abs(sp.p_mw - s.last_p_mw) > 1e-9)
        return (
            PvState(
                asset_id=self.asset_id,
                last_p_mw=sp.p_mw,
                last_q_mvar=sp.q_mvar,
            ),
            AssetOutcome(
                curtailed_energy_mwh=curtailed_mw,
                switching_count=switched,
            ),
        )
