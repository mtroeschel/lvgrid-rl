"""EV charge point: a session-based, deferrable load (M4, step 4.4b, D17).

A vehicle arrives, announces the energy it wants and when it will leave, and
can be charged at any power up to the charge point's rating in between. What
the controller decides is *when* that energy flows; the sessions themselves are
exogenous (``tools/emobpy``, :mod:`lvgrid_rl.data.sources.ev_sessions`). Energy
still missing when the vehicle leaves is **unserved** -- reported in
:class:`AssetOutcome`, priced by the reward. The vehicle leaves on time either
way: the model does not top up on its own in the last steps, so what is
delivered is entirely the controller's doing.

**The action is the power offered to the vehicle**, as a charge point offers it
through the control pilot (IEC 61851): an upper limit that the vehicle draws up
to while it still needs energy. A charge point without a vehicle, or with one
that needs nothing more, draws nothing. That is not a limitation of the action
and is not reported as clipping: it is not the controller proposing something
inadmissible but the vehicle not being there to take it, just as PV does not
count the sun not shining. Only a request beyond the rating is clipped, so
uncontrolled charging is never counted as clipped. For the certified shield
(M9) the setpoint is therefore an upper bound of the injection, never an
underestimate.

**Uncontrolled charging is the neutral action:** the full rating offered all the
time, which is emobpy's ``immediate`` strategy that the sessions come from.

**Information.** The session reaches the model only through the exogenous input
(``ExogenousInput.ev_sessions``), step by step; the model holds no session table
of its own, which would be clairvoyant in a way invariant I3 cannot detect. At
arrival the state takes over the announced energy and departure. That the
departure is known from arrival is an assumption -- users enter it in charging
apps, but not exactly -- and the upper bound of what a controller can know.

**Grid side throughout.** Session energies are what the vehicle draws from the
grid, so charging losses are already in them and the model needs neither a
battery capacity nor an efficiency.

**Sign convention.** Consumer reference: ``p_mw >= 0`` is drawing from the grid.
Vehicle-to-grid is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import ActionSpec, AssetOutcome, Setpoint
from lvgrid_rl.core.schemas import (
    AssetRatings,
    EvSession,
    ExogenousInput,
    GridState,
    Interval,
)

__all__ = ["EvChargerState", "EvCharger"]

_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class EvChargerState:
    """State of a charge point.

    Args:
        asset_id: The asset.
        last_p_mw: Power drawn in the last simulation step.
        connected: Whether a vehicle is connected.
        need_mwh: Energy the connected vehicle still wants.
        arrival_t_index: Arrival step of the connected vehicle's session, which
            tells a new session from the one under way; ``-1`` without one.
        departure_t_index: Its departure step (exclusive); ``-1`` without one.
        remaining_min: Time until it leaves, from the end of the last step.
    """

    asset_id: str
    last_p_mw: float = 0.0
    connected: bool = False
    need_mwh: float = 0.0
    arrival_t_index: int = -1
    departure_t_index: int = -1
    remaining_min: float = 0.0


@dataclass(frozen=True, slots=True)
class EvCharger:
    """A charge point that charges one vehicle at a time, up to its rating.

    Args:
        asset_id: Identifier, ``"load:<index>"`` -- the load element it controls.
        bus: Connection bus.
        ratings: ``p_min_mw = 0``, ``p_max_mw`` the nominal power.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings
    kind: ClassVar[str] = "ev"

    def __post_init__(self) -> None:
        if self.ratings.p_min_mw != 0.0 or self.ratings.p_max_mw <= 0.0:
            raise ValueError("a charge point draws power: need p_min_mw = 0 < p_max_mw")

    # -- sessions -------------------------------------------------------------

    def _session(
        self, s: EvChargerState, x: ExogenousInput
    ) -> tuple[EvSession | None, float]:
        """The session under way in this step and the energy it still needs.

        Raises:
            ValueError: if a new session starts while another is still
                connected, or the input carries a session that is already over.
        """
        session = x.ev_sessions.get(self.asset_id)
        if session is None:
            return None, 0.0
        if not session.arrival_t_index <= x.t_index < session.departure_t_index:
            raise ValueError(
                f"{self.asset_id}: session {session} is not under way at step {x.t_index}"
            )
        if s.connected and s.arrival_t_index == session.arrival_t_index:
            return session, s.need_mwh
        if s.connected:
            raise ValueError(
                f"{self.asset_id}: a session arrives at step "
                f"{session.arrival_t_index} while the one from step "
                f"{s.arrival_t_index} is still connected"
            )
        return session, session.energy_mwh

    def connected_state(
        self, session: EvSession, t_index: int, need_mwh: float, dt_min: float
    ) -> EvChargerState:
        """The state with ``session`` under way before step ``t_index``.

        For an episode that starts in the middle of a session; how much the
        vehicle still needs at that point is the caller's to decide.
        """
        if not session.arrival_t_index <= t_index < session.departure_t_index:
            raise ValueError(f"session {session} is not under way at step {t_index}")
        return EvChargerState(
            asset_id=self.asset_id,
            connected=True,
            need_mwh=need_mwh,
            arrival_t_index=session.arrival_t_index,
            departure_t_index=session.departure_t_index,
            remaining_min=(session.departure_t_index - t_index) * dt_min,
        )

    # -- protocol -------------------------------------------------------------

    def action_spec(self) -> ActionSpec:
        """Power offered, from zero to the rating; neutral is the full rating."""
        return ActionSpec(
            names=("p_mw",),
            bounds=(Interval(0.0, self.ratings.p_max_mw),),
            neutral=(self.ratings.p_max_mw,),
        )

    def initial_state(self, rng: np.random.Generator) -> EvChargerState:
        """No vehicle.

        A session under way at the start is the environment's to set
        (:meth:`connected_state`): it depends on the data, not on chance.
        """
        return EvChargerState(asset_id=self.asset_id)

    def to_setpoint(
        self,
        s: EvChargerState,
        action: np.ndarray,
        info: InformationSet,
        hold_min: float,
    ) -> Setpoint:
        """The power offered over the hold; beyond the rating it is clipped (``p_mw``)."""
        if hold_min <= 0.0:
            raise ValueError("hold_min must be positive")
        requested = float(action[0])
        p = min(max(requested, 0.0), self.ratings.p_max_mw)
        clipping = {"p_mw": p - requested} if abs(p - requested) > _TOL else {}
        return Setpoint(asset_id=self.asset_id, p_mw=p, clipping_info=clipping)

    def limit_to_physics(
        self, s: EvChargerState, sp: Setpoint, x: ExogenousInput, dt_min: float
    ) -> Setpoint:
        """What the vehicle draws of the power offered, in this step.

        Nothing without a vehicle; at most what completes its need within the
        step. Not reported as clipping (module docstring).
        """
        session, need = self._session(s, x)
        cap = need / (dt_min / 60.0) if session is not None else 0.0
        drawn = min(sp.p_mw, cap)
        if drawn == sp.p_mw:
            return sp
        return Setpoint(
            asset_id=sp.asset_id,
            p_mw=drawn,
            q_mvar=sp.q_mvar,
            clipping_info=sp.clipping_info,
        )

    def dynamics(
        self,
        s: EvChargerState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
        dt_min: float,
    ) -> tuple[EvChargerState, AssetOutcome]:
        """Charge over ``dt_min`` and let the vehicle leave when its time is up.

        A vehicle leaves at the end of the last step of its session; what it
        still needs then is unserved. A vehicle that disappears from the input
        before its departure step is treated the same way.

        Raises:
            ValueError: if the setpoint draws more than there is a vehicle to
                take -- :meth:`limit_to_physics` comes first.
        """
        hours = dt_min / 60.0
        session, need = self._session(s, x)
        charged = sp.p_mw * hours
        if session is None:
            if sp.p_mw > _TOL:
                raise ValueError(
                    f"{self.asset_id}: draws {sp.p_mw} MW without a vehicle; "
                    "apply limit_to_physics first"
                )
            unserved = s.need_mwh if s.connected else 0.0
            return (
                EvChargerState(asset_id=self.asset_id),
                AssetOutcome(unserved_energy_mwh=unserved),
            )
        if charged > need + 1e-9:
            raise ValueError(
                f"{self.asset_id}: {sp.p_mw} MW for {dt_min} min exceeds the "
                f"{need} MWh the vehicle needs; apply limit_to_physics first"
            )
        need = max(need - charged, 0.0)
        steps_left = session.departure_t_index - (x.t_index + 1)
        if steps_left <= 0:
            return (
                EvChargerState(asset_id=self.asset_id, last_p_mw=sp.p_mw),
                AssetOutcome(unserved_energy_mwh=need, throughput_energy_mwh=charged),
            )
        return (
            EvChargerState(
                asset_id=self.asset_id,
                last_p_mw=sp.p_mw,
                connected=True,
                need_mwh=need,
                arrival_t_index=session.arrival_t_index,
                departure_t_index=session.departure_t_index,
                remaining_min=steps_left * dt_min,
            ),
            AssetOutcome(throughput_energy_mwh=charged),
        )
