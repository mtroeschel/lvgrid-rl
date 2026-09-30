"""Battery storage: the first actuator with a state of its own.

PV curtailment can only cut; a battery can move energy in time. That is the
reason it comes first in M4 (docs/results/m3.md, consequence 1): with curtailment
alone the optimal policy is close to a fine fixed cap, and a learned controller
should separate from a rule where there is something to shift.

It is also the first asset for which invariant I2 carries weight. The state of
charge is the only memory in the model, it lives in :class:`BatteryState`, and
every method of :class:`BatteryStorage` is a pure function of its arguments. A
predictive safety filter can therefore roll a battery forward over a horizon,
branch, and roll again, without the asset object noticing.

**Sign convention.** Consumer reference, as for every asset: ``p_mw > 0`` is
charging from the grid, ``p_mw < 0`` is discharging into it. Power is measured
at the grid terminal (AC side); the efficiencies convert between that and the
energy stored.

**What the model contains** (architecture section 5): state-of-charge
integration with separate charge and discharge efficiencies, operating limits on
the state of charge, power limits from the ratings (which is where a C-rate
limit is expressed), a standing loss, and the terminal throughput as the basis of
a degradation cost. It does not contain a voltage- or temperature-dependent
efficiency, calendar ageing, or reactive power; the first two are refinements
without a question in this project that needs them, the third can be added the
way the PV model has it (decision D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import ActionSpec, AssetOutcome, Setpoint
from lvgrid_rl.core.schemas import AssetRatings, ExogenousInput, GridState, Interval

__all__ = ["BatteryState", "BatteryStorage"]

_TOL = 1e-12
"""Numerical slack when checking energy against its limits."""


@dataclass(frozen=True, slots=True)
class BatteryState:
    """State of a battery: the stored energy, plus the last setpoint.

    The stored energy is kept in MWh rather than as a fraction, so that the state
    is a physical quantity and the capacity stays a parameter of the asset: the
    same state could be handed to a model with a different capacity without
    being reinterpreted. :meth:`BatteryStorage.soc_frac` converts.
    """

    asset_id: str
    energy_mwh: float
    last_p_mw: float = 0.0


@dataclass(frozen=True, slots=True)
class BatteryStorage:
    """A battery storage system behind an inverter.

    Args:
        asset_id: Identifier, e.g. ``"storage:<index>"``.
        bus: Connection bus.
        ratings: Power limits at the grid terminal, consumer convention:
            ``p_min_mw`` is the negative maximum discharge power, ``p_max_mw``
            the maximum charge power. A C-rate limit is expressed here, as
            capacity times C-rate.
        capacity_mwh: Usable nominal capacity.
        soc_min_frac: Lower operating limit of the state of charge.
        soc_max_frac: Upper operating limit of the state of charge.
        eta_charge_frac: Efficiency from grid terminal into storage.
        eta_discharge_frac: Efficiency from storage to grid terminal.
        standing_loss_mw: Constant draw of the system on its own storage --
            battery management, inverter standby. For home storage this is the
            self-discharge that matters (tens of watts); the electrochemical
            self-discharge of lithium cells is two orders of magnitude smaller
            and is not modelled.
        initial_soc_frac: Range from which :meth:`initial_state` draws the state
            of charge. Defaults to the full operating range, so that training
            sees every starting point; evaluation passes a single point.

    **Operating limits against physics.** ``soc_min_frac`` and ``soc_max_frac``
    bound what the controller may do: charging never lifts the energy above the
    upper limit, discharging never takes it below the lower one. The standing
    loss is not controlled and may drain an idle battery below the lower limit,
    down to empty -- as a real system does. The controller then cannot discharge
    until the battery is recharged.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings
    capacity_mwh: float
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    eta_charge_frac: float = 0.95
    eta_discharge_frac: float = 0.95
    standing_loss_mw: float = 0.0
    initial_soc_frac: Interval | None = None
    kind: ClassVar[str] = "bess"

    def __post_init__(self) -> None:
        if not self.ratings.p_min_mw <= 0.0 <= self.ratings.p_max_mw:
            raise ValueError(
                "A battery's power limits must enclose zero: p_min_mw <= 0 "
                f"(discharge) and p_max_mw >= 0 (charge), got {self.ratings.p_bounds}"
            )
        if self.capacity_mwh <= 0.0:
            raise ValueError("capacity_mwh must be positive")
        if not 0.0 <= self.soc_min_frac < self.soc_max_frac <= 1.0:
            raise ValueError("need 0 <= soc_min_frac < soc_max_frac <= 1")
        for name in ("eta_charge_frac", "eta_discharge_frac"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1], got {value}")
        if self.standing_loss_mw < 0.0:
            raise ValueError("standing_loss_mw must not be negative")
        if self.initial_soc_frac is not None and not (
            0.0 <= self.initial_soc_frac.lo and self.initial_soc_frac.hi <= 1.0
        ):
            raise ValueError("initial_soc_frac must lie within [0, 1]")

    # -- derived quantities ---------------------------------------------------

    @property
    def energy_min_mwh(self) -> float:
        """Lower operating limit in MWh."""
        return self.soc_min_frac * self.capacity_mwh

    @property
    def energy_max_mwh(self) -> float:
        """Upper operating limit in MWh."""
        return self.soc_max_frac * self.capacity_mwh

    def soc_frac(self, s: BatteryState) -> float:
        """State of charge as a fraction of the capacity."""
        return s.energy_mwh / self.capacity_mwh

    def feasible_power(self, s: BatteryState, hold_min: float) -> Interval:
        """Terminal power that can be held for ``hold_min`` from state ``s``.

        Within the rated limits, and such that a constant setpoint neither lifts
        the energy above the upper operating limit nor takes it below the lower
        one by the end of the hold. The standing loss is counted against
        discharging and ignored for charging, so the interval is conservative
        in both directions: every setpoint inside it stays within the limits.

        This is the feasible set the certified shield (M9) will need per asset,
        and it is a box -- which is why the action is a power and not a
        state-of-charge target (invariant I4).
        """
        hours = hold_min / 60.0
        if hours <= 0.0:
            raise ValueError("hold_min must be positive")
        headroom = max(self.energy_max_mwh - s.energy_mwh, 0.0)
        reserve = max(
            s.energy_mwh - self.energy_min_mwh - self.standing_loss_mw * hours, 0.0
        )
        charge = min(self.ratings.p_max_mw, headroom / (self.eta_charge_frac * hours))
        discharge = min(-self.ratings.p_min_mw, reserve * self.eta_discharge_frac / hours)
        return Interval(-discharge, charge)

    # -- protocol -------------------------------------------------------------

    def action_spec(self) -> ActionSpec:
        """Action space in physical units (invariant I4).

        The rated limits, not the currently feasible ones: a state-dependent
        action space would make the normalisation non-affine. The state of
        charge enters in :meth:`to_setpoint`, where it is reported.
        """
        return ActionSpec(
            names=("p_mw",), bounds=(self.ratings.p_bounds,), neutral=(0.0,)
        )

    def initial_state(self, rng: np.random.Generator) -> BatteryState:
        """Initial state, the state of charge drawn from ``initial_soc_frac``."""
        band = self.initial_soc_frac or Interval(self.soc_min_frac, self.soc_max_frac)
        soc = band.lo if band.width == 0.0 else float(rng.uniform(band.lo, band.hi))
        return BatteryState(asset_id=self.asset_id, energy_mwh=soc * self.capacity_mwh)

    def to_setpoint(
        self,
        s: BatteryState,
        action: np.ndarray,
        info: InformationSet,
        hold_min: float,
    ) -> Setpoint:
        """Project the action onto what can be held for ``hold_min``.

        Two limitations, reported separately so that the cause stays visible:
        ``p_mw`` for a request beyond the rated power, ``p_mw_soc`` for one the
        state of charge cannot sustain over the hold. A policy that keeps asking
        to charge a full battery shows up in the second key, not in the first.
        """
        requested = float(action[0])
        clipping: dict[str, float] = {}

        rated = self.ratings.p_bounds
        p_mw = min(max(requested, rated.lo), rated.hi)
        if abs(p_mw - requested) > _TOL:
            clipping["p_mw"] = p_mw - requested

        feasible = self.feasible_power(s, hold_min)
        limited = min(max(p_mw, feasible.lo), feasible.hi)
        if abs(limited - p_mw) > _TOL:
            clipping["p_mw_soc"] = limited - p_mw

        return Setpoint(asset_id=self.asset_id, p_mw=limited, clipping_info=clipping)

    def limit_to_physics(
        self, s: BatteryState, sp: Setpoint, x: ExogenousInput, dt_min: float
    ) -> Setpoint:
        """The battery management cut-off, applied per simulation step.

        A setpoint from :meth:`to_setpoint` is feasible for its whole hold, so
        this binds only for setpoints formed elsewhere -- a baseline, a safety
        mechanism, a hypothetical roll-out with a longer step. When it does, the
        limitation is added to ``clipping_info`` under ``p_mw_bms``: unlike the
        irradiance limit of a PV system it was foreseeable from the state, so it
        counts as an infeasible proposal.
        """
        feasible = self.feasible_power(s, dt_min)
        limited = min(max(sp.p_mw, feasible.lo), feasible.hi)
        if abs(limited - sp.p_mw) <= _TOL:
            return sp
        clipping = dict(sp.clipping_info)
        clipping["p_mw_bms"] = clipping.get("p_mw_bms", 0.0) + (limited - sp.p_mw)
        return Setpoint(
            asset_id=sp.asset_id,
            p_mw=limited,
            q_mvar=sp.q_mvar,
            clipping_info=clipping,
        )

    def dynamics(
        self,
        s: BatteryState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
        dt_min: float,
    ) -> tuple[BatteryState, AssetOutcome]:
        """Integrate the stored energy over ``dt_min`` (invariant I2).

        Raises:
            ValueError: if the setpoint would move the energy past an operating
                limit in its own direction. Clamping here would be the silent
                limitation invariant I4 forbids; a caller that skipped
                :meth:`limit_to_physics` gets an error, not a quietly different
                trajectory.
        """
        hours = dt_min / 60.0
        p = sp.p_mw
        if p >= 0.0:
            stored = p * self.eta_charge_frac * hours
            conversion_loss = p * hours - stored
        else:
            stored = p / self.eta_discharge_frac * hours
            conversion_loss = -stored - (-p) * hours

        after_flow = s.energy_mwh + stored
        if p > 0.0 and after_flow > self.energy_max_mwh + _TOL:
            raise ValueError(
                f"{self.asset_id}: charging at {p} MW for {dt_min} min exceeds "
                "the upper state-of-charge limit; apply limit_to_physics first"
            )
        if p < 0.0 and after_flow < self.energy_min_mwh - _TOL:
            raise ValueError(
                f"{self.asset_id}: discharging at {p} MW for {dt_min} min goes "
                "below the lower state-of-charge limit; apply limit_to_physics first"
            )

        standing = min(self.standing_loss_mw * hours, max(after_flow, 0.0))
        energy = max(after_flow - standing, 0.0)
        switched = int(abs(p - s.last_p_mw) > 1e-9)
        return (
            BatteryState(asset_id=self.asset_id, energy_mwh=energy, last_p_mw=p),
            AssetOutcome(
                throughput_energy_mwh=abs(p) * hours,
                loss_energy_mwh=conversion_loss + standing,
                switching_count=switched,
            ),
        )
