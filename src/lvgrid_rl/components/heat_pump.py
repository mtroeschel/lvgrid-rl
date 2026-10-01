"""Heat pump with a buffer store: a shiftable load with state (stage 1, D15).

The heat pump serves a thermal demand from its buffer store and refills the store
electrically. What the controller decides is *when* the store is refilled; the
demand itself is exogenous. That makes the heat pump the second asset that can
move energy in time, after the battery -- with a twist the battery does not
have: the electricity needed for the same heat depends on the hour, through the
coefficient of performance.

**Stage 1 (architecture section 5, decision D15).**

* The buffer is an energy reservoir over an admissible temperature band. Its
  state is the stored heat above the bottom of the band, in MWh; full is the top
  of the band. Heat drawn below the bottom lowers the temperature below the band
  -- a comfort violation, counted in kelvin-hours -- down to a floor one band
  width below, past which demand goes unserved.
* The COP is an exogenous time series (when2heat, matched to the heat source),
  not a function of the buffer temperature.
* Operation is modulating between ``min_modulation_frac`` of rated power and
  rated power, or off, with minimum run and idle times and a maximum blocking
  duration. The feasible set ``{0} ∪ [P_min, P_max]`` is not convex; the
  certified shield (M9) will need a convex inner approximation or a mixed-integer
  treatment, and the projection here is reported so that its cost stays visible.

**Information.** The thermal demand arrives as a profile series in the
exogenous input and its forecast, the COP as a ratio
(``ExogenousInput.realized_ratio``, ``InformationSet.forecast``). The model never
reads a time series of its own: a heat pump that looked up tomorrow's demand in
an array it holds would be clairvoyant in a way the information-ordering
invariant (I3) cannot see.

**Defaults are starting values, not calibrated ones:** 20 minutes minimum run and
idle time, at most 120 minutes blocked while there is demand, a standing loss of
1 % of the band capacity per hour (set by the environment), a 20 K band.

**Sign convention.** Consumer reference: ``p_mw >= 0`` is the electrical draw.
Thermal quantities are positive heat flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.protocols import ActionSpec, AssetOutcome, Setpoint
from lvgrid_rl.core.schemas import AssetRatings, ExogenousInput, GridState, Interval

__all__ = ["HeatPumpState", "HeatPump"]

_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class HeatPumpState:
    """State of a heat pump and its buffer.

    Args:
        asset_id: The asset.
        energy_mwh: Heat stored above the bottom of the admissible band.
            Negative when the buffer has cooled below the band.
        last_p_mw: Last electrical setpoint.
        running: Whether the compressor is on.
        state_duration_min: Time since the last switch, for the minimum run
            and idle times and the blocking limit.
    """

    asset_id: str
    energy_mwh: float
    last_p_mw: float = 0.0
    running: bool = False
    state_duration_min: float = 0.0


@dataclass(frozen=True, slots=True)
class HeatPump:
    """A modulating heat pump with a buffer store.

    Args:
        asset_id: Identifier, ``"load:<index>"`` -- the load element it controls.
        bus: Connection bus.
        ratings: Electrical limits, ``p_min_mw = 0``, ``p_max_mw`` the rating.
        series_id: Profile of the thermal demand, in MW of heat.
        cop_key: Key of the COP in ``realized_ratio`` and in the forecast.
        capacity_mwh: Heat the band holds from bottom to top.
        band_k: Width of the admissible temperature band, for converting a heat
            deficit into a temperature shortfall.
        min_modulation_frac: Lowest power while running, as a share of rating.
        min_run_min: Minimum time on once switched on.
        min_idle_min: Minimum time off once switched off.
        max_block_min: Longest time the heat pump may be kept off while there is
            heat demand; after that it is switched on.
        standing_loss_mw: Heat lost from the buffer to its surroundings.
        thermostat_on_frac: Default controller: switch on below this fill level.
        thermostat_off_frac: Default controller: switch off above this one.
        initial_buffer_frac: Range of the initial fill level; ``None`` draws
            from the whole band.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings
    series_id: str
    cop_key: str
    capacity_mwh: float
    band_k: float = 20.0
    min_modulation_frac: float = 0.3
    min_run_min: float = 20.0
    min_idle_min: float = 20.0
    max_block_min: float = 120.0
    standing_loss_mw: float = 0.0
    thermostat_on_frac: float = 0.4
    thermostat_off_frac: float = 0.9
    initial_buffer_frac: Interval | None = None
    kind: ClassVar[str] = "hp"

    def __post_init__(self) -> None:
        if self.ratings.p_min_mw != 0.0 or self.ratings.p_max_mw <= 0.0:
            raise ValueError("a heat pump draws power: need p_min_mw = 0 < p_max_mw")
        if self.capacity_mwh <= 0.0 or self.band_k <= 0.0:
            raise ValueError("capacity_mwh and band_k must be positive")
        if not 0.0 < self.min_modulation_frac <= 1.0:
            raise ValueError("min_modulation_frac must lie in (0, 1]")
        if min(self.min_run_min, self.min_idle_min, self.max_block_min) < 0.0:
            raise ValueError("durations must not be negative")
        if self.max_block_min < self.min_idle_min:
            raise ValueError("max_block_min must not be below min_idle_min")
        if self.standing_loss_mw < 0.0:
            raise ValueError("standing_loss_mw must not be negative")
        if not 0.0 <= self.thermostat_on_frac < self.thermostat_off_frac <= 1.0:
            raise ValueError("need 0 <= thermostat_on_frac < thermostat_off_frac <= 1")

    # -- derived quantities ---------------------------------------------------

    @property
    def p_min_running_mw(self) -> float:
        """Lowest electrical power while running."""
        return self.min_modulation_frac * self.ratings.p_max_mw

    @property
    def floor_mwh(self) -> float:
        """Lowest buffer state; heat drawn below it goes unserved."""
        return -self.capacity_mwh

    def buffer_frac(self, s: HeatPumpState) -> float:
        """Fill level of the band; below zero when cooled below the band."""
        return s.energy_mwh / self.capacity_mwh

    def shortfall_k(self, energy_mwh: float) -> float:
        """Temperature below the band for a given buffer state."""
        return max(-energy_mwh, 0.0) / self.capacity_mwh * self.band_k

    def _max_power(
        self, energy_mwh: float, demand_mw: float, cop: float, hours: float
    ) -> float:
        """Electrical power that fills the band exactly to the top in ``hours``."""
        headroom = self.capacity_mwh - energy_mwh
        heat = headroom / hours + demand_mw + self.standing_loss_mw
        return min(self.ratings.p_max_mw, max(heat / cop, 0.0))

    def _quantise(self, p_mw: float) -> float:
        """Onto ``{0} ∪ [P_min, P_max]``: below P_min, the nearer of 0 and P_min."""
        if p_mw <= _TOL:
            return 0.0
        if p_mw < self.p_min_running_mw:
            return self.p_min_running_mw if p_mw >= 0.5 * self.p_min_running_mw else 0.0
        return p_mw

    # -- protocol -------------------------------------------------------------

    def action_spec(self) -> ActionSpec:
        """Electrical power from zero to rated (invariant I4).

        The neutral value is "off". It is the static fallback only: what a heat
        pump does when nobody intervenes is not a fixed power but its own
        thermostat, :meth:`default_action`.
        """
        return ActionSpec(
            names=("p_mw",),
            bounds=(Interval(0.0, self.ratings.p_max_mw),),
            neutral=(0.0,),
        )

    def initial_state(self, rng: np.random.Generator) -> HeatPumpState:
        """Initial fill level drawn from ``initial_buffer_frac``; compressor off.

        The idle time starts as satisfied, so the first decision is free.
        """
        band = self.initial_buffer_frac or Interval(0.0, 1.0)
        frac = band.lo if band.width == 0.0 else float(rng.uniform(band.lo, band.hi))
        return HeatPumpState(
            asset_id=self.asset_id,
            energy_mwh=frac * self.capacity_mwh,
            state_duration_min=self.min_idle_min,
        )

    def default_action(self, s: HeatPumpState, info: InformationSet) -> np.ndarray:
        """The heat pump's own thermostat: what "do nothing" means for it.

        Hysteresis on the fill level: on at rated power below
        ``thermostat_on_frac``, off above ``thermostat_off_frac``, otherwise
        keep running or keep idling. Physical units, like an action.
        """
        frac = self.buffer_frac(s)
        threshold = self.thermostat_off_frac if s.running else self.thermostat_on_frac
        on = frac < threshold
        return np.array([self.ratings.p_max_mw if on else 0.0])

    def to_setpoint(
        self,
        s: HeatPumpState,
        action: np.ndarray,
        info: InformationSet,
        hold_min: float,
    ) -> Setpoint:
        """Project the action onto what the heat pump may and can do.

        Every limitation under its own key, in the order applied:

        * ``p_mw`` -- beyond the rating;
        * ``p_mw_modulation`` -- between zero and the minimum modulation;
        * ``p_mw_buffer`` -- more heat than the buffer can take over the hold,
          judged from the current demand and COP forecast;
        * ``p_mw_min_idle`` -- switching on before the idle time is over;
        * ``p_mw_min_run`` -- switching off before the run time is over (not
          enforced against a full buffer: the high-limit cut-out wins);
        * ``p_mw_max_block`` -- kept off longer than allowed while there is
          demand.
        """
        hours = hold_min / 60.0
        if hours <= 0.0:
            raise ValueError("hold_min must be positive")
        requested = float(action[0])
        clipping: dict[str, float] = {}

        def record(key: str, before: float, after: float) -> float:
            if abs(after - before) > _TOL:
                clipping[key] = clipping.get(key, 0.0) + (after - before)
            return after

        p = record("p_mw", requested, min(max(requested, 0.0), self.ratings.p_max_mw))
        p = record("p_mw_modulation", p, self._quantise(p))

        demand = float(info.forecast[self.series_id][0])
        cop = float(info.forecast[self.cop_key][0])
        cap = self._max_power(s.energy_mwh, demand, cop, hours)
        room = cap >= self.p_min_running_mw
        if p > cap:
            p = record("p_mw_buffer", p, cap if room else 0.0)

        if not s.running and s.state_duration_min < self.min_idle_min and p > 0.0:
            p = record("p_mw_min_idle", p, 0.0)
        if s.running and s.state_duration_min < self.min_run_min and p == 0.0 and room:
            p = record("p_mw_min_run", p, self.p_min_running_mw)
        if (
            not s.running
            and s.state_duration_min >= self.max_block_min
            and p == 0.0
            and demand > 0.0
            and room
        ):
            p = record("p_mw_max_block", p, self.p_min_running_mw)

        return Setpoint(asset_id=self.asset_id, p_mw=p, clipping_info=clipping)

    def limit_to_physics(
        self, s: HeatPumpState, sp: Setpoint, x: ExogenousInput, dt_min: float
    ) -> Setpoint:
        """The buffer's high-limit cut-out, per simulation step.

        The projection in :meth:`to_setpoint` works from the forecast at the
        decision; demand inside the hold can fall below it, and then a setpoint
        that looked feasible would overheat the buffer. This reduces it -- to
        zero if what remains is below the minimum modulation -- and reports
        ``p_mw_buffer_cutoff``.
        """
        if sp.p_mw <= 0.0:
            return sp
        hours = dt_min / 60.0
        demand = float(x.realized_mw[x.series_ids.index(self.series_id)])
        cop = float(x.realized_ratio[self.cop_key])
        cap = self._max_power(s.energy_mwh, demand, cop, hours)
        if sp.p_mw <= cap + _TOL:
            return sp
        limited = cap if cap >= self.p_min_running_mw else 0.0
        clipping = dict(sp.clipping_info)
        clipping["p_mw_buffer_cutoff"] = clipping.get("p_mw_buffer_cutoff", 0.0) + (
            limited - sp.p_mw
        )
        return Setpoint(
            asset_id=sp.asset_id, p_mw=limited, q_mvar=sp.q_mvar, clipping_info=clipping
        )

    def dynamics(
        self,
        s: HeatPumpState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
        dt_min: float,
    ) -> tuple[HeatPumpState, AssetOutcome]:
        """Integrate the buffer over ``dt_min`` (invariant I2).

        Comfort deviation is the temperature below the band integrated over the
        step, by the trapezoidal rule between start and end. Heat demand below
        the floor of the buffer is unserved.

        Raises:
            ValueError: if the setpoint would heat the buffer past the top of the
                band -- the silent limitation invariant I4 forbids would be to
                clamp here; :meth:`limit_to_physics` comes first.
        """
        hours = dt_min / 60.0
        p = sp.p_mw
        cop = float(x.realized_ratio[self.cop_key])
        demand = float(x.realized_mw[x.series_ids.index(self.series_id)])
        heat_in = p * cop * hours
        loss = self.standing_loss_mw * hours
        energy = s.energy_mwh + heat_in - demand * hours - loss
        if p > 0.0 and energy > self.capacity_mwh + 1e-9:
            raise ValueError(
                f"{self.asset_id}: {p} MW for {dt_min} min overheats the buffer; "
                "apply limit_to_physics first"
            )
        # Only rounding can be left above the top here; anything more raised.
        energy = min(energy, self.capacity_mwh)
        unserved = max(self.floor_mwh - energy, 0.0)
        energy = max(energy, self.floor_mwh)

        comfort_kh = (
            0.5 * (self.shortfall_k(s.energy_mwh) + self.shortfall_k(energy)) * hours
        )
        running = p > 0.0
        duration = s.state_duration_min + dt_min if running == s.running else dt_min
        return (
            HeatPumpState(
                asset_id=self.asset_id,
                energy_mwh=energy,
                last_p_mw=p,
                running=running,
                state_duration_min=duration,
            ),
            AssetOutcome(
                unserved_energy_mwh=unserved,
                comfort_deviation_kh=comfort_kh,
                loss_energy_mwh=loss,
                switching_count=int(running != s.running),
            ),
        )
