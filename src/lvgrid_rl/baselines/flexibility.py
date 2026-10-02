"""Reference methods for the flexible assets: B5 and B6 (M4, step 4.5b).

B2 to B4 act on PV only, and against them a controller with batteries, heat
pumps and charge points would be compared with references that leave all of
that idle. These two rules use the flexibility the way it is used in practice
today, each with its own parameter search (``scripts/tune_baselines.py``).

Both leave PV to the **tuned P(U) characteristic** (B4): on their own they do not
act on PV-driven overvoltage, the main problem of the working scenario, and would
lose to B4 trivially. What the comparison then shows is what the load rule adds.

* **B5** ``en14a_dimming`` -- grid-oriented control of controllable consumers
  under §14a EnWG (BNetzA, BK6-22-300): when the transformer is congested *by
  load*, charge points and heat pumps are dimmed to the guaranteed minimum of
  4.2 kW per device (heat pumps above 11 kW to 40 % of their rating). Otherwise
  everything runs as without control: thermostats, charging on arrival, idle
  batteries.
* **B6** ``greedy_local`` -- what each household would do for itself: the
  battery stores the local PV surplus above a threshold and covers the local
  load, the heat pump pre-heats with the surplus, and vehicles charge **as late
  as necessary**, earlier only from the surplus.

"Local" is the connection point: the forecast of the uncontrolled series at the
asset's own bus -- household loads and PV available power -- which is the
information a home energy management system has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from lvgrid_rl.baselines.methods import PUDroop
from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.env.actions import ActionMapper

__all__ = [
    "EN14A_MIN_MW",
    "En14aDimming",
    "GreedyLocal",
    "local_series",
    "flex_baselines",
]

FLEXIBLE_KINDS = frozenset({"bess", "hp", "ev"})

EN14A_MIN_MW = 0.0042
"""Guaranteed minimum power of a controllable consumer under §14a EnWG."""

EN14A_LARGE_HP_MW = 0.011
"""Heat pumps above this rating are dimmed to a share of it instead."""

EN14A_LARGE_HP_FRAC = 0.4


def en14a_cap_mw(kind: str, rated_mw: float) -> float:
    """The power a device may still draw while dimmed."""
    if kind == "hp" and rated_mw > EN14A_LARGE_HP_MW:
        return EN14A_LARGE_HP_FRAC * rated_mw
    return EN14A_MIN_MW


def local_series(env) -> dict[int, tuple[str, ...]]:
    """Per bus, the uncontrolled series and PV profiles connected to it.

    What a home energy management system at that connection point sees. Built
    once from the grid model, outside the control loop.
    """
    net = env.model.net
    controlled = {a.asset_id for a in env.assets if a.kind != "pv"}
    out: dict[int, list[str]] = {}
    for table in ("load", "sgen", "storage"):
        for index, bus in net[table]["bus"].items():
            series = f"{table}:{index}"
            if series in env.series_ids and series not in controlled:
                out.setdefault(int(bus), []).append(series)
    return {bus: tuple(series) for bus, series in out.items()}


def _bus_balance(info: InformationSet, series: Sequence[str]) -> float:
    """Forecast net load of the uncontrolled elements at a bus, consumer sign."""
    return float(sum(info.forecast[s][0] for s in series))


@dataclass
class En14aDimming:
    """B5 -- §14a dimming of heat pumps and charge points under load congestion.

    Dimming starts when the transformer loading reaches ``loading_on_percent``
    while the grid imports power, and ends when it falls below
    ``loading_on_percent - hysteresis_percent`` or the grid exports. At midday
    the transformer is loaded by PV back-feed; dimming load then would make it
    worse, which is why export never triggers.

    Args:
        mapper: The environment's action mapper.
        assets: The environment's assets, in mapper order.
        droop: The tuned P(U) rule for the PV systems.
        loading_on_percent: Transformer loading that starts dimming.
        hysteresis_percent: How far below it dimming ends.
    """

    mapper: ActionMapper
    assets: Sequence
    droop: PUDroop
    loading_on_percent: float = 100.0
    hysteresis_percent: float = 10.0
    name: str = "en14a_dimming"
    _active: bool = field(default=False, init=False)

    def reset(self, info: InformationSet) -> None:
        """Not dimming at the start of an episode."""
        self._active = False
        self.droop.reset(info)

    def act(self, info: InformationSet) -> np.ndarray:
        """The P(U) action for PV; uncontrolled behaviour, dimmed while congested."""
        loading = float(info.measurements["trafo_loading_percent_max"])
        importing = float(info.measurements["p_slack_mw"]) > 0.0
        if not importing or loading < self.loading_on_percent - self.hysteresis_percent:
            self._active = False
        elif loading >= self.loading_on_percent:
            self._active = True

        physical = self.mapper.to_physical(self.droop.act(info))
        if not self._active:
            return self.mapper.to_normalised(physical)
        cursor = 0
        for asset, spec in zip(self.assets, self.mapper.specs, strict=True):
            if asset.kind in ("hp", "ev"):
                cap = en14a_cap_mw(asset.kind, asset.ratings.p_max_mw)
                physical[cursor] = min(physical[cursor], cap)
            cursor += spec.dim
        return self.mapper.to_normalised(physical)


@dataclass
class GreedyLocal:
    """B6 -- local self-consumption and charging as late as necessary.

    Per bus, the forecast PV surplus over the uncontrolled load is handed out in
    a fixed order: heat pumps first (pre-heating, while the buffer is below its
    thermostat's off level), then vehicles, then the battery, which takes only
    the part above ``surplus_threshold_frac`` of the PV rating at the bus -- zero
    is pure self-consumption, above zero it shaves the infeed peak. The battery
    covers the net load of the uncontrolled elements at its bus from storage;
    not that of heat pumps and vehicles, whose offered power can exceed what
    they draw and would turn the discharge into export. A vehicle gets the full rating as
    soon as its laxity -- time to departure minus the time full power needs --
    falls to ``margin_h`` plus one control step, and before that only surplus.
    An empty charge point offers its rating, so that a vehicle arriving within
    the hold starts at once.

    Args:
        mapper: The environment's action mapper.
        assets: The environment's assets, in mapper order.
        droop: The tuned P(U) rule for the PV systems.
        local: Uncontrolled series per bus (:func:`local_series`).
        hold_h: Duration of one control step.
        surplus_threshold_frac: Share of the bus's PV rating below which the
            battery does not charge.
        margin_h: Laxity at which a vehicle starts charging at full power.
    """

    mapper: ActionMapper
    assets: Sequence
    droop: PUDroop
    local: Mapping[int, tuple[str, ...]]
    hold_h: float = 0.25
    surplus_threshold_frac: float = 0.0
    margin_h: float = 1.0
    name: str = "greedy_local"
    _pv_rating_mw: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for asset in self.assets:
            if asset.kind == "pv":
                self._pv_rating_mw[asset.bus] = self._pv_rating_mw.get(
                    asset.bus, 0.0
                ) + abs(asset.ratings.p_min_mw)

    def reset(self, info: InformationSet) -> None:
        """Memoryless apart from the droop rule."""
        self.droop.reset(info)

    def act(self, info: InformationSet) -> np.ndarray:
        """Hand out the local surplus, then cover the local load."""
        physical = self.mapper.to_physical(self.droop.act(info))
        balance = {bus: _bus_balance(info, s) for bus, s in self.local.items()}
        surplus = {bus: max(-b, 0.0) for bus, b in balance.items()}
        offsets = np.cumsum([0] + [spec.dim for spec in self.mapper.specs])
        order = {"hp": 0, "ev": 1, "bess": 2}
        flexible = sorted(
            (i for i, a in enumerate(self.assets) if a.kind in order),
            key=lambda i: order[self.assets[i].kind],
        )
        for i in flexible:
            asset = self.assets[i]
            state = info.asset_states[asset.asset_id]
            left = surplus.get(asset.bus, 0.0)
            at = offsets[i]
            if asset.kind == "hp":
                if (
                    left >= asset.p_min_running_mw
                    and asset.buffer_frac(state) < asset.thermostat_off_frac
                ):
                    boost = min(left, asset.ratings.p_max_mw)
                    physical[at] = max(physical[at], boost)
                    surplus[asset.bus] = left - boost
            elif asset.kind == "ev":
                physical[at] = self._ev(asset, state, left)
                if state.connected:
                    # What the vehicle can take, not what is offered to it.
                    drawn = min(physical[at], state.need_mwh / self.hold_h)
                    surplus[asset.bus] = max(left - drawn, 0.0)
            else:
                physical[at] = self._battery(asset, state, left, balance)
                if physical[at] > 0.0:
                    surplus[asset.bus] = left - physical[at]
        return self.mapper.to_normalised(physical)

    def _ev(self, asset, state, surplus_mw: float) -> float:
        if not state.connected:
            # A vehicle may arrive within the hold. Offering nothing would make
            # it wait for the next decision -- up to a control step, which a
            # tight session cannot spare (measured: 9 sessions short in the
            # validation episodes); the next decision holds it back.
            return asset.ratings.p_max_mw
        if state.need_mwh <= 0.0:
            return 0.0
        rated = asset.ratings.p_max_mw
        laxity_h = state.remaining_min / 60.0 - state.need_mwh / rated
        if laxity_h - self.hold_h <= self.margin_h:
            return rated
        # From the surplus only, and no more than the vehicle can take.
        return min(surplus_mw, rated, state.need_mwh / self.hold_h)

    def _battery(self, asset, state, surplus_mw: float, balance) -> float:
        feasible = asset.feasible_power(state, self.hold_h * 60.0)
        threshold = self.surplus_threshold_frac * self._pv_rating_mw.get(asset.bus, 0.0)
        charge = max(surplus_mw - threshold, 0.0)
        if charge > 0.0:
            return min(charge, feasible.hi)
        load = max(balance.get(asset.bus, 0.0), 0.0)
        return max(-load, feasible.lo)


def flex_baselines(results: Mapping, positions: tuple[int, ...], env) -> dict:
    """Builders for B5 and B6 from the tuned parameters, by results-table name.

    Empty when the environment has no flexible asset or the parameters have not
    been tuned: an untuned flexibility rule is not offered as a reference.
    """
    if not FLEXIBLE_KINDS & set(env.mapper.kinds or ()):
        return {}
    out = {}
    for method in ("en14a_dimming", "greedy_local"):
        entry = results.get(method)
        if entry is None:
            continue
        droop = entry["pv_rule"]["p_u_droop"]
        params = dict(entry["params"])
        label = ",".join(str(v) for v in params.values())

        def build(e, method=method, droop=droop, params=params):
            pv = PUDroop(e.mapper, positions, droop["v_start"], droop["v_max"])
            if method == "en14a_dimming":
                return En14aDimming(e.mapper, e.assets, pv, **params)
            return GreedyLocal(
                e.mapper,
                e.assets,
                pv,
                local_series(e),
                hold_h=e.config.control_dt_min / 60.0,
                **params,
            )

        out[f"{method}({label})"] = build
    return out
