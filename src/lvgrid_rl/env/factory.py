"""Assembling a ready-to-use environment from configuration.

Kept apart from :class:`~lvgrid_rl.env.lv_grid_env.LVGridEnv` on purpose: the
environment should depend on prepared arrays, not on SimBench, so that a later
data source or a surrogate power flow can be substituted without touching it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lvgrid_rl.components.bess import BatteryStorage
from lvgrid_rl.components.ev_charger import EvCharger
from lvgrid_rl.components.heat_pump import HeatPump
from lvgrid_rl.components.pv import PvMode, PvSystem
from lvgrid_rl.core.schemas import AssetRatings, Interval
from lvgrid_rl.data.sources.simbench import AssetCategory, categorize
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSampler, EpisodeSpec
from lvgrid_rl.env.lv_grid_env import EnvConfig, LVGridEnv
from lvgrid_rl.env.splits import WeekSplit
from lvgrid_rl.grid.loader import GridModel, load_grid
from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

__all__ = [
    "StorageSizing",
    "DEFAULT_STORAGE",
    "HeatPumpSizing",
    "DEFAULT_HEAT_PUMPS",
    "EvSizing",
    "DEFAULT_EV",
    "make_env",
    "absolute_profiles",
]


@dataclass(frozen=True, slots=True)
class StorageSizing:
    """How controllable batteries are placed and sized (M4).

    One battery at the bus of every PV system, sized from that system's rated
    power in the scenario-modified grid -- so a scenario that scales PV scales
    the storage with it.

    Args:
        kw_per_kwp: Rated charge and discharge power per kWp of PV.
        kwh_per_kwp: Usable capacity per kWp of PV.
        eta_charge_frac: Charging efficiency.
        eta_discharge_frac: Discharging efficiency. With both at 0.95 the round
            trip is 90 %, the value SimBench gives its own storage units
            (``efficiency_percent`` = 0.95, read here as one way).
        soc_min_frac: Lower operating limit.
        soc_max_frac: Upper operating limit.
        evaluation_soc_frac: State of charge at the start of every evaluation
            week. Fixed so that controllers and seeds start from the same point;
            training draws from the full operating range instead.

    The SimBench grid's own five storage units are not affected. They sit at
    buses without PV, follow published profiles, and stay part of the
    uncontrolled load, so that the uncontrolled grid is the one M3 measured.
    """

    kw_per_kwp: float = 1.0
    kwh_per_kwp: float = 2.0
    eta_charge_frac: float = 0.95
    eta_discharge_frac: float = 0.95
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    evaluation_soc_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.kw_per_kwp <= 0.0 or self.kwh_per_kwp <= 0.0:
            raise ValueError("kw_per_kwp and kwh_per_kwp must be positive")


DEFAULT_STORAGE = StorageSizing()
"""1 kW and 2 kWh per kWp of PV, the M4 working configuration."""


@dataclass(frozen=True, slots=True)
class HeatPumpSizing:
    """How the grid's heat pumps become controllable (M4, D15).

    Every load with a SimBench heat pump profile is replaced by a
    :class:`HeatPump` at the same element. Its thermal demand is the SimBench
    electrical profile times the when2heat COP of its heat source; its buffer
    holds ``buffer_hours`` of rated thermal output, rated thermal output being
    the electrical rating times the heat pump's seasonal performance factor
    over the year.

    Args:
        buffer_hours: Buffer capacity in hours of rated thermal output.
        sink: when2heat heat sink for the COP.
        standing_loss_frac_per_h: Buffer heat loss per hour, as a share of its
            capacity. 0.5 %: a buffer of this size (two hours of rated heat is
            on the order of 1,000 litres over a 20 K band) loses about 3 kWh a
            day, which is 0.5 % of its band capacity per hour.
        evaluation_buffer_frac: Fill level at the start of every evaluation week.
        when2heat_csv: The raw when2heat file, for the first call; afterwards
            the cache is used.
        cache_dir: Where the repaired COP is cached.
    """

    buffer_hours: float = 2.0
    sink: str = "floor"
    standing_loss_frac_per_h: float = 0.005
    evaluation_buffer_frac: float = 0.5
    when2heat_csv: str = "data/raw/when2heat/when2heat-2023-07-27.csv"
    cache_dir: str = "data/cache"

    def __post_init__(self) -> None:
        if self.buffer_hours <= 0.0:
            raise ValueError("buffer_hours must be positive")


DEFAULT_HEAT_PUMPS = HeatPumpSizing()
"""Two hours of rated heat, floor sink, the M4 working configuration (D15)."""


@dataclass(frozen=True, slots=True)
class EvSizing:
    """How the grid's charge points become controllable (M4, D17).

    Every load with a SimBench ``HLS_*`` profile is replaced by an
    :class:`EvCharger` at the same element, at the nominal power in the profile
    name. Its sessions are the emobpy vehicle-year of that charge point, scaled
    to the annual energy of its SimBench curve in the scenario
    (:mod:`lvgrid_rl.data.sources.ev_sessions`), so the uncontrolled grid keeps
    the dataset's energy and gets the vehicles' timing.

    Args:
        run_dir: The emobpy tool's output for this grid.
        manifest: The manifest the files are verified against: the committed
            one by default, so that the data is provably the dataset of record;
            ``None`` for the run's own (synthetic test data).
    """

    run_dir: str = "data/raw/emobpy/2016-home-only"
    manifest: str | None = "configs/ev/emobpy-2016-home-only.manifest.json"


DEFAULT_EV = EvSizing()
"""The 2016 emobpy run, home charging only (D17)."""


def _ev_chargers(model: GridModel, frame, sizing: EvSizing):
    """Charge point models and their session tables on the simulation index.

    Returns:
        ``(assets, sessions)``: the models, and a mapping from asset id to its
        :class:`~lvgrid_rl.data.sources.ev_sessions.SessionTable`.
    """
    import pandas as pd

    from lvgrid_rl.data.sources.ev_sessions import (
        home_sessions,
        read_emobpy_run,
        scale_to_annual,
    )
    from lvgrid_rl.data.sources.simbench import ev_rated_power_kw

    vehicles = read_emobpy_run(sizing.run_dir, sizing.manifest)
    hours = (frame.index[1] - frame.index[0]) / pd.Timedelta("1h")
    assets, sessions = [], {}
    for position, element_index in enumerate(model.net.load.index):
        row = model.net.load.iloc[position]
        profile = str(row.get("profile", "") or "")
        if categorize(profile) is not AssetCategory.EV_CHARGER:
            continue
        asset_id = f"load:{element_index}"
        if asset_id not in vehicles:
            raise ValueError(
                f"{asset_id} has no vehicle in {sizing.run_dir}; the run was made "
                "for another grid (scripts/list_charge_points.py, tools/emobpy)"
            )
        vehicle = vehicles[asset_id]
        p_max_mw = ev_rated_power_kw(profile) / 1000.0
        if abs(vehicle.nominal_power_kw / 1000.0 - p_max_mw) > 1e-9:
            raise ValueError(
                f"{asset_id}: the run charges at {vehicle.nominal_power_kw} kW, "
                f"the grid's profile {profile} says {p_max_mw * 1000} kW"
            )
        table = home_sessions(vehicle.frame, frame.index, p_max_mw)
        target_mwh = float(frame[asset_id].sum() * hours)
        sessions[asset_id] = scale_to_annual(table, target_mwh, frame.index)
        assets.append(
            EvCharger(
                asset_id=asset_id,
                bus=int(row["bus"]),
                ratings=AssetRatings(p_min_mw=0.0, p_max_mw=p_max_mw),
            )
        )
    return assets, sessions


def _heat_pumps(model: GridModel, frame, sizing: HeatPumpSizing, evaluate: bool):
    """Heat pump models, their thermal demand columns and their COP series.

    Returns:
        ``(assets, demand_columns, cop_series)``: the models; a mapping from
        ``"heat:<asset_id>"`` to the thermal demand in MW; a mapping from
        ``"cop:<asset_id>"`` to the COP on the simulation index.
    """
    from lvgrid_rl.data.sources.when2heat import (
        cached_cop,
        cop_on_index,
        heat_source_of,
        thermal_demand_mw,
    )

    cop = cached_cop(sizing.when2heat_csv, sizing.cache_dir)
    band = (
        Interval(sizing.evaluation_buffer_frac, sizing.evaluation_buffer_frac)
        if evaluate
        else None
    )
    assets, demand, ratios = [], {}, {}
    for position, element_index in enumerate(model.net.load.index):
        row = model.net.load.iloc[position]
        profile = str(row.get("profile", "") or "")
        if categorize(profile) is not AssetCategory.HEAT_PUMP:
            continue
        asset_id = f"load:{element_index}"
        electrical = frame[asset_id]
        series = cop_on_index(
            cop.column(heat_source_of(profile), sizing.sink), frame.index
        )
        thermal = thermal_demand_mw(electrical, series)
        rated = float(row["p_mw"])
        spf = float(thermal.sum() / electrical.sum()) if electrical.sum() > 0 else 3.0
        capacity = sizing.buffer_hours * rated * spf
        heat_key, cop_key = f"heat:{asset_id}", f"cop:{asset_id}"
        demand[heat_key] = thermal.to_numpy()
        ratios[cop_key] = series.to_numpy()
        assets.append(
            HeatPump(
                asset_id=asset_id,
                bus=int(row["bus"]),
                ratings=AssetRatings(p_min_mw=0.0, p_max_mw=rated),
                series_id=heat_key,
                cop_key=cop_key,
                capacity_mwh=capacity,
                standing_loss_mw=sizing.standing_loss_frac_per_h * capacity,
                initial_buffer_frac=band,
            )
        )
    return assets, demand, ratios


def _add_storage(
    model: GridModel, pv: list[PvSystem], sizing: StorageSizing, evaluate: bool
):
    """Create one pandapower storage element per PV system and its model.

    Created after the profiles have been read, so the new elements carry no
    profile and do not appear among the uncontrolled series.
    """
    import pandapower as pp

    band = (
        Interval(sizing.evaluation_soc_frac, sizing.evaluation_soc_frac)
        if evaluate
        else None
    )
    batteries: list[BatteryStorage] = []
    for system in pv:
        rated_pv_mw = -system.ratings.p_min_mw
        p_mw = sizing.kw_per_kwp * rated_pv_mw
        capacity_mwh = sizing.kwh_per_kwp * rated_pv_mw
        index = pp.create_storage(
            model.net,
            bus=system.bus,
            p_mw=0.0,
            max_e_mwh=capacity_mwh,
            sn_mva=p_mw,
            name=f"controlled battery at {system.asset_id}",
            type="controlled_bess",
        )
        batteries.append(
            BatteryStorage(
                asset_id=f"storage:{index}",
                bus=system.bus,
                ratings=AssetRatings(p_min_mw=-p_mw, p_max_mw=p_mw, s_max_mva=p_mw),
                capacity_mwh=capacity_mwh,
                soc_min_frac=sizing.soc_min_frac,
                soc_max_frac=sizing.soc_max_frac,
                eta_charge_frac=sizing.eta_charge_frac,
                eta_discharge_frac=sizing.eta_discharge_frac,
                initial_soc_frac=band,
            )
        )
    return batteries


def absolute_profiles(model: GridModel, timebase: TimeBase):
    """Absolute power per element and simulation step, consumer sign convention.

    Taken from the **scenario-modified** grid, so that scaling factors applied by
    the scenario are reflected. Returned resampled onto the simulation step size.
    """
    import pandas as pd
    import simbench as sb

    from lvgrid_rl.data.resample import resample_frame
    from lvgrid_rl.data.sources.simbench import SIMBENCH_TIME_FORMAT
    from lvgrid_rl.data.timebase import to_utc_index

    source_dt_min = 15  # SimBench series resolution
    if source_dt_min % timebase.sim_dt_min != 0:
        raise ValueError(
            f"sim_dt_min={timebase.sim_dt_min} cannot be formed from the "
            f"{source_dt_min}-minute SimBench series without a grid offset. "
            "EN 50160 permits {1, 2, 5, 10}; the source restricts that to "
            "{1, 5}. Both constraints apply at once."
        )

    values = sb.get_absolute_values(model.net, profiles_instead_of_study_cases=True)
    raw_time = model.net["profiles"]["load"]["time"]
    index = to_utc_index(pd.to_datetime(raw_time, format=SIMBENCH_TIME_FORMAT))

    columns: dict[str, np.ndarray] = {}
    for table, sign in (("load", 1.0), ("sgen", -1.0), ("storage", 1.0)):
        key = (table, "p_mw")
        if key not in values or values[key].empty:
            continue
        frame = values[key]
        for position, element_index in enumerate(model.net[table].index):
            columns[f"{table}:{element_index}"] = (
                sign * frame.iloc[:, position].to_numpy()
            )

    frame = pd.DataFrame(columns, index=index)
    frame.index.name = "timestamp_utc"
    return resample_frame(frame, timebase, policies={})


def make_env(
    code: str = "1-LV-rural1--2-sw",
    scenario: str = "moderate_growth",
    split_path: Path | str = "configs/split/1-LV-rural1--2-sw.json",
    set_name: str = "train",
    config: EnvConfig | None = None,
    episode_spec: EpisodeSpec | None = None,
    pv_mode: str = "p_only",
    seed: int | None = None,
    storage: StorageSizing | None = None,
    heat_pumps: HeatPumpSizing | None = None,
    ev: EvSizing | None = None,
) -> LVGridEnv:
    """Build an environment for one grid, scenario and evaluation set.

    Args:
        code: SimBench code of the grid.
        scenario: Name of a scenario in the scenario library.
        split_path: Committed week split for this grid.
        set_name: Which evaluation set to draw episodes from.
        config: Environment configuration.
        episode_spec: Episode configuration.
        pv_mode: Which quantities the PV systems expose. ``p_only`` is the main
            study (decision D4); ``pq`` is needed for reference method B3, whose
            Q(U) characteristic has nothing to act on otherwise.
        seed: Base seed of the environment.
        storage: Controllable batteries at the PV buses, or ``None`` for PV
            curtailment alone -- the M3 setup.
        heat_pumps: Make the grid's heat pumps controllable, or ``None`` to
            leave them on their SimBench profiles as uncontrolled load.
        ev: Make the grid's charge points controllable with emobpy sessions,
            or ``None`` to leave them on their SimBench profiles.
    """
    config = config or EnvConfig()
    timebase = TimeBase(config.sim_dt_min, config.control_dt_min)
    model = load_grid(code, SCENARIO_LIBRARY[scenario])
    frame = absolute_profiles(model, timebase)

    assets: list = []
    for position, element_index in enumerate(model.net.sgen.index):
        row = model.net.sgen.iloc[position]
        if categorize(str(row.get("profile", "") or "")) is not AssetCategory.PV:
            continue
        asset_id = f"sgen:{element_index}"
        rated = abs(float(row["p_mw"]))

        assets.append(
            PvSystem(
                asset_id=asset_id,
                bus=int(row["bus"]),
                mode=PvMode(pv_mode),
                ratings=AssetRatings(
                    p_min_mw=-rated,
                    p_max_mw=0.0,
                    s_max_mva=float(row["sn_mva"]) if "sn_mva" in row else None,
                ),
                series_id=asset_id,
            )
        )

    evaluate = episode_spec is not None and episode_spec.mode is EpisodeMode.EVALUATE
    ratio_series: dict[str, np.ndarray] = {}
    if heat_pumps is not None:
        pumps, demand, ratio_series = _heat_pumps(model, frame, heat_pumps, evaluate)
        assets += pumps
        # Thermal demand travels as a profile series, so that it reaches the
        # heat pump through the exogenous input and the forecast like any other
        # profile (I3). It is not a grid element and is never written to one.
        for key, values in demand.items():
            frame[key] = values

    ev_sessions = None
    if ev is not None:
        chargers, ev_sessions = _ev_chargers(model, frame, ev)
        assets += chargers

    controlled = {a.asset_id for a in assets}
    uncontrolled = {
        name: frame[name].to_numpy()
        for name in frame.columns
        if name not in controlled and not name.startswith("heat:")
    }
    if storage is not None:
        # One battery per PV system -- and only per PV system: the list holds
        # heat pumps too when they are controllable.
        pv = [a for a in assets if a.kind == "pv"]
        assets += _add_storage(model, pv, storage, evaluate)

    split = WeekSplit.from_json(Path(split_path).read_text(encoding="utf-8"))
    sampler = EpisodeSampler(
        split=split,
        set_name=set_name,
        index=frame.index,
        steps_per_day=24 * 60 // config.sim_dt_min,
        n_buses=model.n_evaluated_buses,
        budget_windows=int(0.05 * 1008),
        spec=episode_spec,
    )

    series_ids = tuple(frame.columns)
    return LVGridEnv(
        model=model,
        assets=assets,
        profiles=frame.to_numpy(),
        series_ids=series_ids,
        timestamps=list(frame.index.to_pydatetime()),
        profiles_index=frame.index,
        uncontrolled=uncontrolled,
        sampler=sampler,
        config=config,
        seed=seed,
        ratio_series=ratio_series,
        ev_sessions=ev_sessions,
    )
