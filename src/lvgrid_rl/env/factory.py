"""Assembling a ready-to-use environment from configuration.

Kept apart from :class:`~lvgrid_rl.env.lv_grid_env.LVGridEnv` on purpose: the
environment should depend on prepared arrays, not on SimBench, so that a later
data source or a surrogate power flow can be substituted without touching it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lvgrid_rl.components.pv import PvMode, PvSystem
from lvgrid_rl.data.sources.simbench import AssetCategory, categorize
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.episodes import EpisodeSampler, EpisodeSpec
from lvgrid_rl.env.lv_grid_env import EnvConfig, LVGridEnv
from lvgrid_rl.env.splits import WeekSplit
from lvgrid_rl.grid.loader import GridModel, load_grid
from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

__all__ = ["make_env", "absolute_profiles"]


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
    """
    config = config or EnvConfig()
    timebase = TimeBase(config.sim_dt_min, config.control_dt_min)
    model = load_grid(code, SCENARIO_LIBRARY[scenario])
    frame = absolute_profiles(model, timebase)

    assets: list[PvSystem] = []
    for position, element_index in enumerate(model.net.sgen.index):
        row = model.net.sgen.iloc[position]
        if categorize(str(row.get("profile", "") or "")) is not AssetCategory.PV:
            continue
        asset_id = f"sgen:{element_index}"
        rated = abs(float(row["p_mw"]))
        from lvgrid_rl.core.schemas import AssetRatings

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

    controlled = {a.asset_id for a in assets}
    uncontrolled = {
        name: frame[name].to_numpy() for name in frame.columns if name not in controlled
    }

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
        uncontrolled=uncontrolled,
        sampler=sampler,
        config=config,
        seed=seed,
    )
