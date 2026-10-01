#!/usr/bin/env python3
"""Write the grid's charge points to ``configs/ev/charge_points.json``.

    uv run python scripts/list_charge_points.py

The emobpy tool (``tools/emobpy``) runs in a separate, frozen environment
without SimBench, so it reads the charge points from this file rather than from
the grid. One entry per charge point: the load element it is, its bus, its
SimBench profile, its **nominal** power from the profile name (decision D17:
the scenario's growth factor stays on the energy, not on the power) and the
annual energy of its SimBench curve in the scenario, to which the generated
sessions are scaled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lvgrid_rl.data.sources.simbench import AssetCategory, categorize, ev_rated_power_kw
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.factory import absolute_profiles
from lvgrid_rl.grid.loader import load_grid
from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY


def main() -> None:
    """List the charge points of one grid and scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="moderate_growth")
    parser.add_argument("--out", type=Path, default=Path("configs/ev/charge_points.json"))
    args = parser.parse_args()

    model = load_grid(args.code, SCENARIO_LIBRARY[args.scenario])
    timebase = TimeBase(5, 15)
    frame = absolute_profiles(model, timebase)
    hours = timebase.sim_dt_min / 60.0
    entries = []
    for position, element_index in enumerate(model.net.load.index):
        row = model.net.load.iloc[position]
        profile = str(row.get("profile", "") or "")
        if categorize(profile) is not AssetCategory.EV_CHARGER:
            continue
        asset_id = f"load:{element_index}"
        entries.append(
            {
                "asset_id": asset_id,
                "bus": int(row["bus"]),
                "simbench_profile": profile,
                "nominal_power_kw": ev_rated_power_kw(profile),
                "annual_energy_mwh": round(float(frame[asset_id].sum() * hours), 6),
            }
        )
    payload = {"grid": args.code, "scenario": args.scenario, "charge_points": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.out} ({len(entries)} charge points)")


if __name__ == "__main__":
    main()
