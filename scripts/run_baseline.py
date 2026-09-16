#!/usr/bin/env python3
"""Quasi-static annual simulation without control (reference case B0).

    uv run python scripts/run_baseline.py --scenario moderate_growth --stride 3

Runs the grid over the SimBench year at its native 15-minute resolution,
assesses EN 50160 per connection point and week, and performs the P4 pre-check:
does a feasible fallback action exist -- PV fully curtailed, flexible loads at
minimum? Without one, no controller can guarantee anything.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from lvgrid_rl.core.protocols import Setpoint
from lvgrid_rl.data.sources.simbench import AssetCategory, categorize
from lvgrid_rl.grid.loader import load_grid
from lvgrid_rl.grid.metrics import FallbackReport, violation_metrics
from lvgrid_rl.grid.powerflow import PandapowerEngine
from lvgrid_rl.grid.pq import K100_BAND_PU, EN50160Evaluator
from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

STEPS_PER_YEAR = 35136
SAMPLES_PER_WINDOW = 1  # 15-minute source data; see note in the output


def _profiles(net):
    import simbench as sb

    values = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    return (
        values[("load", "p_mw")],
        values[("load", "q_mvar")],
        values[("sgen", "p_mw")],
        values[("storage", "p_mw")],
    )


def main() -> None:
    """Run the baseline and print EN 50160 and P4 results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="reference", choices=SCENARIO_LIBRARY)
    parser.add_argument("--stride", type=int, default=1, help="use every n-th step")
    parser.add_argument("--steps", type=int, default=STEPS_PER_YEAR)
    args = parser.parse_args()

    spec = SCENARIO_LIBRARY[args.scenario]
    model = load_grid(args.code, spec)
    lp, lq, sp, stp = _profiles(model.net)
    engine = PandapowerEngine(model)
    positions = model.evaluated_bus_positions
    evaluator = EN50160Evaluator(model.n_evaluated_buses, SAMPLES_PER_WINDOW)

    print(f"grid      {model.code}")
    print(f"scenario  {spec.name}: {spec.notes}")
    print(f"modified  {model.scenario_modifications}")
    print(f"repaired  {model.zip_rows_repaired} ZIP load rows")
    print(f"assessed  {model.n_evaluated_buses} connection points\n")

    # Which loads may the fallback switch off? Heat pumps and charge points are
    # controllable devices in the sense of section 14a EnWG; households are not.
    flexible_loads = [
        categorize(str(p or "")) in (AssetCategory.HEAT_PUMP, AssetCategory.EV_CHARGER)
        for p in model.net.load["profile"]
    ]
    print(f"flexible  {sum(flexible_loads)} of {len(flexible_loads)} loads\n")

    worst_v = [9.0, 0.0]
    worst_load = 0.0
    overload_steps = 0
    fb_infeasible = 0
    fb_first: int | None = None
    fb_v = [9.0, 0.0]
    fb_load = 0.0
    n = 0
    t0 = time.perf_counter()

    for t in range(0, args.steps, args.stride):
        lpv, lqv, spv = lp.loc[t].values, lq.loc[t].values, sp.loc[t].values
        stv = stp.loc[t].values if len(stp.columns) else []
        setpoints: dict[str, Setpoint] = {}
        for pos, idx in enumerate(model.net.load.index):
            setpoints[f"load:{idx}"] = Setpoint(
                f"load:{idx}", float(lpv[pos]), float(lqv[pos])
            )
        for pos, idx in enumerate(model.net.sgen.index):
            setpoints[f"sgen:{idx}"] = Setpoint(f"sgen:{idx}", -float(spv[pos]))
        # Storage must be driven from its profile too. Left out, the elements
        # keep their nominal values from the dataset for the whole run -- in
        # this grid a constant 367 kW of injection, which silently distorts
        # every loading figure.
        for pos, idx in enumerate(model.net.storage.index):
            setpoints[f"storage:{idx}"] = Setpoint(f"storage:{idx}", float(stv[pos]))

        state = engine.run(setpoints, t)
        if not state.converged:
            continue
        n += 1
        evaluator.add_sample(state.vm_pu[positions])
        m = violation_metrics(state, positions)
        worst_v[0] = min(worst_v[0], m.min_vm_pu)
        worst_v[1] = max(worst_v[1], m.max_vm_pu)
        worst_load = max(
            worst_load, m.max_line_loading_percent, m.max_trafo_loading_percent
        )
        if m.overload_excess_percent > 0:
            overload_steps += 1

        # P4: the fallback action as defined in section 6.9 -- PV fully
        # curtailed AND flexible loads at minimum. Curtailing PV alone is a
        # weaker fallback and would understate feasibility.
        fallback = dict(setpoints)
        for idx in model.net.sgen.index:
            fallback[f"sgen:{idx}"] = Setpoint(f"sgen:{idx}", 0.0)
        for pos, idx in enumerate(model.net.load.index):
            if flexible_loads[pos]:
                fallback[f"load:{idx}"] = Setpoint(f"load:{idx}", 0.0, 0.0)
        # Storage stays free to support the grid, as the definition foresees.
        fb_state = engine.run_hypothetical(fallback, t)
        if fb_state.converged:
            fm = violation_metrics(fb_state, positions)
            fb_v[0] = min(fb_v[0], fm.min_vm_pu)
            fb_v[1] = max(fb_v[1], fm.max_vm_pu)
            fb_load = max(
                fb_load, fm.max_line_loading_percent, fm.max_trafo_loading_percent
            )
            if (
                fm.min_vm_pu < K100_BAND_PU[0]
                or fm.max_vm_pu > K100_BAND_PU[1]
                or fm.overload_excess_percent > 0
            ):
                fb_infeasible += 1
                fb_first = t if fb_first is None else fb_first

    evaluator.finalize()
    runtime = time.perf_counter() - t0

    print(f"steps            {n} converged, {runtime / max(n, 1) * 1000:.1f} ms each")
    print(f"voltage          [{worst_v[0]:.4f}, {worst_v[1]:.4f}] pu")
    print(f"worst loading    {worst_load:.1f} %")
    print(f"overload steps   {overload_steps} ({overload_steps / max(n, 1):.1%})")
    print(f"\nEN 50160 pass rate  {evaluator.pass_rate():.4f}")
    print(f"worst bus P95       {evaluator.worst_bus_p95():.4f} pu")
    complete = [w for w in evaluator.weeks if w.complete]
    if complete:
        util = np.concatenate([w.budget_utilisation() for w in complete])
        print(f"budget utilisation  max {util.max():.3f}, mean {util.mean():.3f}")
    print(f"weeks assessed      {len(complete)} complete")

    report = FallbackReport(
        steps=n,
        infeasible_steps=fb_infeasible,
        worst_vm_pu=(fb_v[0], fb_v[1]),
        worst_loading_percent=fb_load,
        first_infeasible_step=fb_first,
    )
    print(f"\n{report.summary()}")
    if args.stride > 1:
        print(
            f"\nNote: stride={args.stride}, so the weekly windows are not "
            "standard-conforming. Use stride=1 for reported numbers."
        )


if __name__ == "__main__":
    main()
