#!/usr/bin/env python3
"""Survey the SimBench low-voltage grids for where the control problem actually is.

    uv run python scripts/survey_grids.py --samples 200

Runs a sampled quasi-static annual simulation without any control and reports,
per grid, the voltage range, the worst transformer and line loading, and how
many sampled steps violate the +/-10 % voltage band. This is the reference case
B0 and the basis for choosing a working grid.
"""

from __future__ import annotations

import argparse

import numpy as np

from lvgrid_rl.core.protocols import Setpoint
from lvgrid_rl.grid.loader import load_grid
from lvgrid_rl.grid.powerflow import PandapowerEngine

DEFAULT_CODES = (
    "1-LV-rural1--2-sw",
    "1-LV-rural2--2-sw",
    "1-LV-rural3--2-sw",
    "1-LV-semiurb4--2-sw",
    "1-LV-semiurb5--2-sw",
    "1-LV-urban6--2-sw",
)
STEPS_PER_YEAR = 35136


def survey(code: str, samples: int) -> dict[str, float]:
    """Sample one grid over the year and collect the extreme values."""
    import simbench as sb

    model = load_grid(code)
    values = sb.get_absolute_values(model.net, profiles_instead_of_study_cases=True)
    lp, lq = values[("load", "p_mw")], values[("load", "q_mvar")]
    sp = values[("sgen", "p_mw")]
    engine = PandapowerEngine(model)
    positions = model.evaluated_bus_positions

    stride = max(STEPS_PER_YEAR // samples, 1)
    vmin, vmax, trafo, line, violations, converged = 9.0, 0.0, 0.0, 0.0, 0, 0
    for t in range(0, STEPS_PER_YEAR, stride):
        setpoints: dict[str, Setpoint] = {}
        lpv, lqv, spv = lp.loc[t].values, lq.loc[t].values, sp.loc[t].values
        for pos, idx in enumerate(model.net.load.index):
            setpoints[f"load:{idx}"] = Setpoint(
                f"load:{idx}", float(lpv[pos]), float(lqv[pos])
            )
        for pos, idx in enumerate(model.net.sgen.index):
            setpoints[f"sgen:{idx}"] = Setpoint(f"sgen:{idx}", -float(spv[pos]))
        state = engine.run(setpoints, t)
        if not state.converged:
            continue
        converged += 1
        v = state.vm_pu[positions]
        vmin, vmax = min(vmin, float(np.nanmin(v))), max(vmax, float(np.nanmax(v)))
        trafo = max(trafo, float(np.nanmax(state.trafo_loading_percent)))
        line = max(line, float(np.nanmax(state.line_loading_percent)))
        if np.nanmax(v) > 1.10 or np.nanmin(v) < 0.90:
            violations += 1

    return {
        "buses": len(model.net.bus),
        "connection_points": model.n_evaluated_buses,
        "length_km": float(model.net.line.length_km.sum()),
        "trafo_kva": float(model.net.trafo.sn_mva.sum() * 1000),
        "vmin_pu": vmin,
        "vmax_pu": vmax,
        "trafo_max_percent": trafo,
        "line_max_percent": line,
        "voltage_violations": violations,
        "converged": converged,
    }


def main() -> None:
    """Print the survey table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--codes", nargs="*", default=list(DEFAULT_CODES))
    args = parser.parse_args()

    header = (
        f"{'grid':22s} {'bus':>4s} {'cp':>4s} {'km':>6s} {'trafo':>7s} "
        f"{'vmin':>7s} {'vmax':>7s} {'trafo%':>7s} {'line%':>7s} {'U-viol':>8s}"
    )
    print(header)
    print("-" * len(header))
    for code in args.codes:
        r = survey(code, args.samples)
        print(
            f"{code:22s} {r['buses']:4.0f} {r['connection_points']:4.0f} "
            f"{r['length_km']:6.2f} {r['trafo_kva']:6.0f}k "
            f"{r['vmin_pu']:7.4f} {r['vmax_pu']:7.4f} "
            f"{r['trafo_max_percent']:7.1f} {r['line_max_percent']:7.1f} "
            f"{r['voltage_violations']:4.0f}/{r['converged']:.0f}"
        )


if __name__ == "__main__":
    main()
