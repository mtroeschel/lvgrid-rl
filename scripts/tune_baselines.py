#!/usr/bin/env python3
"""Parameter search for the reference methods, on the validation weeks.

    uv run python scripts/tune_baselines.py --weeks 2
    uv run python scripts/tune_baselines.py --flex --workers 12   # B5, B6 (M4 4.5b)

The deadbands and slopes of the droop methods are implicitly designed for a hard
+/-10 % criterion. Under the EN 50160 percentile criterion the optimum is
different, because tolerance is permitted. Without this search the RL agent wins
against a deliberately badly tuned reference, which is the most attackable point
in this literature (architecture section 6.6, consequence 5).

**Objective.** Lexicographic rather than a weighted sum: first minimise the
number of violating assessment windows, then the curtailed energy. A scalarised
objective would need weights, and choosing them would smuggle the very trade-off
the study is supposed to measure into the baseline tuning.

**B5 and B6** (``--flex``) are tuned in the full M4 configuration -- batteries,
heat pumps and charge points -- with PV on the already tuned P(U) rule, and
lexicographically on violating windows, then the overload integral (within a
tolerance of 1 %), then the cost of control: curtailment, unserved and deferred
EV energy and comfort deviation, at their reward weights. Load rules act on
the overload that the PV objective does not see, and deferred EV energy counts
like unserved, or the search would prefer the rule that moves load past the
end of the episode.
``--flex`` keeps the PV results in the file and adds its own.

Tuning runs on the **validation** weeks. Using the test weeks would make the
comparison meaningless, and using the training weeks would give the baselines an
advantage the agent does not have.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lvgrid_rl.baselines.flexibility import En14aDimming, GreedyLocal, local_series
from lvgrid_rl.baselines.methods import FixedCap, PUDroop, QUDroop, asset_bus_positions
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import (
    DEFAULT_EV,
    DEFAULT_HEAT_PUMPS,
    DEFAULT_STORAGE,
    make_env,
)
from lvgrid_rl.env.reward import RewardConfig
from lvgrid_rl.eval.runner import run_controller

CAPS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
V_STARTS = (1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07)
V_SPANS = (0.02, 0.03, 0.04, 0.06)
V_MAX_LIMIT = 1.10
"""Upper bound of the K95 band.

Combinations whose ``v_max`` lies above it are skipped. A characteristic that
only reaches full curtailment beyond the admissible band cannot hold the band --
the first search picked ``v_max = 1.12`` precisely because the lexicographic
objective saw no violations on the tuning episodes and then minimised
curtailment, which rewards the weakest possible intervention.
"""


def _score(summary: dict) -> tuple[float, float]:
    """Lexicographic score: violations first, curtailment second."""
    return (summary["k95_windows"], summary["curtailed_mwh"])


LOADING_ON = (70.0, 80.0, 90.0, 100.0, 110.0)
"""B5: transformer loading that starts dimming, in percent."""
SURPLUS_THRESHOLDS = (0.0, 0.2, 0.4)
"""B6: share of the bus's PV rating below which the battery does not charge."""
MARGINS_H = (0.25, 0.5, 1.0, 2.0)
"""B6: laxity at which a vehicle starts charging at full power."""
OVERLOAD_TOLERANCE_FRAC = 0.01
"""Overload integrals within this share of the best count as equal.

The overload is continuous, and a strict lexicographic order would let a
difference in the third decimal -- 5.195 against 5.201 in the first search --
decide over a cost twelve per cent apart. Violating windows are counts and need
no tolerance."""


def _flex_cost(summary: dict) -> float:
    """Cost of control at the reward weights; deferred EV energy as unserved."""
    weights = RewardConfig().objective
    return (
        -weights["pv_curtailment"].weight * summary["curtailed_mwh"]
        - weights["ev_unserved"].weight
        * (summary["ev_unserved_mwh"] + summary["ev_deferred_mwh"])
        - weights["hp_comfort"].weight * summary["hp_comfort_kh"]
    )


def _flex_score(summary: dict) -> tuple[float, float, float]:
    """Violations, then overload, then the cost of control."""
    return (summary["k95_windows"], summary["overload_cost"], _flex_cost(summary))


def _flex_run(method: str, params: dict, droop: dict, weeks: int, days: int) -> dict:
    """One B5 or B6 setting over the validation episodes, in its own process."""
    spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=days, randomise_budget=False)
    env = make_env(
        seed=1,
        episode_spec=spec,
        set_name="val",
        storage=DEFAULT_STORAGE,
        heat_pumps=DEFAULT_HEAT_PUMPS,
        ev=DEFAULT_EV,
    )
    pv = PUDroop(env.mapper, asset_bus_positions(env), droop["v_start"], droop["v_max"])
    if method == "en14a_dimming":
        controller = En14aDimming(env.mapper, env.assets, pv, **params)
    else:
        controller = GreedyLocal(
            env.mapper,
            env.assets,
            pv,
            local_series(env),
            hold_h=env.config.control_dt_min / 60.0,
            **params,
        )
    summary = run_controller(
        env, controller, method, "val", n_episodes=weeks, seed=1
    ).summary()
    return {"method": method, "params": params, "summary": summary}


def _tune_flex(args, tuned: dict) -> dict:
    """Search B5 and B6; returns their entries for the results."""
    droop = tuned["results"]["p_u_droop"]["params"]
    settings = [
        ("en14a_dimming", {"loading_on_percent": level}) for level in LOADING_ON
    ] + [
        ("greedy_local", {"surplus_threshold_frac": theta, "margin_h": margin})
        for theta in SURPLUS_THRESHOLDS
        for margin in MARGINS_H
    ]
    print(f"{len(settings)} settings on {args.workers} workers, P(U) {droop}")
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(
            pool.map(
                _flex_run,
                *zip(
                    *[(m, p, droop, args.weeks, args.days) for m, p in settings],
                    strict=True,
                ),
            )
        )
    print(f"{'method':48s} {'k95':>5s} {'overload':>9s} {'cost':>8s}")
    results = {}
    for method in ("en14a_dimming", "greedy_local"):
        scored = [(_flex_score(r["summary"]), r) for r in runs if r["method"] == method]
        for score, run in scored:
            print(
                f"{method + str(run['params']):48s} {score[0]:5.0f} "
                f"{score[1]:9.3f} {score[2]:8.3f}"
            )
        fewest = min(score[0] for score, _ in scored)
        scored = [(s, r) for s, r in scored if s[0] == fewest]
        lowest = min(score[1] for score, _ in scored)
        scored = [
            (s, r)
            for s, r in scored
            if s[1] <= lowest * (1.0 + OVERLOAD_TOLERANCE_FRAC) + 1e-9
        ]
        score, run = min(scored, key=lambda item: (item[0][2], item[0][1]))
        results[method] = {
            "params": run["params"],
            "k95": score[0],
            "overload": score[1],
            "cost": score[2],
            "pv_rule": {"p_u_droop": droop},
            "overload_tolerance_frac": OVERLOAD_TOLERANCE_FRAC,
            "assets": "storage, heat pumps, charge points (M4 defaults)",
        }
    return results


def main() -> None:
    """Search the parameter grids and write the best settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="validation episodes to average over; too few and the search picks "
        "a setting that happened to suit one week",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="days per episode. The grid has 27 settings, so the whole search "
        "costs roughly weeks * days * 96 * 27 control steps",
    )
    parser.add_argument("--out", type=Path, default=Path("configs/baseline/tuned.json"))
    parser.add_argument("--pv-mode", default="p_only", choices=("p_only", "pq"))
    parser.add_argument(
        "--flex",
        action="store_true",
        help="tune B5 and B6 in the full M4 configuration, keeping the PV results",
    )
    parser.add_argument("--workers", type=int, default=4, help="for --flex")
    args = parser.parse_args()

    if args.flex:
        tuned = json.loads(args.out.read_text(encoding="utf-8"))
        tuned["results"].update(_tune_flex(args, tuned))
        tuned["flex_note"] = (
            "B5/B6: full M4 assets, PV on the tuned P(U); lexicographic on "
            "violating windows, overload integral, then curtailment + unserved "
            "and deferred EV energy + comfort at the reward weights."
        )
        tuned["flex_weeks"] = args.weeks
        tuned["flex_days_per_episode"] = args.days
        args.out.write_text(
            json.dumps(tuned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwritten: {args.out}")
        return

    spec = EpisodeSpec(
        mode=EpisodeMode.TRAIN, length_days=args.days, randomise_budget=False
    )

    def build():
        return make_env(seed=1, episode_spec=spec, set_name="val", pv_mode=args.pv_mode)

    positions = asset_bus_positions(build())
    results: dict[str, dict] = {}

    print(f"{'method':22s} {'k95':>6s} {'curtailed MWh':>14s}")
    print(
        "Objective: violating windows first, curtailed energy second. If every "
        "setting scores zero violations, the episodes did not challenge the "
        "search and the result is noise -- raise --weeks and --days."
    )
    print("-" * 46)

    best: tuple[tuple[float, float], dict] | None = None
    for cap in CAPS:
        env = build()
        controller = FixedCap(env.mapper, cap=cap)
        summary = run_controller(
            env, controller, "fixed_cap", "val", n_episodes=args.weeks, seed=1
        ).summary()
        score = _score(summary)
        print(f"{'fixed_cap(' + str(cap) + ')':22s} {score[0]:6.0f} {score[1]:14.3f}")
        if best is None or score < best[0]:
            best = (score, {"cap": cap})
    results["fixed_cap"] = {"params": best[1], "k95": best[0][0], "curtailed": best[0][1]}

    for name, factory in (
        ("p_u_droop", lambda e, s, m: PUDroop(e.mapper, positions, s, m)),
        *(
            [("q_u_droop", lambda e, s, m: QUDroop(e.mapper, positions, s, m))]
            if args.pv_mode == "pq"
            else []
        ),
    ):
        best = None
        for v_start in V_STARTS:
            for span in V_SPANS:
                if round(v_start + span, 3) > V_MAX_LIMIT:
                    continue
                env = build()
                controller = factory(env, v_start, v_start + span)
                summary = run_controller(
                    env, controller, name, "val", n_episodes=args.weeks, seed=1
                ).summary()
                score = _score(summary)
                label = f"{name}({v_start},{round(v_start + span, 3)})"
                print(f"{label:22s} {score[0]:6.0f} {score[1]:14.3f}")
                if best is None or score < best[0]:
                    best = (
                        score,
                        {"v_start": v_start, "v_max": round(v_start + span, 3)},
                    )
        results[name] = {
            "params": best[1],
            "k95": best[0][0],
            "curtailed": best[0][1],
        }

    print("\nbest settings:")
    for name, entry in results.items():
        print(f"  {name:12s} {entry['params']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "note": (
                    "Tuned on the validation weeks with a lexicographic objective: "
                    "violating assessment windows first, curtailed energy second."
                ),
                "weeks": args.weeks,
                "days_per_episode": args.days,
                "pv_mode": args.pv_mode,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
