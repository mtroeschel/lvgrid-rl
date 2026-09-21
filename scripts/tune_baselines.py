#!/usr/bin/env python3
"""Parameter search for the reference methods, on the validation weeks.

    uv run python scripts/tune_baselines.py --weeks 2

The deadbands and slopes of the droop methods are implicitly designed for a hard
+/-10 % criterion. Under the EN 50160 percentile criterion the optimum is
different, because tolerance is permitted. Without this search the RL agent wins
against a deliberately badly tuned reference, which is the most attackable point
in this literature (architecture section 6.6, consequence 5).

**Objective.** Lexicographic rather than a weighted sum: first minimise the
number of violating assessment windows, then the curtailed energy. A scalarised
objective would need weights, and choosing them would smuggle the very trade-off
the study is supposed to measure into the baseline tuning.

Tuning runs on the **validation** weeks. Using the test weeks would make the
comparison meaningless, and using the training weeks would give the baselines an
advantage the agent does not have.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lvgrid_rl.baselines.methods import FixedCap, PUDroop, QUDroop, asset_bus_positions
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import make_env
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
    args = parser.parse_args()

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
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
