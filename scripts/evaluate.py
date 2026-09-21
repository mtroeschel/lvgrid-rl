#!/usr/bin/env python3
"""Evaluate a saved checkpoint and the reference methods on one evaluation set.

    uv run python scripts/evaluate.py --run-dir results/m3/<run_id> --set test

Separate from training on purpose. The acceptance figure belongs on the **test**
weeks (architecture section 10), the numbers printed at the end of a training run
are on the validation weeks, and a policy should not have to be retrained to be
assessed on a different set.

Episodes come from :class:`EpisodeMode.EVALUATE`: complete calendar weeks, in
fixed order, each week once, starting from a zero budget. That is what makes the
EN 50160 pass rate standard-conforming and what lets results from different seeds
be compared -- sampled episodes differ per seed and cannot be aggregated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lvgrid_rl.agents.policy_controller import PolicyController
from lvgrid_rl.baselines.methods import (
    DoNothing,
    FixedCap,
    PUDroop,
    RandomController,
    asset_bus_positions,
)
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import make_env
from lvgrid_rl.env.lv_grid_env import EnvConfig
from lvgrid_rl.eval.runner import run_controller


def load_tuned(path: Path) -> dict:
    """Load the tuned baseline parameters, refusing to guess."""
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned baseline parameters at {path}. Run\n"
            "    uv run python scripts/tune_baselines.py\n"
            "first; comparing against untuned references is not admissible."
        )
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def main() -> None:
    """Evaluate policy and baselines on one set and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="set_name",
        default="test",
        choices=("train", "val", "test", "stress", "holdout"),
    )
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="moderate_growth")
    parser.add_argument("--tuned", type=Path, default=Path("configs/baseline/tuned.json"))
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output file; defaults to <run-dir>/eval/<set>.json",
    )
    args = parser.parse_args()

    from lvgrid_rl.agents.factory import SUPPORTED_ALGOS

    tuned = load_tuned(args.tuned)
    cap = tuned["fixed_cap"]["params"]["cap"]
    droop = tuned["p_u_droop"]["params"]

    checkpoint = args.run_dir / "checkpoints" / "final.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    module_name, _, attribute = SUPPORTED_ALGOS[args.algo].rpartition(".")
    algo_class = getattr(__import__(module_name, fromlist=[attribute]), attribute)
    model = algo_class.load(checkpoint, device="cpu")

    # Complete weeks, fixed order, zero budget: the standard-conforming setting.
    spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)

    def build():
        return make_env(
            code=args.code,
            scenario=args.scenario,
            set_name=args.set_name,
            config=EnvConfig(),
            episode_spec=spec,
            seed=args.seed,
        )

    positions = asset_bus_positions(build())
    controllers = {
        "policy": lambda e: PolicyController(model),
        "do_nothing": lambda e: DoNothing(e.mapper),
        "random": lambda e: RandomController(e.mapper, seed=args.seed),
        f"fixed_cap({cap})": lambda e: FixedCap(e.mapper, cap=cap),
        f"p_u_droop({droop['v_start']}/{droop['v_max']})": lambda e: PUDroop(
            e.mapper, positions, droop["v_start"], droop["v_max"]
        ),
    }

    rows = []
    for name, build_controller in controllers.items():
        env = build()
        result = run_controller(
            env, build_controller(env), name, args.set_name, seed=args.seed
        )
        rows.append(result.summary())

    header = (
        f"{'controller':24s} {'pass rate':>10s} {'k95':>6s} {'k100':>6s} "
        f"{'overload':>10s} {'curtailed MWh':>14s}"
    )
    print(f"set      {args.set_name} ({rows[0]['episodes']} complete weeks)")
    print(f"run      {args.run_dir.name}\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['controller']:24s} {row['pass_rate']:10.4f} {row['k95_windows']:6.0f} "
            f"{row['k100_windows']:6.0f} {row['overload_cost']:10.3f} "
            f"{row['curtailed_mwh']:14.4f}"
        )

    out = args.out or (args.run_dir / "eval" / f"{args.set_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "set": args.set_name,
                "episode_mode": "evaluate",
                "baseline_params": tuned,
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
