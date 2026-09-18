#!/usr/bin/env python3
"""Train an agent on the low-voltage grid environment.

    uv run python scripts/train.py --steps 50000 --workers 8

Writes a run directory containing the resolved configuration, the run manifest,
checkpoints, TensorBoard events and an evaluation of the trained policy against
the validation weeks.

**On the training budget.** Compute is dominated by the power flow, not by the
network: at roughly 21 ms per power flow and three power flows per decision, one
control step costs about 63 ms. A budget is therefore better counted in power
flows than in control steps. 50,000 control steps are 150,000 power flows, about
an hour on one worker and a few minutes across eight.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lvgrid_rl.agents.factory import AgentSpec, make_agent
from lvgrid_rl.agents.policy_controller import PolicyController
from lvgrid_rl.baselines.methods import DoNothing, FixedCap, PUDroop, asset_bus_positions
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import make_env
from lvgrid_rl.env.lv_grid_env import EnvConfig
from lvgrid_rl.eval.runner import run_controller
from lvgrid_rl.experiment.reproducibility import RunManifest, SeedSet


def build_parser() -> argparse.ArgumentParser:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="moderate_growth")
    parser.add_argument("--run-dir", type=Path, default=Path("results/m3"))
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-days", type=int, default=2)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="rollout length per worker; PPO collects this many before each "
        "update, so a total step budget below workers * n_steps is misleading",
    )
    return parser


def main() -> None:
    """Train, evaluate against the tuned baselines, and write the run directory."""
    args = build_parser().parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    config = EnvConfig()
    timebase = TimeBase(config.sim_dt_min, config.control_dt_min)
    seeds = SeedSet.from_base(args.seed)

    train_spec = EpisodeSpec(mode=EpisodeMode.TRAIN, length_days=2)

    def make_worker(index: int):
        def factory():
            # Worker seeds are derived from the training seed and the worker
            # index, so parallel rollouts differ but stay reproducible.
            return make_env(
                code=args.code,
                scenario=args.scenario,
                set_name="train",
                config=config,
                episode_spec=train_spec,
                seed=int(seeds.worker_generator("train", index).integers(2**31)),
            )

        return factory

    vec_cls = SubprocVecEnv if args.workers > 1 else DummyVecEnv
    vec_env = vec_cls([make_worker(i) for i in range(args.workers)])

    manifest = RunManifest.create(
        config={
            "algo": args.algo,
            "code": args.code,
            "scenario": args.scenario,
            "steps": args.steps,
            "workers": args.workers,
            "sim_dt_min": config.sim_dt_min,
            "control_dt_min": config.control_dt_min,
        },
        data_manifest_hash="simbench-builtin",
        base_seed=args.seed,
        notes="M3 training run, PV curtailment only",
    )
    run_dir = args.run_dir / manifest.run_id
    manifest.write(run_dir)

    hyperparams: dict[str, int] = {}
    if args.n_steps is not None:
        hyperparams["n_steps"] = args.n_steps
    spec = AgentSpec(algo=args.algo, hyperparams=hyperparams)
    model = make_agent(
        spec, vec_env, timebase, seed=seeds.train, tensorboard_log=str(run_dir / "tb")
    )
    print(f"run_id   {manifest.run_id}")
    print(f"gamma    {model.gamma:.5f} (derived from control_dt)")
    print(f"workers  {args.workers}, steps {args.steps}")

    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    elapsed = time.perf_counter() - started
    model.save(run_dir / "checkpoints" / "final")
    vec_env.close()
    print(f"trained  {args.steps} steps in {elapsed / 60:.1f} min")

    # Evaluate the policy and the baselines on exactly the same episodes.
    eval_spec = EpisodeSpec(
        mode=EpisodeMode.TRAIN, length_days=args.eval_days, randomise_budget=False
    )
    positions = asset_bus_positions(
        make_env(code=args.code, scenario=args.scenario, set_name="val")
    )

    rows = []
    controllers = {
        "policy": lambda e: PolicyController(model),
        "do_nothing": lambda e: DoNothing(e.mapper),
        "fixed_cap_0.5": lambda e: FixedCap(e.mapper, cap=0.5),
        "p_u_droop": lambda e: PUDroop(e.mapper, positions, 1.04, 1.10),
    }
    for name, build in controllers.items():
        env = make_env(
            code=args.code,
            scenario=args.scenario,
            set_name="val",
            config=config,
            episode_spec=eval_spec,
            seed=seeds.eval,
        )
        result = run_controller(
            env, build(env), name, "val", n_episodes=args.eval_episodes, seed=seeds.eval
        )
        rows.append(result.summary())

    header = (
        f"{'controller':16s} {'reward':>10s} {'curtailed MWh':>14s} "
        f"{'k95':>6s} {'overload':>9s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['controller']:16s} {row['reward']:10.2f} {row['curtailed_mwh']:14.4f} "
            f"{row['k95_windows']:6.0f} {row['overload_cost']:9.3f}"
        )

    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nwritten: {run_dir}")


if __name__ == "__main__":
    main()
