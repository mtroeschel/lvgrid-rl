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
from dataclasses import asdict, replace
from pathlib import Path

from lvgrid_rl.agents.factory import CUSTOM_POLICIES, AgentSpec, make_agent
from lvgrid_rl.agents.policy_controller import PolicyController
from lvgrid_rl.baselines.flexibility import flex_baselines
from lvgrid_rl.baselines.methods import DoNothing, FixedCap, PUDroop, asset_bus_positions
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import (
    DEFAULT_EV,
    DEFAULT_HEAT_PUMPS,
    DEFAULT_STORAGE,
    EvSizing,
    HeatPumpSizing,
    StorageSizing,
    make_env,
)
from lvgrid_rl.env.lv_grid_env import EnvConfig
from lvgrid_rl.env.obs import ObservationLayoutMode, ObservationSpec
from lvgrid_rl.env.reward import RewardConfig, RewardMode
from lvgrid_rl.eval.kpi_schema import KPI_SCHEMA
from lvgrid_rl.eval.runner import run_controller
from lvgrid_rl.experiment.callbacks import (
    DEFAULT_DUALS,
    LagrangianCallback,
    TrainingProgress,
)
from lvgrid_rl.experiment.reproducibility import RunManifest, SeedSet


def build_parser() -> argparse.ArgumentParser:
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("configs/agent/ppo.yaml"),
        help="agent specification: algorithm, hyperparameters, policy settings",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="moderate_growth")
    parser.add_argument("--run-dir", type=Path, default=Path("results/m3"))
    parser.add_argument(
        "--eval-set",
        default="val",
        choices=("val", "test"),
        help="set for the report at the end of training. The acceptance figure "
        "belongs on the test weeks and is produced by scripts/evaluate.py, so "
        "that a policy need not be retrained to be assessed on another set.",
    )
    parser.add_argument(
        "--tuned",
        type=Path,
        default=Path("configs/baseline/tuned.json"),
        help="tuned baseline parameters from scripts/tune_baselines.py",
    )
    parser.add_argument(
        "--allow-untuned",
        action="store_true",
        help="compare against default baseline parameters. Only for smoke runs: "
        "the results are not admissible as evidence, and the controllers are "
        "labelled accordingly in the table.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress the progress display. It writes to stderr, so a "
        "redirected stdout keeps only the results table either way.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="seconds between progress updates",
    )
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="save the checkpoint and stop, without the report on --eval-set. "
        "The report recomputes every baseline, which is deterministic and "
        "identical across runs, and costs over an hour per run; the acceptance "
        "figure comes from scripts/evaluate.py on the test weeks either way.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="intra-op threads for PyTorch. One is faster for networks this "
        "small: between environment steps a larger pool falls asleep and has to "
        "be woken for every one of the many small operations of the shared "
        "policy (7.6 ms per decision at seven threads, 1.0 ms at one), and a "
        "minibatch update is no slower. Recorded, because the thread count can "
        "change the order of floating-point reductions.",
    )
    parser.add_argument(
        "--storage",
        action="store_true",
        help="add a controllable battery at every PV bus, 1 kW and 2 kWh per "
        "kWp (the M4 configuration); without it the actuators are PV only, as "
        "in M3",
    )
    parser.add_argument(
        "--heat-pumps",
        action="store_true",
        help="make the grid's heat pumps controllable, with buffer stores and "
        "when2heat COP (decision D15); without it they follow their SimBench "
        "profiles as uncontrolled load",
    )
    parser.add_argument(
        "--ev",
        action="store_true",
        help="make the grid's charge points controllable, with emobpy sessions "
        "scaled to the SimBench energy (decision D17); without it they follow "
        "their SimBench profiles as uncontrolled load",
    )
    parser.add_argument(
        "--reward-mode",
        default=RewardMode.FIXED_WEIGHTS.value,
        choices=[m.value for m in RewardMode],
        help="fixed_weights scalarises the constraint terms with their weights; "
        "lagrangian rewards only the objective terms and prices the constraints "
        "with multipliers updated by dual ascent (architecture section 6.4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="rollout length per worker; PPO collects this many before each "
        "update, so a total step budget below workers * n_steps is misleading",
    )
    return parser


def load_baseline_params(path: Path, allow_untuned: bool) -> tuple[dict, bool]:
    """Load the tuned baseline parameters.

    Falling back to defaults when the file is missing would be the worst of both
    worlds: the run would look like a comparison against tuned references and
    would not be one. The deadbands and slopes of the droop methods are designed
    for a hard band, and under the percentile criterion the optimum is different
    -- comparing against the untuned settings is how an agent wins against a
    deliberately weak reference (architecture section 6.6, consequence 5).

    Raises:
        FileNotFoundError: if the file is missing and ``allow_untuned`` is not
            set.
    """
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["results"], True
    if not allow_untuned:
        raise FileNotFoundError(
            f"No tuned baseline parameters at {path}. Run\n"
            "    uv run python scripts/tune_baselines.py\n"
            "first, or pass --allow-untuned for a smoke run whose baseline "
            "comparison is not admissible as evidence."
        )
    return (
        {
            "fixed_cap": {"params": {"cap": 0.5}},
            "p_u_droop": {"params": {"v_start": 1.04, "v_max": 1.10}},
        },
        False,
    )


def main() -> None:
    """Train, evaluate against the tuned baselines, and write the run directory."""
    args = build_parser().parse_args()

    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )

    # Loaded before training: a missing tuning run must fail in seconds, not
    # after four hours of compute.
    tuned, is_tuned = load_baseline_params(args.tuned, args.allow_untuned)
    if not is_tuned:
        print("WARNING: comparing against untuned baselines; not admissible.\n")

    import torch

    torch.set_num_threads(args.torch_threads)
    reward_mode = RewardMode(args.reward_mode)
    storage: StorageSizing | None = DEFAULT_STORAGE if args.storage else None
    heat_pumps: HeatPumpSizing | None = DEFAULT_HEAT_PUMPS if args.heat_pumps else None
    ev: EvSizing | None = DEFAULT_EV if args.ev else None
    spec = AgentSpec.from_yaml(args.agent_config)
    if args.n_steps is not None:
        spec = replace(spec, hyperparams={**spec.hyperparams, "n_steps": args.n_steps})
    # The observation layout follows from the policy: action mode 2 needs the
    # per-asset blocks, every other policy the flat layout of M3.
    obs_layout = (
        ObservationLayoutMode.PER_ASSET
        if spec.policy in CUSTOM_POLICIES
        else ObservationLayoutMode.FLAT
    )
    config = EnvConfig(
        reward=RewardConfig(mode=reward_mode),
        observation=ObservationSpec(layout=obs_layout),
    )
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
                storage=storage,
                heat_pumps=heat_pumps,
                ev=ev,
            )

        return factory

    manifest = RunManifest.create(
        config={
            # The full agent specification, not just the algorithm name: in M3
            # the hash covered only "ppo", and the hyperparameters actually
            # used were nowhere on record.
            "agent": spec.as_config(),
            "code": args.code,
            "scenario": args.scenario,
            "steps": args.steps,
            "workers": args.workers,
            "sim_dt_min": config.sim_dt_min,
            "control_dt_min": config.control_dt_min,
            "eval_set": args.eval_set,
            "reward_mode": reward_mode.value,
            "storage": asdict(storage) if storage is not None else None,
            "heat_pumps": asdict(heat_pumps) if heat_pumps is not None else None,
            "ev": asdict(ev) if ev is not None else None,
            "obs_layout": obs_layout.value,
            "torch_threads": args.torch_threads,
            # Part of the hashed configuration: the dual settings decide the
            # trajectory of a Lagrangian run as much as weights decide a
            # fixed-weight one.
            **(
                {"duals": {name: asdict(d) for name, d in DEFAULT_DUALS.items()}}
                if reward_mode is RewardMode.LAGRANGIAN
                else {}
            ),
        },
        data_manifest_hash="simbench-builtin",
        base_seed=args.seed,
        notes=(
            "M3 training run, PV curtailment only"
            if reward_mode is RewardMode.FIXED_WEIGHTS
            else "Lagrangian validation on the M3 setup, PV curtailment only"
        ),
    )
    run_dir = args.run_dir / manifest.run_id
    manifest.write(run_dir)
    # What scripts/evaluate.py needs to rebuild the same environment. The
    # manifest keeps only a hash of the configuration.
    (run_dir / "env.json").write_text(
        json.dumps(
            {
                "storage": asdict(storage) if storage is not None else None,
                "heat_pumps": asdict(heat_pumps) if heat_pumps is not None else None,
                "ev": asdict(ev) if ev is not None else None,
                "obs_layout": obs_layout.value,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    vec_cls = SubprocVecEnv if args.workers > 1 else DummyVecEnv
    # VecMonitor is not optional: without it Stable-Baselines3 logs no episode
    # return at all, so `rollout/ep_rew_mean` is missing from TensorBoard and
    # there is no way to tell whether training converged or was cut short.
    vec_env = VecMonitor(
        vec_cls([make_worker(i) for i in range(args.workers)]),
        filename=str(run_dir / "monitor"),
    )

    model = make_agent(
        spec, vec_env, timebase, seed=seeds.train, tensorboard_log=str(run_dir / "tb")
    )
    print(f"run_id   {manifest.run_id}")
    print(f"gamma    {model.gamma:.5f} (derived from control_dt)")
    print(f"workers  {args.workers}, steps {args.steps}")

    print(f"reward   {reward_mode.value}")

    callbacks = []
    if not args.no_progress:
        callbacks.append(TrainingProgress(args.steps, interval_s=args.progress_interval))
    lagrangian = None
    if reward_mode is RewardMode.LAGRANGIAN:
        lagrangian = LagrangianCallback(
            config.reward,
            DEFAULT_DUALS,
            history_path=run_dir / "lagrangian.json",
        )
        callbacks.append(lagrangian)
    callback = CallbackList(callbacks) if callbacks else None
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=callback)
    elapsed = time.perf_counter() - started
    model.save(run_dir / "checkpoints" / "final")
    vec_env.close()
    print(f"trained  {args.steps} steps in {elapsed / 60:.1f} min")
    if lagrangian is not None:
        for name, value in lagrangian.multipliers.items():
            bound = lagrangian.duals[name].lambda_max
            flag = "  AT BOUND: degenerated to a fixed weight" if value >= bound else ""
            print(f"lambda   {name:18s} {value:9.3f}{flag}")
    if args.no_final_eval:
        print(f"\nwritten: {run_dir}")
        return

    # Evaluate the policy and the baselines on exactly the same episodes.
    # Complete calendar weeks, fixed order, zero budget. The previous setting
    # sampled episodes with replacement from the set, so every seed evaluated on
    # a different selection of weeks -- which made the baselines differ between
    # runs although they are deterministic, and made cross-seed aggregation
    # meaningless.
    eval_spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
    probe = make_env(
        code=args.code,
        scenario=args.scenario,
        set_name=args.eval_set,
        storage=storage,
        heat_pumps=heat_pumps,
        ev=ev,
    )
    positions = asset_bus_positions(probe)

    cap = tuned["fixed_cap"]["params"]["cap"]
    droop = tuned["p_u_droop"]["params"]
    suffix = "" if is_tuned else " (UNTUNED)"

    rows = []
    controllers = {
        "policy": lambda e: PolicyController(model),
        "do_nothing": lambda e: DoNothing(e.mapper),
        f"fixed_cap({cap}){suffix}": lambda e: FixedCap(e.mapper, cap=cap),
        f"p_u_droop({droop['v_start']}/{droop['v_max']}){suffix}": lambda e: PUDroop(
            e.mapper, positions, droop["v_start"], droop["v_max"]
        ),
    }
    # B5 and B6, when the run has flexible assets and they are tuned.
    if is_tuned:
        controllers.update(flex_baselines(tuned, positions, probe))
    for name, build in controllers.items():
        env = make_env(
            code=args.code,
            scenario=args.scenario,
            set_name=args.eval_set,
            config=config,
            episode_spec=eval_spec,
            seed=seeds.eval,
            storage=storage,
            heat_pumps=heat_pumps,
            ev=ev,
        )
        # n_episodes defaults to one per week in the set, which is the
        # standard-conforming choice.
        result = run_controller(env, build(env), name, args.eval_set, seed=seeds.eval)
        rows.append(result.summary())

    header = (
        f"{'controller':24s} {'pass rate':>10s} {'k95':>6s} "
        f"{'overload':>10s} {'curtailed MWh':>14s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['controller']:24s} {row['pass_rate']:10.4f} "
            f"{row['k95_windows']:6.0f} {row['overload_cost']:10.3f} "
            f"{row['curtailed_mwh']:14.4f}"
        )

    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "summary.json").write_text(
        json.dumps(
            {
                "baselines_tuned": is_tuned,
                "baseline_params": tuned,
                "eval_set": args.eval_set,
                "kpi_schema": KPI_SCHEMA,
                "episode_mode": "evaluate",
                "results": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {run_dir}")


if __name__ == "__main__":
    main()
