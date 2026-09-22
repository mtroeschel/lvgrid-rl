#!/usr/bin/env python3
"""Pareto comparison: curtailed energy against residual overload.

    uv run python scripts/pareto.py --set test --policies results/m3/*/eval/test.json

The acceptance criterion asks whether the policy beats the reference methods on
the EN 50160 pass rate and the overload integral. It does not ask the question a
reviewer will: **does the policy find a better trade-off, or does it simply
curtail more?**

On the M3 test set the answer was not decidable from the existing numbers. The
tuned cap curtails 3.1 MWh and leaves 313 units of overload; the policies curtail
16 to 38 MWh and leave 0 to 108. No cap in the tuned grid operates at the
policies' curtailment level, so there was nothing to compare against. This script
sweeps caps across that range and plots both on the same axes.

Reading the plot: both axes are costs, so **down and to the left is better**. A
policy point that lies below the cap curve buys its compliance more cheaply than
a rigid cap; a point on the curve means the policy has learned an expensive
equivalent of a cap.

Policy points are read from existing ``eval/<set>.json`` files rather than
recomputed, because a full test set costs about twenty minutes per controller.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from lvgrid_rl.baselines.methods import DoNothing, FixedCap, PUDroop, asset_bus_positions
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import make_env
from lvgrid_rl.env.lv_grid_env import EnvConfig
from lvgrid_rl.eval.runner import run_controller

DEFAULT_CAPS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60)
"""Chosen to bracket the policies' curtailment level, which the tuning grid did
not reach: its lowest cap of 0.3 still curtails far less than any policy."""


@dataclass(frozen=True, slots=True)
class Point:
    """One controller on the curtailment/overload plane."""

    label: str
    kind: str
    curtailed_mwh: float
    overload_cost: float
    pass_rate: float

    def dominates(self, other: Point) -> bool:
        """Is this point at least as good in both costs, and better in one?

        Both axes are costs, so domination means lower or equal on both and
        strictly lower on at least one.
        """
        not_worse = (
            self.curtailed_mwh <= other.curtailed_mwh
            and self.overload_cost <= other.overload_cost
        )
        better = (
            self.curtailed_mwh < other.curtailed_mwh
            or self.overload_cost < other.overload_cost
        )
        return not_worse and better


def pareto_front(points: list[Point]) -> list[Point]:
    """Non-dominated points, sorted by curtailment."""
    front = [p for p in points if not any(q.dominates(p) for q in points)]
    return sorted(front, key=lambda p: p.curtailed_mwh)


def load_policy_points(paths: list[Path], set_name: str) -> list[Point]:
    """Read policy results from existing evaluation files.

    Raises:
        ValueError: if a file was produced for a different set. Mixing sets on
            one plane would be meaningless, and the mistake is easy to make with
            a glob.
    """
    points: list[Point] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("set") != set_name:
            raise ValueError(
                f"{path} holds results for set {payload.get('set')!r}, not {set_name!r}"
            )
        for row in payload["results"]:
            if row["controller"] != "policy":
                continue
            points.append(
                Point(
                    label=f"policy {path.parent.parent.name[:8]}",
                    kind="policy",
                    curtailed_mwh=row["curtailed_mwh"],
                    overload_cost=row["overload_cost"],
                    pass_rate=row["pass_rate"],
                )
            )
    return points


def main() -> None:
    """Sweep caps, collect policy points and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="test")
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--scenario", default="moderate_growth")
    parser.add_argument("--caps", type=float, nargs="*", default=list(DEFAULT_CAPS))
    parser.add_argument(
        "--policies",
        type=Path,
        nargs="*",
        default=[],
        help="existing eval/<set>.json files to take policy points from",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=None,
        help="limit to the first N weeks of the set. Weeks stay complete, so "
        "the pass rate remains standard-conforming; only the sample shrinks.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/pareto"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)

    def build():
        return make_env(
            code=args.code,
            scenario=args.scenario,
            set_name=args.set_name,
            config=EnvConfig(),
            episode_spec=spec,
            seed=0,
        )

    probe = build()
    n_weeks = args.max_weeks or probe.sampler.n_weeks
    positions = asset_bus_positions(probe)

    points: list[Point] = []

    env = build()
    summary = run_controller(
        env, DoNothing(env.mapper), "do_nothing", args.set_name, n_episodes=n_weeks
    ).summary()
    points.append(
        Point(
            "do_nothing",
            "reference",
            summary["curtailed_mwh"],
            summary["overload_cost"],
            summary["pass_rate"],
        )
    )

    for cap in args.caps:
        env = build()
        summary = run_controller(
            env,
            FixedCap(env.mapper, cap=cap),
            f"fixed_cap({cap})",
            args.set_name,
            n_episodes=n_weeks,
        ).summary()
        points.append(
            Point(
                f"cap {cap:.2f}",
                "fixed_cap",
                summary["curtailed_mwh"],
                summary["overload_cost"],
                summary["pass_rate"],
            )
        )

    tuned_path = Path("configs/baseline/tuned.json")
    if tuned_path.exists():
        droop = json.loads(tuned_path.read_text(encoding="utf-8"))["results"][
            "p_u_droop"
        ]["params"]
        env = build()
        summary = run_controller(
            env,
            PUDroop(env.mapper, positions, droop["v_start"], droop["v_max"]),
            "p_u_droop",
            args.set_name,
            n_episodes=n_weeks,
        ).summary()
        points.append(
            Point(
                f"droop {droop['v_start']}/{droop['v_max']}",
                "droop",
                summary["curtailed_mwh"],
                summary["overload_cost"],
                summary["pass_rate"],
            )
        )

    points.extend(load_policy_points(list(args.policies), args.set_name))

    front = pareto_front(points)
    front_labels = {p.label for p in front}

    header = (
        f"{'controller':26s} {'curtailed MWh':>14s} {'overload':>10s} "
        f"{'pass rate':>10s}  front"
    )
    print(f"set {args.set_name}, {n_weeks} complete weeks\n")
    print(header)
    print("-" * len(header))
    for p in sorted(points, key=lambda p: p.curtailed_mwh):
        mark = "  *" if p.label in front_labels else ""
        print(
            f"{p.label:26s} {p.curtailed_mwh:14.3f} {p.overload_cost:10.3f} "
            f"{p.pass_rate:10.4f}{mark}"
        )
    print("\n* = non-dominated. Both axes are costs, so lower is better on both.")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": args.set_name,
        "weeks": n_weeks,
        "scenario": args.scenario,
        "points": [
            {
                "label": p.label,
                "kind": p.kind,
                "curtailed_mwh": p.curtailed_mwh,
                "overload_cost": p.overload_cost,
                "pass_rate": p.pass_rate,
                "on_front": p.label in front_labels,
            }
            for p in points
        ],
    }
    (args.out / f"pareto_{args.set_name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not args.no_plot:
        try:
            _plot(points, front, args.out / f"pareto_{args.set_name}.png", args.set_name)
        except ModuleNotFoundError:
            print("\nmatplotlib missing; install the 'viz' extra for the plot.")
    print(f"\nwritten: {args.out}")


def _plot(points: list[Point], front: list[Point], path: Path, set_name: str) -> None:
    """Draw the curtailment/overload plane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fixed_cap": ("o", "tab:blue", "fixed cap"),
        "policy": ("s", "tab:red", "policy"),
        "droop": ("^", "tab:green", "P(U) droop"),
        "reference": ("x", "tab:gray", "do nothing"),
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [p.curtailed_mwh for p in front],
        [p.overload_cost for p in front],
        color="0.7",
        linewidth=1,
        zorder=1,
        label="Pareto front",
    )
    seen: set[str] = set()
    for p in points:
        marker, colour, label = styles.get(p.kind, ("d", "black", p.kind))
        ax.scatter(
            p.curtailed_mwh,
            p.overload_cost,
            marker=marker,
            color=colour,
            zorder=2,
            label=label if label not in seen else None,
        )
        seen.add(label)
        ax.annotate(
            p.label,
            (p.curtailed_mwh, p.overload_cost),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    ax.set_xlabel("curtailed energy [MWh]")
    ax.set_ylabel("residual overload [percent-hours above 100 %]")
    ax.set_title(f"Cost of compliance, {set_name} set")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
