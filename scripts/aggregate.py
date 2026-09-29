"""Aggregate results across seeds: interquartile mean with bootstrap intervals.

    uv run python scripts/aggregate.py --results results/m3/*/eval/test.json
        --pareto results/pareto/pareto_test.json

**Why not mean and standard deviation.** Distributions across RL seeds are
regularly skewed and sometimes multimodal -- a single unlucky run can move a mean
by more than the effect being measured. The interquartile mean trims the outer
quartiles before averaging and is therefore far less sensitive to one outlier,
while still using more of the data than a median. Confidence intervals come from
a percentile bootstrap rather than a normal approximation, which would assume the
very symmetry that is absent (architecture section 9.3, following the practice
of `rliable`).

**What the derived metric is.** The headline question is not how much overload a
policy leaves, but whether it leaves less than a rigid cap that curtails the same
amount of energy. This script therefore interpolates the cap curve from a Pareto
run and reports, per seed, the overload at that seed's curtailment level -- then
aggregates the advantage over seeds. That is the number the acceptance record
should carry.

**On few seeds.** With three or four seeds a bootstrap interval is wide and the
interquartile mean is close to a median. The script says so rather than printing
a narrow-looking figure that invites over-reading.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "iqm",
    "bootstrap_ci",
    "Aggregate",
    "aggregate_metric",
    "CapReference",
    "cap_reference",
    "cap_curve_overload",
    "uncontrolled_overload",
    "SMALL_REFERENCE_FRACTION",
    "WIDE_GAP_FRACTION",
]

SMALL_REFERENCE_FRACTION = 0.05
"""Below this share of the uncontrolled overload, a percentage is not reported.

A relative advantage over a reference that is already near zero says nothing
about the size of the improvement -- in M3 it turned a two-unit difference into
"96 % better", against an uncontrolled value of 366.
"""

WIDE_GAP_FRACTION = 0.10
"""Bracket width, as a share of the swept range, above which the curve is
considered poorly resolved at that point."""

MIN_SEEDS_FOR_CI = 5
"""Below this the interval is reported but flagged as indicative.

Architecture section 9.3 asks for at least five seeds, better ten.
"""


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50 % of the sample.

    With fewer than four values the trimming would remove everything, so the
    plain mean is returned -- which is honest, because with three seeds there is
    no robustness to be had.
    """
    array = np.sort(np.asarray(values, dtype=float))
    if array.size < 4:
        return float(array.mean())
    lower = int(np.floor(array.size * 0.25))
    upper = int(np.ceil(array.size * 0.75))
    return float(array[lower:upper].mean())


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the interquartile mean.

    Resampling is seeded so the reported interval is reproducible; an interval
    that moves between report runs cannot be checked by a reviewer.
    """
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, array.size), replace=True)
    stats = np.array([iqm(row) for row in draws])
    half = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(stats, half)),
        float(np.quantile(stats, 1.0 - half)),
    )


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Aggregated value of one metric over seeds."""

    metric: str
    n_seeds: int
    values: tuple[float, ...]
    iqm: float
    ci_low: float
    ci_high: float

    @property
    def indicative_only(self) -> bool:
        """Too few seeds for the interval to carry weight."""
        return self.n_seeds < MIN_SEEDS_FOR_CI

    def format_row(self) -> str:
        """One line for the results table."""
        flag = "  (indicative, n < 5)" if self.indicative_only else ""
        return (
            f"{self.metric:22s} {self.iqm:10.4f}  "
            f"[{self.ci_low:9.4f}, {self.ci_high:9.4f}]  n={self.n_seeds}{flag}"
        )


def aggregate_metric(metric: str, values: Sequence[float], seed: int = 0) -> Aggregate:
    """Aggregate one metric over seeds."""
    low, high = bootstrap_ci(values, seed=seed)
    return Aggregate(
        metric=metric,
        n_seeds=len(values),
        values=tuple(float(v) for v in values),
        iqm=iqm(values),
        ci_low=low,
        ci_high=high,
    )


@dataclass(frozen=True, slots=True)
class CapReference:
    """What a fixed cap achieves at a given curtailment, and how well it is known.

    The interpolated value alone is not enough to judge a comparison. Two things
    can make it misleading, and both occurred in the M3 results:

    * the reference can be so small that a percentage against it inflates a
      difference of two overload units into "96 % better";
    * the policy can sit in a **gap** of the cap sweep, where the interpolation
      is a straight line through unmeasured ground. In M3 all five policies fell
      into a 12.6 MWh gap between the caps at 0.15 and 0.10.

    The fields therefore carry the bracket, so the caller can say how much the
    number is worth.
    """

    overload: float
    lower_mwh: float
    upper_mwh: float

    @property
    def gap_mwh(self) -> float:
        """Distance between the two measured caps bracketing the query point."""
        return self.upper_mwh - self.lower_mwh


def cap_reference(pareto_payload: dict, curtailed_mwh: float) -> CapReference | None:
    """Interpolate the cap curve and report the bracket it came from.

    Linear interpolation between the measured cap points. Returns ``None``
    outside the measured range rather than extrapolating: beyond the sweep the
    curve is unknown, and an extrapolated comparison would look like a
    measurement.
    """
    caps = sorted(
        (p["curtailed_mwh"], p["overload_cost"])
        for p in pareto_payload["points"]
        if p["kind"] == "fixed_cap"
    )
    if not caps:
        return None
    xs = [c[0] for c in caps]
    ys = [c[1] for c in caps]
    if curtailed_mwh < xs[0] or curtailed_mwh > xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= curtailed_mwh <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span == 0:
                return CapReference(ys[i], xs[i], xs[i + 1])
            f = (curtailed_mwh - xs[i]) / span
            return CapReference(ys[i] + f * (ys[i + 1] - ys[i]), xs[i], xs[i + 1])
    return None


def cap_curve_overload(pareto_payload: dict, curtailed_mwh: float) -> float | None:
    """Overload a fixed cap leaves when curtailing the given amount."""
    reference = cap_reference(pareto_payload, curtailed_mwh)
    return None if reference is None else reference.overload


def uncontrolled_overload(pareto_payload: dict) -> float:
    """Overload of the uncontrolled grid, as the scale for relative statements.

    Without a scale a difference of two overload units reads as either "96 %
    better" or "half a percent of the problem", and only the second is
    informative.
    """
    for point in pareto_payload["points"]:
        if point["kind"] == "reference":
            return float(point["overload_cost"])
    caps = [p for p in pareto_payload["points"] if p["kind"] == "fixed_cap"]
    return max((float(p["overload_cost"]) for p in caps), default=float("nan"))


def load_policy_rows(paths: Sequence[Path], set_name: str) -> list[dict]:
    """Collect the policy row from each result file."""
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("set") != set_name:
            raise ValueError(
                f"{path} holds results for set {payload.get('set')!r}, not {set_name!r}"
            )
        for row in payload["results"]:
            if row["controller"] == "policy":
                rows.append({**row, "run": path.parent.parent.name})
    return rows


def main() -> None:
    """Aggregate the seeds and print the acceptance table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--set", dest="set_name", default="test")
    parser.add_argument("--pareto", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/aggregate"))
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args()

    rows = load_policy_rows(args.results, args.set_name)
    if not rows:
        raise SystemExit("No policy rows found in the given result files.")

    print(f"set {args.set_name}, {len(rows)} seeds\n")
    print("per seed:")
    print(
        f"  {'run':14s} {'pass rate':>10s} {'k95':>6s} {'overload':>10s} "
        f"{'curtailed MWh':>14s}"
    )
    for row in rows:
        print(
            f"  {row['run'][:12]:14s} {row['pass_rate']:10.4f} "
            f"{row['k95_windows']:6.0f} {row['overload_cost']:10.3f} "
            f"{row['curtailed_mwh']:14.3f}"
        )

    metrics = ("pass_rate", "k95_windows", "overload_cost", "curtailed_mwh")
    aggregates = [
        aggregate_metric(m, [r[m] for r in rows], seed=args.seed) for m in metrics
    ]

    print(f"\n{'metric':22s} {'IQM':>10s}  {'95 % bootstrap CI':>24s}")
    print("-" * 66)
    for a in aggregates:
        print(a.format_row())

    advantage_payload = None
    if args.pareto is not None:
        pareto_payload = json.loads(args.pareto.read_text(encoding="utf-8"))
        scale = uncontrolled_overload(pareto_payload)
        caps = [
            p["curtailed_mwh"]
            for p in pareto_payload["points"]
            if p["kind"] == "fixed_cap"
        ]
        swept = (max(caps) - min(caps)) if caps else float("nan")

        per_seed = []
        for row in rows:
            reference = cap_reference(pareto_payload, row["curtailed_mwh"])
            if reference is None:
                continue
            difference = reference.overload - row["overload_cost"]
            small = reference.overload < SMALL_REFERENCE_FRACTION * scale
            wide = reference.gap_mwh > WIDE_GAP_FRACTION * swept
            per_seed.append(
                {
                    "run": row["run"],
                    "curtailed_mwh": row["curtailed_mwh"],
                    "policy_overload": row["overload_cost"],
                    "cap_overload": reference.overload,
                    "difference": difference,
                    "difference_of_uncontrolled_percent": 100.0 * difference / scale,
                    "advantage_percent": (
                        100.0 * difference / reference.overload
                        if reference.overload > 0
                        else float("nan")
                    ),
                    "reference_is_small": small,
                    "bracket_mwh": [reference.lower_mwh, reference.upper_mwh],
                    "bracket_gap_mwh": reference.gap_mwh,
                    "poorly_resolved": wide,
                }
            )

        if per_seed:
            print("\nagainst a fixed cap curtailing the same energy:")
            print(
                f"  {'run':12s} {'curtailed':>10s} {'policy':>9s} {'cap':>9s} "
                f"{'diff':>8s} {'of uncontr.':>12s}  notes"
            )
            for p in per_seed:
                notes = []
                if p["reference_is_small"]:
                    notes.append("reference near zero")
                if p["poorly_resolved"]:
                    notes.append(f"gap {p['bracket_gap_mwh']:.1f} MWh")
                print(
                    f"  {p['run'][:10]:12s} {p['curtailed_mwh']:10.2f} "
                    f"{p['policy_overload']:9.3f} {p['cap_overload']:9.3f} "
                    f"{p['difference']:8.3f} "
                    f"{p['difference_of_uncontrolled_percent']:11.2f} %  "
                    + ("; ".join(notes) if notes else "")
                )

            # The absolute difference is aggregated unconditionally; the
            # percentage only where the reference is large enough to carry one.
            absolute = aggregate_metric(
                "overload_difference", [p["difference"] for p in per_seed], seed=args.seed
            )
            print("\n" + absolute.format_row())
            relative_values = [
                p["advantage_percent"]
                for p in per_seed
                if not p["reference_is_small"] and np.isfinite(p["advantage_percent"])
            ]
            relative = None
            if relative_values:
                relative = aggregate_metric(
                    "overload_advantage_%", relative_values, seed=args.seed
                )
                print(relative.format_row())
            else:
                print(
                    "  overload_advantage_%   not reported: every reference is "
                    f"below {SMALL_REFERENCE_FRACTION:.0%} of the uncontrolled "
                    f"overload ({scale:.1f}), so a percentage would inflate a "
                    "difference of a few units into a near-total advantage."
                )

            if any(p["poorly_resolved"] for p in per_seed):
                print(
                    "\n  WARNING: policies fall into a gap of the cap sweep, so "
                    "the reference is a straight line through unmeasured ground. "
                    "Refine the sweep around the policies before reporting."
                )

            advantage_payload = {
                "uncontrolled_overload": scale,
                "per_seed": per_seed,
                "absolute": {
                    "iqm": absolute.iqm,
                    "ci_low": absolute.ci_low,
                    "ci_high": absolute.ci_high,
                    "n_seeds": absolute.n_seeds,
                },
                "relative": (
                    None
                    if relative is None
                    else {
                        "iqm": relative.iqm,
                        "ci_low": relative.ci_low,
                        "ci_high": relative.ci_high,
                        "n_seeds": relative.n_seeds,
                    }
                ),
            }

    if any(a.indicative_only for a in aggregates):
        print(
            f"\nNote: fewer than {MIN_SEEDS_FOR_CI} seeds. The interval is "
            "reported but should not be read as a confidence statement "
            "(architecture section 9.3)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "set": args.set_name,
        "n_seeds": len(rows),
        "per_seed": rows,
        "aggregates": [
            {
                "metric": a.metric,
                "iqm": a.iqm,
                "ci_low": a.ci_low,
                "ci_high": a.ci_high,
                "values": list(a.values),
                "indicative_only": a.indicative_only,
            }
            for a in aggregates
        ],
        "overload_advantage": advantage_payload,
    }
    path = args.out / f"aggregate_{args.set_name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
