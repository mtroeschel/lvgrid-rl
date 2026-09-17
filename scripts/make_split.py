#!/usr/bin/env python3
"""Build the evaluation split for a scenario and commit it to the configuration.

    uv run python scripts/make_split.py --code 1-LV-rural1--2-sw

Writes ``configs/split/<code>.json`` and prints the coverage report. The split is
committed on purpose: from M3 onwards every milestone reports on the same weeks,
which is what makes the numbers comparable across the project. Regenerating it
changes every reported KPI, so the file should only be rewritten deliberately.

See section 6.5 of the architecture document and decision D13.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lvgrid_rl.env.splits import SplitSpec, build_split, weekly_features


def main() -> None:
    """Build, report and write the split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--out-dir", type=Path, default=Path("configs/split"))
    parser.add_argument("--seed", type=int, default=SplitSpec().seed)
    parser.add_argument("--embargo-weeks", type=int, default=SplitSpec().embargo_weeks)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    from lvgrid_rl.data.sources.simbench import load_simbench

    data = load_simbench(args.code)
    features = weekly_features(data)
    spec = SplitSpec(seed=args.seed, embargo_weeks=args.embargo_weeks)
    split = build_split(features, spec)

    pd.set_option("display.width", 200)
    print(f"grid       {args.code}")
    print(f"weeks      {len(features)} complete calendar weeks")
    print(f"axes       {spec.axes}, {spec.n_bins} bins -> {spec.n_bins**2} strata")
    print(f"embargo    {spec.embargo_weeks} week(s)\n")
    print(split.coverage_report(features).round(3).to_string())

    occupied = {v for v in split.strata.values() if v >= 0}
    covered = split.strata_covered("test")
    print(f"\ntest covers {len(covered)} of {len(occupied)} occupied strata")
    if covered != occupied:
        print("  WARNING: the test set misses a stratum")

    if args.print_only:
        return
    path = split.write(args.out_dir / f"{args.code}.json")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
