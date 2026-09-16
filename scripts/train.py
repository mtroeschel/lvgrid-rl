#!/usr/bin/env python3
"""Entry point for a training run.

In M0 this is a smoke test only: resolve the configuration, write a manifest,
done. That verifies the reproducibility chain -- configuration hash, seeds, run
directory -- before any model exists.

    python scripts/train.py --seed 1 --run-dir results/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lvgrid_rl.experiment.reproducibility import RunManifest


def main() -> None:
    """Write a run manifest and report the run identifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()

    # Placeholder for the Hydra-resolved configuration (from M1).
    config = {"grid": "1-LV-rural1--2-sw", "env": {"sim_dt_min": 5}}

    manifest = RunManifest.create(
        config=config,
        data_manifest_hash="pending",
        base_seed=args.seed,
        notes="M0 smoke test, no training",
    )
    path = manifest.write(args.run_dir / manifest.run_id)
    print(f"run_id={manifest.run_id}")
    print(f"manifest={path}")


if __name__ == "__main__":
    main()
