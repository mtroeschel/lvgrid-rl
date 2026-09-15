#!/usr/bin/env python3
"""Einstiegspunkt fuer einen Trainingslauf.

In M0 nur ein Rauchtest: Konfiguration aufloesen, Manifest schreiben, Ende.
Das prueft die Reproduzierbarkeitskette (Konfigurationshash, Seeds, Run-Ordner)
bevor irgendein Modell existiert.

    python scripts/train.py --seed 1 --run-dir results/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lvgrid_rl.experiment.reproducibility import RunManifest


def main() -> None:
    """Schreibt ein Run-Manifest und meldet die Run-Kennung."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()

    # Platzhalter fuer die Hydra-aufgeloeste Konfiguration (ab M1).
    config = {"grid": "1-LV-rural1--0-sw", "env": {"sim_dt_min": 5}}

    manifest = RunManifest.create(
        config=config,
        data_manifest_hash="pending",
        base_seed=args.seed,
        notes="M0 Rauchtest, kein Training",
    )
    path = manifest.write(args.run_dir / manifest.run_id)
    print(f"run_id={manifest.run_id}")
    print(f"manifest={path}")


if __name__ == "__main__":
    main()
