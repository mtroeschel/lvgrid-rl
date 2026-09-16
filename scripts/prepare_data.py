#!/usr/bin/env python3
"""Bereitet die Zeitreihen eines SimBench-Netzes auf und legt sie im Cache ab.

    uv run python scripts/prepare_data.py --code 1-LV-rural1--2-sw --sim-dt 5

Erzeugt eine Parquet-Datei mit den auf die Simulationsschrittweite gebrachten
Profilen sowie ein Manifest mit dem Inhalts-Hash. Ein erneuter Aufruf mit
identischen Parametern trifft den Cache und rechnet nicht neu.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lvgrid_rl.data.cache import CacheKey, ProfileCache
from lvgrid_rl.data.resample import resample_frame
from lvgrid_rl.data.sources.simbench import load_simbench
from lvgrid_rl.data.timebase import TimeBase


def main() -> None:
    """Liest ein SimBench-Netz, resampelt die Profile und schreibt den Cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--sim-dt", type=int, default=5, help="Minuten, aus {1,2,5,10}")
    parser.add_argument("--control-dt", type=int, default=15)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--force", action="store_true", help="Cache ignorieren")
    args = parser.parse_args()

    import simbench

    timebase = TimeBase(sim_dt_min=args.sim_dt, control_dt_min=args.control_dt)
    key = CacheKey(
        source="simbench",
        source_version=simbench.__version__,
        dataset=args.code,
        sim_dt_min=args.sim_dt,
    )
    cache = ProfileCache(args.cache_dir)

    if cache.has(key) and not args.force:
        profiles, manifest = cache.load(key)
        print(f"Cache-Treffer: {cache.path_for(key).name}")
    else:
        data = load_simbench(args.code)
        profiles = resample_frame(data.profiles, timebase, policies={})
        manifest = cache.store(
            key,
            profiles,
            notes=(
                "Quellauflösung 15 min, stückweise konstant hochgerechnet. "
                "Zeitachse aus deutscher Ortszeit mit Sommerzeit nach UTC "
                "konvertiert (ambiguous=infer)."
            ),
        )
        print(f"Geschrieben: {cache.path_for(key).name}")
        cats: dict[str, int] = {}
        for a in data.assets:
            cats[a.category.value] = cats.get(a.category.value, 0) + 1
        print(f"  Anlagen: {cats}")
        print(f"  Anschlusspunkte: {len(data.connection_point_buses)} Busse")

    print(f"  Zeitschritte: {manifest.n_rows} x {manifest.n_columns} Profile")
    print(f"  Zeitraum UTC: {manifest.start_utc} -> {manifest.end_utc}")
    print(f"  Inhalts-Hash: {manifest.short_hash}")
    print(f"  gamma (aus control_dt): {timebase.suggested_gamma():.5f}")


if __name__ == "__main__":
    main()
