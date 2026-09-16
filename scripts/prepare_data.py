#!/usr/bin/env python3
"""Prepare the time series of a SimBench grid and store them in the cache.

    uv run python scripts/prepare_data.py --code 1-LV-rural1--2-sw --sim-dt 5

Writes a Parquet file with the profiles brought onto the simulation step size,
plus a manifest holding the content hash. Active and reactive power are stored
in one frame, with reactive columns prefixed, so the hash covers both and a
reactive series cannot drift out of step with its active counterpart. Calling
the script again with identical parameters hits the cache instead of
recomputing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lvgrid_rl.data.cache import CacheKey, ProfileCache
from lvgrid_rl.data.resample import resample_frame
from lvgrid_rl.data.sources.simbench import load_simbench, split_pq
from lvgrid_rl.data.timebase import TimeBase


def main() -> None:
    """Read a SimBench grid, resample its profiles and write the cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--sim-dt", type=int, default=5, help="minutes, from {1,2,5,10}")
    parser.add_argument("--control-dt", type=int, default=15)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--force", action="store_true", help="ignore the cache")
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
        print(f"cache hit: {cache.path_for(key).name}")
    else:
        data = load_simbench(args.code)
        profiles = resample_frame(data.merged(), timebase, policies={})
        manifest = cache.store(
            key,
            profiles,
            notes=(
                "Source resolution 15 min, upsampled piecewise constant. Time "
                "axis converted from German local time with daylight saving to "
                "UTC. Reactive power carried as a prefixed column group."
            ),
        )
        print(f"written: {cache.path_for(key).name}")
        categories: dict[str, int] = {}
        for asset in data.assets:
            categories[asset.category.value] = categories.get(asset.category.value, 0) + 1
        print(f"  assets: {categories}")
        print(f"  connection points: {len(data.connection_point_buses)} buses")

    active, reactive = split_pq(profiles)
    print(f"  time steps: {manifest.n_rows}")
    print(f"  profiles:   {active.shape[1]} active, {reactive.shape[1]} reactive")
    print(f"  period UTC: {manifest.start_utc} -> {manifest.end_utc}")
    print(f"  content hash: {manifest.short_hash}")
    print(f"  gamma (from control_dt): {timebase.suggested_gamma():.5f}")


if __name__ == "__main__":
    main()
