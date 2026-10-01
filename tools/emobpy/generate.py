"""Generate one EV-year per charge point with emobpy (decision D17).

    cd tools/emobpy
    uv sync --locked
    uv run python generate.py                 # all charge points, 53 weeks
    uv run python generate.py --weeks 2 --charge-points-limit 1   # a smoke run

Reads ``configs/ev/charge_points.json`` (written by
``scripts/list_charge_points.py`` in the project environment) and writes, under
``data/raw/emobpy/<run>/``, one Parquet file per charge point and a
``manifest.json`` with every choice, seed and retry and the SHA-256 of each
file. The project reads only those files.

**What is generated.** Per charge point one vehicle: a driver type drawn with
the shares of the German mobility survey MiD 2017 that emobpy's Germany case
uses (commuters 62 %, of them 78 % full time and 22 % part time; non-commuters
38 %), a vehicle model drawn from the four of the emobpy paper, charging **only
at home** at the charge point's nominal power and, on trips longer than the
battery allows, at fast chargers en route -- energy that never reaches the
low-voltage grid. Home charging is emobpy's ``immediate`` strategy: the energy
the controllable charge point must deliver per stay.

**Three workarounds, each recorded in the manifest.**

* ``emobpy.tools.set_seed`` seeds only numba's generator, while the mobility
  model draws from numpy's global one; same seed, different result. The global
  generators are seeded here as well, which makes runs reproducible.
* emobpy samples daily tours until its rules are met, with no iteration limit;
  a full commuter year did not finish in 30 minutes. Mobility is therefore
  generated **week by week** (Monday to Sunday, every week starting and ending
  at home), each in its own process under a time limit.
* A week that exceeds the limit (or fails) is **retried with the next seed**.
  The 2016 run needed 56 retries over 265 full-time commuter weeks and 2 over
  106 non-commuter weeks. The retry keeps only weeks emobpy can produce, which
  is what an unlimited run would also deliver if it ever finished, but it is a
  selection; every attempt is in the manifest.

**Reproducing a run.** Whether a week times out depends on the machine, so a
fresh run on another machine (or under other load) may choose other seeds for
weeks near the limit. ``--replay <manifest>`` takes the successful seed of
every week from an earlier manifest and runs it without a time limit; its
output is bit-identical to the run it replays. The charging of each week is
seeded with the week's seed as well -- emobpy draws the fast charger en route
from numpy's global generator.

The state of charge at the end of a week is the start of the next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DRIVER_SHARES = {"fulltime": 0.62 * 0.78, "parttime": 0.62 * 0.22, "freetime": 0.38}
DEPARTURE_STATS = {
    "fulltime": "DepartureDestinationTrip_Worker.csv",
    "parttime": "DepartureDestinationTrip_Worker.csv",
    "freetime": "DepartureDestinationTrip_Free.csv",
}
VEHICLES = (
    ("Hyundai", "KONA Electric 64 kWh", 2019),
    ("Renault", "Zoe Q90", 2019),
    ("Tesla", "Model 3 Long Range AWD", 2020),
    ("Volkswagen", "ID.3", 2020),
)
FIRST_MONDAY = date(2015, 12, 28)  # the week containing 1 January 2016
DESTINATIONS = ("errands", "escort", "leisure", "shopping", "workplace")


def _seed_all(seed: int) -> None:
    """Seed every generator emobpy draws from; set_seed alone does not."""
    from emobpy.tools import set_seed

    set_seed(seed=seed, dir="config_files")
    np.random.seed(seed)
    random.seed(seed)


def _mobility_job(driver: str, seed: int, monday: str, folder: str) -> None:
    """One week of mobility; run as ``generate.py --mobility-job`` in its own process."""
    from emobpy import Mobility

    _seed_all(seed)
    m = Mobility(config_folder="config_files")
    m.set_params(
        "week", 168, 0.25, driver, date.fromisoformat(monday).strftime("%m/%d/%Y")
    )
    m.set_stats("TripsPerDay.csv", DEPARTURE_STATS[driver], "DistanceDurationTrip.csv")
    m.set_rules(driver)
    m.run()
    m.save_profile(str(folder))


def _mobility_with_retries(
    task: dict, timeout_s: float | None, max_attempts: int
) -> dict:
    """Run one week; on time-out or failure, the next seed. Returns the record.

    A replayed week (``task["replay_seed"]``) runs only the seed that succeeded
    in the run it replays, without a time limit.
    """
    attempts = []
    if task.get("replay_seed") is not None:
        seeds = [task["replay_seed"]]
        timeout_s = None
    else:
        base = task["base_seed"] * 1_000_000 + task["cp"] * 10_000 + task["week"] * 100
        seeds = [base + attempt for attempt in range(max_attempts)]
    for seed in seeds:
        folder = Path(task["work"]) / f"cp{task['cp']}_w{task['week']:02d}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        command = [
            sys.executable,
            __file__,
            "--mobility-job",
            json.dumps([task["driver"], seed, str(task["monday"]), str(folder)]),
        ]
        started = time.time()
        try:
            proc = subprocess.run(command, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            attempts.append(
                {
                    "seed": seed,
                    "outcome": "timeout",
                    "seconds": round(time.time() - started, 1),
                }
            )
            continue
        elapsed = time.time() - started
        if proc.returncode != 0 or not any(folder.glob("*.pickle")):
            lines = proc.stderr.decode(errors="replace").strip().splitlines()
            attempts.append(
                {
                    "seed": seed,
                    "outcome": "failed",
                    "seconds": round(elapsed, 1),
                    "error": lines[-1] if lines else f"exit code {proc.returncode}",
                }
            )
            continue
        attempts.append({"seed": seed, "outcome": "ok", "seconds": round(elapsed, 1)})
        return {**task, "folder": str(folder), "seed": seed, "attempts": attempts}
    raise RuntimeError(
        f"charge point {task['cp']} week {task['week']}: no week after {attempts}"
    )


def _week_charging(
    record: dict, vehicle, power_kw: float, soc_init: float
) -> tuple[pd.DataFrame, float, dict]:
    """Consumption, availability and immediate home charging for one week.

    Seeded with the week's mobility seed: availability draws the fast charger
    en route from numpy's global generator.
    """
    from emobpy import Availability, Charging, Consumption, DataBase, HeatInsulation

    _seed_all(record["seed"])

    db = DataBase(record["folder"])
    db.loadfiles_batch(kind="driving")
    mname = next(iter(db.db))
    consumption = Consumption(mname, vehicle)
    consumption.load_setting_mobility(db)
    consumption.run(
        heat_insulation=HeatInsulation(True),
        weather_country="DE",
        weather_year=2016,
        passenger_mass=75,
        passenger_sensible_heat=70,
        passenger_nr=1.5,
        air_cabin_heat_transfer_coef=20,
        air_flow=0.02,
        driving_cycle_type="WLTC",
        road_type=0,
        road_slope=0,
    )
    consumption.save_profile(record["folder"])
    db.update()
    db.loadfiles_batch(kind="consumption")
    cname = next(k for k, v in db.db.items() if v["kind"] == "consumption")
    stations = {
        "prob_charging_point": {
            **{d: {"none": 1.0} for d in DESTINATIONS},
            "home": {"home": 1.0},
            # Fast charging only en route; emobpy's own values, which keep it to
            # trips the battery cannot cover.
            "driving": {"none": 0.99, "fast75": 0.005, "fast150": 0.005},
        },
        "capacity_charging_point": {
            "home": power_kw,
            "none": 0,
            "fast75": 75,
            "fast150": 150,
        },
    }
    availability = Availability(cname, db)
    availability._set_battery_rules(soc_init=soc_init)  # noqa: SLF001 - no public setter
    availability.set_scenario(stations)
    availability.run()
    availability.save_profile(record["folder"])
    db.update()
    db.loadfiles_batch(kind="availability")
    aname = next(k for k, v in db.db.items() if v["kind"] == "availability")
    charging = Charging(aname)
    charging.load_scenario(db)
    charging.set_sub_scenario("immediate")
    charging.run()
    frame = charging.timeseries.copy()
    start = pd.Timestamp(record["monday"])
    frame.insert(0, "datetime_local", start + pd.to_timedelta(frame["hh"], unit="h"))
    meta = {"availability_success": bool(availability.success)}
    return frame, float(frame["actual_soc"].iloc[-1]), meta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _relative(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)


def main() -> None:
    """Generate, then write the files and the manifest."""
    if len(sys.argv) == 3 and sys.argv[1] == "--mobility-job":
        _mobility_job(*json.loads(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--charge-points", type=Path, default=ROOT / "configs/ev/charge_points.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/emobpy")
    parser.add_argument("--run", default="2016-home-only")
    parser.add_argument("--base-seed", type=int, default=2016)
    parser.add_argument("--weeks", type=int, default=53)
    parser.add_argument("--charge-points-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per week")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="manifest of an earlier run: reuse its successful seeds, no time limit",
    )
    args = parser.parse_args()
    source = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else None

    import emobpy
    from emobpy import BEVspecs

    if not Path("config_files").exists():  # emobpy's Germany case
        shutil.copytree(
            Path(emobpy.__file__).parent / "data/eg2/config_files", "config_files"
        )

    points = json.loads(args.charge_points.read_text(encoding="utf-8"))["charge_points"]
    points = points[: args.charge_points_limit] if args.charge_points_limit else points
    out = args.out / args.run
    work = out / "_work"
    out.mkdir(parents=True, exist_ok=True)

    # Choices per charge point, from a generator of their own so that changing
    # the number of weeks does not change who drives what.
    plan = []
    drivers, shares = zip(*DRIVER_SHARES.items())
    for cp, point in enumerate(points):
        rng = np.random.default_rng([args.base_seed, cp])
        driver = str(rng.choice(drivers, p=np.array(shares) / sum(shares)))
        vehicle = VEHICLES[int(rng.integers(len(VEHICLES)))]
        plan.append({"cp": cp, "point": point, "driver": driver, "vehicle": vehicle})

    # A replay takes, per week, the seed that succeeded and the attempts that led
    # to it; the selection the time limit made is kept, not made again.
    history = {}
    if source is not None:
        if source["base_seed"] != args.base_seed or source["weeks"] != args.weeks:
            raise ValueError("replay: base seed or number of weeks differ")
        entries = {e["asset_id"]: e for e in source["charge_points"]}
        for p in plan:
            entry = entries[p["point"]["asset_id"]]
            if (entry["driver"], tuple(entry["vehicle"])) != (p["driver"], p["vehicle"]):
                raise ValueError(f"replay: {entry['asset_id']} drew differently")
            for w in entry["weeks"]:
                history[(p["cp"], w["week"])] = w["attempts"]

    tasks = [
        {
            "cp": p["cp"],
            "week": w,
            "driver": p["driver"],
            "monday": FIRST_MONDAY + timedelta(days=7 * w),
            "base_seed": args.base_seed,
            "work": str(work),
            "replay_seed": history[(p["cp"], w)][-1]["seed"] if history else None,
        }
        for p in plan
        for w in range(args.weeks)
    ]
    print(f"phase A: {len(tasks)} mobility weeks on {args.workers} workers", flush=True)
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(
            pool.map(
                lambda t: _mobility_with_retries(t, args.timeout, args.max_attempts),
                tasks,
            )
        )
    print(f"phase A done in {(time.time() - started) / 60:.1f} min", flush=True)

    bev = BEVspecs()
    manifest = {
        "tool": "tools/emobpy/generate.py",
        "tool_sha256": _sha256(Path(__file__)),
        "versions": {
            "python": sys.version.split()[0],
            "emobpy": ".".join(map(str, emobpy.__version__))
            if isinstance(emobpy.__version__, tuple)
            else str(emobpy.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "base_seed": args.base_seed,
        "timeout_s": args.timeout,
        "replay_of": None
        if source is None
        else {"path": _relative(args.replay), "sha256": _sha256(args.replay)},
        "first_monday": FIRST_MONDAY.isoformat(),
        "weeks": args.weeks,
        "charge_points": [],
    }
    for p in plan:
        vehicle = bev.model(p["vehicle"])
        weeks = sorted(
            (r for r in records if r["cp"] == p["cp"]), key=lambda r: r["week"]
        )
        frames, soc, week_meta = [], 0.8, []
        for record in weeks:
            frame, soc, meta = _week_charging(
                record, vehicle, p["point"]["nominal_power_kw"], soc
            )
            frames.append(frame)
            attempts = history.get((p["cp"], record["week"]), record["attempts"])
            week_meta.append({"week": record["week"], "attempts": attempts, **meta})
        year = pd.concat(frames, ignore_index=True)
        path = out / f"{p['point']['asset_id'].replace(':', '_')}.parquet"
        year.to_parquet(path, index=False)
        manifest["charge_points"].append(
            {
                **p["point"],
                "driver": p["driver"],
                "vehicle": list(p["vehicle"]),
                "battery_capacity_kwh": float(vehicle.parameters["battery_cap"]),
                "charging_eff": float(
                    vehicle.parameters.get("battery_charging_eff", 0.9)
                ),
                "file": path.name,
                "sha256": _sha256(path),
                "weeks": week_meta,
            }
        )
        print(
            f"{p['point']['asset_id']}: {p['driver']}, "
            + " ".join(map(str, p["vehicle"])),
            flush=True,
        )

    retries = {}
    for cp in manifest["charge_points"]:
        n = sum(len(w["attempts"]) - 1 for w in cp["weeks"])
        retries.setdefault(cp["driver"], [0, 0])
        retries[cp["driver"]][0] += n
        retries[cp["driver"]][1] += len(cp["weeks"])
    manifest["retries_by_driver"] = {
        d: {"retries": r, "weeks": w} for d, (r, w) in retries.items()
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shutil.rmtree(work, ignore_errors=True)
    print(f"written: {out}", flush=True)


if __name__ == "__main__":
    main()
