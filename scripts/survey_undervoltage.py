#!/usr/bin/env python3
"""Survey whether heat pump and EV load make the lower EN 50160 limit bind.

    uv run python scripts/survey_undervoltage.py --workers 12

M4, step 4.5a. Runs four configurations -- ``moderate_growth`` and
``undervoltage_stress``, each with SimBench profiles and with controllable heat
pumps and charge points -- over all weeks, one week per task on a process pool.
Every finished week is appended to
``results/survey/undervoltage-<scenario>-<assets>.jsonl`` at once, and a
restart skips the weeks already there, so an interrupted survey resumes.

Runs ``do_nothing`` over every week of the committed split (all sets, 51
complete weeks) in evaluation mode and records the ten-minute means at every
assessed connection point. Two asset configurations:

* ``simbench`` -- heat pumps and charge points follow their SimBench profiles,
  the M3 grid;
* ``controllable`` -- heat pumps on their thermostats and vehicles charging on
  arrival (emobpy sessions), which is what "do nothing" means once they are
  controllable (D15, D17).

Reported per configuration, per bus and week and summed: windows below and
above the K95 band (0.90 / 1.10 pu), below the K100 floor (0.85 pu), the
lowest ten-minute mean and the lowest instantaneous value, and how many
bus-weeks fail and on which side. K95 allows 5 % of a week's windows outside
the band (50 of 1,008); a lower-side count is reported against that budget.

The windows are taken from the environment's own EN 50160 aggregator by
wrapping its ``add_sample``; nothing in the environment changes. Writes
``results/survey/undervoltage-<scenario>-<assets>.json`` with the totals.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from lvgrid_rl.baselines.methods import DoNothing
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import DEFAULT_EV, DEFAULT_HEAT_PUMPS, make_env
from lvgrid_rl.grid.pq import K95_BAND_PU, K100_BAND_PU

SETS = ("train", "val", "test", "stress", "holdout", "embargoed")
CONFIGS = (
    ("moderate_growth", "simbench"),
    ("moderate_growth", "controllable"),
    ("undervoltage_stress", "simbench"),
    ("undervoltage_stress", "controllable"),
)
_ENVS: dict = {}


def _week(env, controller) -> dict:
    """One evaluation week: per-bus counts from the recorded window means."""
    windows: list[np.ndarray] = []
    lowest = [np.inf]
    original = env.aggregator.add_sample

    def recording(vm_pu):
        lowest[0] = min(lowest[0], float(np.min(vm_pu)))
        mean = original(vm_pu)
        if mean is not None:
            windows.append(np.array(mean, copy=True))
        return mean

    env.aggregator.add_sample = recording
    try:
        _, info = env.reset(seed=0)
        controller.reset(env._information_set(env._t, env._last_grid))  # noqa: SLF001
        while True:
            state = env._information_set(env._t, env._last_grid)  # noqa: SLF001
            *_, terminated, truncated, _ = env.step(controller.act(state))
            if terminated or truncated:
                break
        state = env.aggregator.state()
    finally:
        env.aggregator.add_sample = original

    means = np.vstack(windows)
    lo95, hi95 = K95_BAND_PU
    lo100, _ = K100_BAND_PU
    below95 = (means < lo95).sum(axis=0)
    above95 = (means > hi95).sum(axis=0)
    below100 = (means < lo100).sum(axis=0)
    budget = env.aggregator.budget_windows
    k95 = np.asarray(state.violations_k95_count)
    k100 = np.asarray(state.violations_k100_count)
    # The recorded windows must reproduce the aggregator's own counts.
    assert np.array_equal(k95, below95 + above95), "window record and aggregator differ"
    failed = (k95 > budget) | (k100 > 0)
    return {
        "week": list(info["week"]),
        "windows": int(means.shape[0]),
        "below_k95": below95.tolist(),
        "above_k95": above95.tolist(),
        "below_k100": below100.tolist(),
        "min_mean_pu": float(means.min()),
        "min_instant_pu": lowest[0],
        "failed_buses": int(failed.sum()),
        "failed_low_buses": int((failed & (below95 >= above95)).sum()),
        "buses": int(k95.size),
        "budget_windows": int(budget),
    }


def _env(code: str, scenario: str, assets: str, set_name: str):
    """The environment for one configuration and set, kept per worker process."""
    key = (code, scenario, assets, set_name)
    if key not in _ENVS:
        # One at a time: twelve workers keeping every environment they met ran
        # the machine out of memory.
        _ENVS.clear()
        controllable = assets == "controllable"
        _ENVS[key] = make_env(
            code=code,
            scenario=scenario,
            set_name=set_name,
            episode_spec=EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False),
            seed=0,
            heat_pumps=DEFAULT_HEAT_PUMPS if controllable else None,
            ev=DEFAULT_EV if controllable else None,
        )
    return _ENVS[key]


def _task(code: str, scenario: str, assets: str, set_name: str, k: int) -> dict:
    """Week ``k`` of one set, in the set's fixed evaluation order."""
    env = _env(code, scenario, assets, set_name)
    env.sampler._cursor = k  # noqa: SLF001 - the k-th week of the fixed order
    record = _week(env, DoNothing(env.mapper))
    return {"scenario": scenario, "assets": assets, "set": set_name, **record}


def _totals(weeks: list[dict]) -> dict:
    total = {
        "weeks": len(weeks),
        "bus_weeks": sum(w["buses"] for w in weeks),
        "failed_bus_weeks": sum(w["failed_buses"] for w in weeks),
        "failed_low_bus_weeks": sum(w["failed_low_buses"] for w in weeks),
        "below_k95_windows": sum(sum(w["below_k95"]) for w in weeks),
        "above_k95_windows": sum(sum(w["above_k95"]) for w in weeks),
        "below_k100_windows": sum(sum(w["below_k100"]) for w in weeks),
        "max_below_k95_bus_week": max(max(w["below_k95"]) for w in weeks),
        "bus_weeks_with_low_windows": sum(
            sum(1 for v in w["below_k95"] if v > 0) for w in weeks
        ),
        "min_mean_pu": min(w["min_mean_pu"] for w in weeks),
        "min_instant_pu": min(w["min_instant_pu"] for w in weeks),
    }
    total["pass_rate"] = 1 - total["failed_bus_weeks"] / total["bus_weeks"]
    return total


def main() -> None:
    """Run the outstanding weeks, then write the totals per configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="1-LV-rural1--2-sw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("results/survey"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def path(scenario, assets, suffix):
        return args.out_dir / f"undervoltage-{scenario}-{assets}.{suffix}"

    done: dict[tuple, list[dict]] = {}
    for scenario, assets in CONFIGS:
        file = path(scenario, assets, "jsonl")
        lines = file.read_text(encoding="utf-8").splitlines() if file.exists() else []
        done[(scenario, assets)] = [json.loads(line) for line in lines if line]

    # The weeks of a set and their order depend on the split and the time index
    # only, so one plain environment per set is enough to list them.
    order = {
        set_name: list(
            _env(args.code, "moderate_growth", "simbench", set_name).sampler._weeks
        )  # noqa: SLF001
        for set_name in SETS
    }
    tasks = []
    for scenario, assets in CONFIGS:
        seen = {(w["set"], tuple(w["week"])) for w in done[(scenario, assets)]}
        for set_name in SETS:
            for k, week in enumerate(order[set_name]):
                if (set_name, tuple(week)) not in seen:
                    tasks.append((args.code, scenario, assets, set_name, k))
    _ENVS.clear()
    print(f"{len(tasks)} weeks outstanding on {args.workers} workers", flush=True)

    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_task, *t) for t in tasks]
        for n, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            key = (record["scenario"], record["assets"])
            with path(*key, "jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            done[key].append(record)
            print(
                f"[{n}/{len(tasks)} {(time.time() - started) / 60:.0f} min] "
                f"{key[0]} {key[1]} {record['set']} {record['week']}: "
                f"min mean {record['min_mean_pu']:.4f}, "
                f"below K95 {sum(record['below_k95'])}, "
                f"above K95 {sum(record['above_k95'])}, "
                f"failed {record['failed_buses']}/{record['buses']}",
                flush=True,
            )

    for (scenario, assets), weeks in done.items():
        if not weeks:
            continue
        total = _totals(weeks)
        path(scenario, assets, "json").write_text(
            json.dumps(
                {"scenario": scenario, "assets": assets, "total": total}, indent=1
            ),
            encoding="utf-8",
        )
        print(f"\n{scenario} / {assets}:\n{json.dumps(total, indent=1)}")


if __name__ == "__main__":
    main()
