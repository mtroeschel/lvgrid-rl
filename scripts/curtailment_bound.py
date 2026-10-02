#!/usr/bin/env python3
"""How little curtailment can resolve the overload? A bound with perfect foresight.

    uv run python scripts/curtailment_bound.py --voltage               # with batteries
    uv run python scripts/curtailment_bound.py --voltage --no-storage  # PV only

For each test week, a linear programme with the whole week known in advance
minimises the curtailed PV energy subject to

* the transformer and every line at or below their rating -- flows from the
  radial topology, the sum of the injections downstream of each element;
* the batteries (M4 sizing) charging and discharging within their power and
  operating limits, with their efficiencies, starting the week at the evaluation
  state of charge and ending it no emptier.

Heat pumps and charge points are left on their SimBench profiles: their
flexibility is not used, which makes the bound conservative in that respect.

With ``--voltage`` every assessed bus is also kept at or below ``--v-max`` at
every step -- stricter than EN 50160, which tolerates 5 % of a week's windows
above 1.10 pu. Without it the programme is not a bound worth having: on the
test weeks it found a plan without curtailment that held the thermal limits and
failed K95 in the power flow, because batteries emptied at night at full power
lift the voltage at the end of the long feeder. Voltage is linearised: the
uncontrolled voltages come from a real power flow per control step
(``do_nothing``), the effect of an injection change from a sensitivity dV/dP
measured on the grid with hypothetical power flows at the step of highest PV
infeed. Reactive power and losses are not in the programme either, so the plan
is **replayed in the full AC power flow** of the environment
(``EpisodeMode.EVALUATE``), and the KPIs of the replay are what counts. A
``--margin`` below one tightens the ratings in the programme to leave room for
what it does not model.

The data is constant within each 15-minute interval, so the programme runs on
the control grid; the replay holds each setpoint for its control step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from lvgrid_rl.baselines.methods import DoNothing
from lvgrid_rl.core.protocols import Setpoint
from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
from lvgrid_rl.env.factory import DEFAULT_STORAGE, make_env
from lvgrid_rl.eval.runner import run_controller

STEP_H = 0.25


def _downstream(net) -> tuple[int, dict[int, set[int]]]:
    """The transformer's low-voltage bus and, per line, the buses behind it."""
    root = int(net.trafo.lv_bus.iloc[0])
    neighbours: dict[int, list[tuple[int, int]]] = {}
    for index, row in net.line.iterrows():
        a, b = int(row.from_bus), int(row.to_bus)
        neighbours.setdefault(a, []).append((b, index))
        neighbours.setdefault(b, []).append((a, index))
    children: dict[int, list[int]] = {}
    line_to: dict[int, int] = {}  # bus -> the line feeding it
    order, seen = [root], {root}
    for bus in order:
        for other, line in neighbours.get(bus, []):
            if other not in seen:
                seen.add(other)
                children.setdefault(bus, []).append(other)
                line_to[other] = line
                order.append(other)
    subtree: dict[int, set[int]] = {}
    for bus in reversed(order):
        subtree[bus] = {bus}.union(*(subtree[c] for c in children.get(bus, [])))
    return root, {line_to[bus]: subtree[bus] for bus in order[1:]}


def _uncontrolled_voltages(env) -> dict[int, np.ndarray]:
    """Voltage at the assessed buses per control step, without any control."""
    positions = env.model.evaluated_bus_positions
    out: dict[int, np.ndarray] = {}
    controller = DoNothing(env.mapper)
    env.sampler.reset_cursor()
    for _ in range(env.sampler.n_weeks):
        env.reset(seed=0)
        while True:
            info = env._information_set(env._t, env._last_grid)  # noqa: SLF001
            t = env._t  # noqa: SLF001
            *_, truncated, _ = env.step(controller.act(info))
            out[t] = np.asarray(env._last_grid.vm_pu)[positions].copy()  # noqa: SLF001
            if truncated:
                break
    env.sampler.reset_cursor()
    return out


def _sensitivity(env, buses: list[int], dp_mw: float = 0.005) -> np.ndarray:
    """dV/dP of the assessed buses for an injection change at each of ``buses``.

    Measured with hypothetical power flows (invariant I5) at the step of highest
    PV infeed, PV at full infeed and batteries idle. Consumer sign: drawing
    more lowers the voltage, so the entries are negative.
    """
    pv = [a for a in env.assets if a.kind == "pv"]
    columns = [env.series_ids.index(a.series_id) for a in pv]
    t = int(np.argmin(env._profiles[:, columns].sum(axis=1)))  # noqa: SLF001
    base = env._setpoints_for(t)  # noqa: SLF001
    for asset in env.assets:
        p = (
            float(env._profiles[t, env.series_ids.index(asset.series_id)])  # noqa: SLF001
            if asset.kind == "pv"
            else 0.0
        )
        base[asset.asset_id] = Setpoint(asset.asset_id, p_mw=p)
    positions = env.model.evaluated_bus_positions
    v0 = np.asarray(env.engine.run_hypothetical(base, t).vm_pu)[positions]
    sensitivity = np.zeros((len(positions), len(buses)))
    for k, bus in enumerate(buses):
        # The PV system at that bus carries the perturbation: dp less infeed,
        # the same injection change as curtailing or charging by dp.
        asset = next(a for a in env.assets if a.kind == "pv" and a.bus == bus)
        perturbed = dict(base)
        perturbed[asset.asset_id] = Setpoint(
            asset.asset_id, p_mw=base[asset.asset_id].p_mw + dp_mw
        )
        v = np.asarray(env.engine.run_hypothetical(perturbed, t).vm_pu)[positions]
        sensitivity[:, k] = (v - v0) / dp_mw
    return sensitivity


def _week_programme(
    env,
    start: int,
    end: int,
    storage: bool,
    margin: float,
    voltage: tuple[dict, np.ndarray, float] | None = None,
) -> dict:
    """Solve one week; returns the plan per control step and its curtailment."""
    net = env.model.net
    stride = env.steps_per_control
    steps = np.arange(start, end, stride)
    n_t = len(steps)
    profiles = env._profiles[steps]  # noqa: SLF001 - the full week, by design
    series = list(env.series_ids)

    bus_of = {}
    for table in ("load", "sgen", "storage"):
        for index, bus in net[table]["bus"].items():
            bus_of[f"{table}:{index}"] = int(bus)
    buses = sorted(set(net.bus.index))
    position = {b: k for k, b in enumerate(buses)}

    pv = [a for a in env.assets if a.kind == "pv"]
    batteries = [a for a in env.assets if a.kind == "bess"] if storage else []
    base = np.zeros((n_t, len(buses)))
    for k, name in enumerate(series):
        if name in bus_of and not name.startswith("heat:"):
            base[:, position[bus_of[name]]] += profiles[:, k]
    available = np.stack(
        [-profiles[:, series.index(a.series_id)] for a in pv], axis=1
    )  # positive

    # Variables: curtailment per PV and step, then per battery and step the
    # charge, the discharge and the energy at the end of the step.
    n_pv, n_b = len(pv), len(batteries)
    n_var = n_t * (n_pv + 3 * n_b)

    def x(i, t):
        return t * n_pv + i

    offset = n_t * n_pv

    def ch(j, t):
        return offset + (t * n_b + j) * 3

    def dis(j, t):
        return ch(j, t) + 1

    def energy(j, t):
        return ch(j, t) + 2

    cost = np.zeros(n_var)
    lower = np.zeros(n_var)
    upper = np.zeros(n_var)
    for t in range(n_t):
        for i in range(n_pv):
            cost[x(i, t)] = STEP_H
            upper[x(i, t)] = available[t, i]
        for j, battery in enumerate(batteries):
            # A trace of throughput cost, so that the battery does not cycle for
            # nothing; three orders below the price of curtailment.
            cost[ch(j, t)] = cost[dis(j, t)] = 1e-3 * STEP_H
            upper[ch(j, t)] = battery.ratings.p_max_mw
            upper[dis(j, t)] = -battery.ratings.p_min_mw
            lower[energy(j, t)] = battery.energy_min_mwh
            upper[energy(j, t)] = battery.energy_max_mwh

    # Net injection of a set of buses at step t, as a sparse row of the
    # variables plus a constant.
    pv_bus = [position[a.bus] for a in pv]
    battery_bus = [position[a.bus] for a in batteries]

    def flow_row(t, members: set[int]):
        cols, vals = [], []
        for i, b in enumerate(pv_bus):
            if b in members:
                cols.append(x(i, t))
                vals.append(1.0)
        for j, b in enumerate(battery_bus):
            if b in members:
                cols += [ch(j, t), dis(j, t)]
                vals += [1.0, -1.0]
        constant = float(base[t, sorted(members)].sum())
        return cols, vals, constant

    root, behind = _downstream(net)
    s_trafo = float(net.trafo.sn_mva.iloc[0]) * margin
    elements = [("trafo", set(range(len(buses))), s_trafo)]
    for line, members in behind.items():
        limit = np.sqrt(3) * 0.4 * float(net.line.max_i_ka.loc[line]) * margin
        elements.append((f"line:{line}", {position[b] for b in members}, limit))

    rows, cols_all, vals_all, rhs = [], [], [], []
    r = 0
    for t in range(n_t):
        for _, members, limit in elements:
            cols, vals, constant = flow_row(t, members)
            # -limit <= constant + a.x <= limit, as two upper bounds.
            for sign in (1.0, -1.0):
                rows += [r] * len(cols)
                cols_all += cols
                vals_all += [sign * v for v in vals]
                rhs.append(limit - sign * constant)
                r += 1
    if voltage is not None:
        # V0[t, b] + sum_k S[b, k] * (injection change at PV bus k) <= v_max.
        v0, sens, v_max = voltage
        pv_buses = sorted({a.bus for a in pv})
        for t, step in enumerate(steps):
            for b in range(sens.shape[0]):
                cols, vals = [], []
                for i, a in enumerate(pv):
                    s_bk = sens[b, pv_buses.index(a.bus)]
                    cols.append(x(i, t))
                    vals.append(s_bk)
                for j, a in enumerate(batteries):
                    s_bk = sens[b, pv_buses.index(a.bus)]
                    cols += [ch(j, t), dis(j, t)]
                    vals += [s_bk, -s_bk]
                rows += [r] * len(cols)
                cols_all += cols
                vals_all += vals
                rhs.append(v_max - float(v0[int(step)][b]))
                r += 1
    a_ub = sp.csr_matrix((vals_all, (rows, cols_all)), shape=(r, n_var))

    eq_rows, eq_cols, eq_vals, eq_rhs = [], [], [], []
    q = 0
    for j, battery in enumerate(batteries):
        start_energy = DEFAULT_STORAGE.evaluation_soc_frac * battery.capacity_mwh
        loss = battery.standing_loss_mw * STEP_H
        for t in range(n_t):
            # E_t - E_{t-1} - eta_c ch dt + dis / eta_d dt = -loss
            eq_rows += [q, q, q]
            eq_cols += [energy(j, t), ch(j, t), dis(j, t)]
            eq_vals += [
                1.0,
                -battery.eta_charge_frac * STEP_H,
                STEP_H / battery.eta_discharge_frac,
            ]
            if t > 0:
                eq_rows.append(q)
                eq_cols.append(energy(j, t - 1))
                eq_vals.append(-1.0)
                eq_rhs.append(-loss)
            else:
                eq_rhs.append(start_energy - loss)
            q += 1
        # No emptier at the end of the week than at its start.
        lower[energy(j, n_t - 1)] = start_energy
    a_eq = sp.csr_matrix((eq_vals, (eq_rows, eq_cols)), shape=(q, n_var)) if q else None

    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.array(rhs),
        A_eq=a_eq,
        b_eq=np.array(eq_rhs) if q else None,
        bounds=np.column_stack([lower, upper]),
        method="highs",
    )
    if result.status != 0:
        return {"status": result.message, "curtailed_mwh": float("nan"), "plan": {}}
    v = result.x
    plan = {}
    for t, step in enumerate(steps):
        pv_set = {a.asset_id: -(available[t, i] - v[x(i, t)]) for i, a in enumerate(pv)}
        battery_set = {
            a.asset_id: v[ch(j, t)] - v[dis(j, t)] for j, a in enumerate(batteries)
        }
        plan[int(step)] = {**pv_set, **battery_set}
    return {
        "status": "optimal",
        "curtailed_mwh": float(sum(v[x(i, t)] for t in range(n_t) for i in range(n_pv)))
        * STEP_H,
        "plan": plan,
    }


class _Replay:
    """Plays a precomputed plan: PV infeed caps and battery setpoints."""

    name = "lp_replay"

    def __init__(self, env, plan: dict) -> None:
        self.env = env
        self.plan = plan

    def reset(self, info) -> None:
        pass

    def act(self, info):
        physical = self.env.mapper.default_physical(info)
        setpoints = self.plan[info.t_index]
        cursor = 0
        for asset, spec in zip(self.env.assets, self.env.mapper.specs, strict=True):
            if asset.asset_id in setpoints:
                physical[cursor] = setpoints[asset.asset_id]
            cursor += spec.dim
        return self.env.mapper.to_normalised(physical, info)


def main() -> None:
    """Solve every test week, then replay the plans in the power flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-storage", action="store_true")
    parser.add_argument("--set", dest="set_name", default="test")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument(
        "--voltage",
        action="store_true",
        help="also keep every assessed bus at or below --v-max at every step",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=1.09,
        help="voltage limit in the programme, below 1.10 for the linearisation "
        "error: at 1.097 a battery discharging 176 kW at night at the end of the "
        "long feeder was predicted at 1.095 pu and reached 1.101 in the power flow",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/bound"))
    args = parser.parse_args()

    storage = not args.no_storage
    spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
    env = make_env(
        seed=0,
        set_name=args.set_name,
        episode_spec=spec,
        storage=DEFAULT_STORAGE if storage else None,
    )
    voltage = None
    if args.voltage:
        print("uncontrolled voltages and sensitivity ...", flush=True)
        pv_buses = sorted({a.bus for a in env.assets if a.kind == "pv"})
        voltage = (_uncontrolled_voltages(env), _sensitivity(env, pv_buses), args.v_max)
    plan, weeks = {}, []
    started = time.time()
    for week in env.sampler._weeks:  # noqa: SLF001
        start, end = env.sampler._week_bounds[week]  # noqa: SLF001
        solved = _week_programme(env, start, end, storage, args.margin, voltage)
        plan.update(solved["plan"])
        weeks.append(
            {
                "week": list(week),
                "status": solved["status"],
                "lp_mwh": solved["curtailed_mwh"],
            }
        )
        print(
            f"{week}: {solved['status']}, {solved['curtailed_mwh']:.3f} MWh", flush=True
        )
    lp_total = sum(w["lp_mwh"] for w in weeks)
    print(f"programme: {lp_total:.2f} MWh in {time.time() - started:.0f} s", flush=True)

    summary = run_controller(
        env, _Replay(env, plan), "lp_replay", args.set_name, seed=0
    ).summary()
    keys = (
        "pass_rate",
        "k95_windows",
        "k100_windows",
        "overload_cost",
        "curtailed_mwh",
        "storage_throughput_mwh",
        "bess_energy_change_mwh",
        "clipped_steps",
    )
    replay = {k: summary[k] for k in keys}
    print(json.dumps(replay, indent=1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.set_name}-{'storage' if storage else 'pv-only'}-m{args.margin}"
    if args.voltage:
        name += f"-v{args.v_max}"
    (args.out_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "storage": storage,
                "margin": args.margin,
                "voltage": args.voltage,
                "v_max": args.v_max,
                "lp_curtailed_mwh": lp_total,
                "weeks": weeks,
                "replay": replay,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
