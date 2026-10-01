"""Charging sessions for the grid's charge points (M4, step 4.4a, decision D16).

SimBench gives charge points as fixed load curves (``HLS_*``), which describe
when energy was drawn but not when a vehicle could have drawn it. A controllable
charge point needs sessions: arrival, departure, energy. Decision D16:

* **Arrival and energy come from SimBench.** Every contiguous charging block of
  an ``HLS_*`` profile is one session; blocks separated by no more than
  ``merge_gap_min`` are one interrupted session. This keeps the timing and the
  energy of the grid dataset, as D15 did for the heat pumps.
* **Departure comes from ElaadNL.** The connection time of private (home)
  charging, conditional on the hour of arrival, is drawn from ElaadNL's
  published distribution; it is the one quantity the load curve cannot supply.
* **Power is the nominal rating** of the profile (3.7, 11 or 22 kW). The
  scenario's growth factor stays on the energy, not on the power: a 22 kW
  wallbox scaled to 28.3 kW does not exist.

A session is feasible by construction: the departure is no earlier than the
energy needs at nominal power, and no later than the next arrival at the same
charge point. Both corrections are counted and reported, because each one moves
the result away from the ElaadNL distribution.

**What the SimBench curves are.** Their blocks are small and frequent -- a
median of 3 to 5 kWh, some 220 a year per charge point -- and at 11 or 22 kW the
power reaches its maximum in only 1 to 7 % of a block. They read as averaged
expected values rather than single real sessions. Combined with overnight
connection times this gives small energies a long window, so the flexibility
the charge points offer is, if anything, overstated. Recorded with D16.

**The ElaadNL file.** ElaadNL publishes aggregated distributions through a
dashboard without a direct download. :func:`read_dwell_csv` reads an
intermediate CSV -- one row per arrival hour and connection-time bin -- into
which the dashboard export is converted once; until it is in place, tests and
development use :func:`synthetic_dwell`, which is labelled as such everywhere it
appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DwellDistribution",
    "SessionTable",
    "extract_sessions",
    "assign_departures",
    "read_dwell_csv",
    "synthetic_dwell",
]


@dataclass(frozen=True, slots=True)
class DwellDistribution:
    """Connection time conditional on the local hour of arrival.

    Args:
        duration_bins_h: Right edges of the connection-time bins, in hours,
            increasing; a draw is uniform within its bin.
        probabilities: Array of shape ``(24, n_bins)``; row ``h`` is the
            distribution for arrivals in local hour ``h`` and sums to one.
        source: Where it came from, for the run record -- ``"synthetic"`` until
            the ElaadNL file is in place.
    """

    duration_bins_h: tuple[float, ...]
    probabilities: np.ndarray
    source: str

    def __post_init__(self) -> None:
        edges = np.asarray(self.duration_bins_h)
        if (
            edges.ndim != 1
            or edges.size == 0
            or np.any(np.diff(edges) <= 0)
            or edges[0] <= 0
        ):
            raise ValueError("duration_bins_h must be positive and increasing")
        p = np.asarray(self.probabilities, dtype=np.float64)
        if p.shape != (24, edges.size):
            raise ValueError(f"probabilities must have shape (24, {edges.size})")
        if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each arrival hour needs a distribution that sums to one")
        object.__setattr__(self, "probabilities", p)

    def draw_h(self, arrival_hour: int, rng: np.random.Generator) -> float:
        """One connection time in hours for an arrival in ``arrival_hour``."""
        edges = np.asarray(self.duration_bins_h)
        k = int(rng.choice(edges.size, p=self.probabilities[arrival_hour]))
        lo = 0.0 if k == 0 else edges[k - 1]
        return float(rng.uniform(lo, edges[k]))


@dataclass(frozen=True, slots=True)
class SessionTable:
    """Sessions of one charge point, on simulation step indices.

    Args:
        arrival: Step at which the vehicle arrives.
        departure: Step at which it leaves (exclusive): it can charge in
            ``[arrival, departure)``.
        energy_mwh: Energy it wants by departure.
        p_max_mw: Nominal charging power of the charge point.
        raised_to_feasible: Sessions whose drawn departure was moved later
            because the energy would not fit at nominal power.
        capped_at_next: Sessions whose departure was moved earlier to the next
            arrival.
        source: The connection-time distribution used.
    """

    arrival: np.ndarray
    departure: np.ndarray
    energy_mwh: np.ndarray
    p_max_mw: float
    raised_to_feasible: int
    capped_at_next: int
    source: str

    def __len__(self) -> int:
        return int(self.arrival.size)


def extract_sessions(
    profile_mw: pd.Series, merge_gap_min: float = 15.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charging blocks of a load curve: start step, end step, energy.

    Args:
        profile_mw: The SimBench charge point profile on the simulation index,
            consumer convention (``>= 0``).
        merge_gap_min: Blocks separated by at most this are one session.

    Returns:
        ``(start, end, energy_mwh)``, with ``end`` exclusive.
    """
    values = profile_mw.to_numpy(dtype=np.float64)
    if np.any(values < -1e-12):
        raise ValueError("a charge point profile must not feed in")
    step_h = (profile_mw.index[1] - profile_mw.index[0]) / pd.Timedelta("1h")
    on = values > 1e-9
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    starts, ends = list(edges[::2]), list(edges[1::2])
    merge_steps = int(round(merge_gap_min / 60.0 / step_h))
    merged_s, merged_e = [], []
    for s, e in zip(starts, ends, strict=True):
        if merged_e and s - merged_e[-1] <= merge_steps:
            merged_e[-1] = e
        else:
            merged_s.append(s)
            merged_e.append(e)
    start = np.asarray(merged_s, dtype=np.int64)
    end = np.asarray(merged_e, dtype=np.int64)
    energy = np.array(
        [values[a:b].sum() * step_h for a, b in zip(start, end, strict=True)]
    )
    return start, end, energy


def assign_departures(
    profile_mw: pd.Series,
    p_max_mw: float,
    dwell: DwellDistribution,
    rng: np.random.Generator,
    merge_gap_min: float = 15.0,
) -> SessionTable:
    """Sessions with arrival and energy from the profile, departure drawn.

    The connection time is drawn for the local hour of arrival, then made
    feasible: no earlier than the energy needs at ``p_max_mw``, no later than
    the next arrival. The SimBench block itself is a lower bound too -- the
    vehicle was there while it drew power.
    """
    start, end, energy = extract_sessions(profile_mw, merge_gap_min)
    index = profile_mw.index
    step_h = (index[1] - index[0]) / pd.Timedelta("1h")
    local_hour = index[start].tz_convert("Europe/Berlin").hour
    departure = np.empty_like(start)
    raised = capped = 0
    n_steps = len(index)
    for i, (a, e, energy_i) in enumerate(zip(start, end, energy, strict=True)):
        drawn = a + int(np.ceil(dwell.draw_h(int(local_hour[i]), rng) / step_h))
        needed = a + int(np.ceil(energy_i / p_max_mw / step_h - 1e-9))
        floor = max(needed, e)
        if drawn < floor:
            raised += 1
            drawn = floor
        limit = start[i + 1] if i + 1 < start.size else n_steps
        if drawn > limit:
            capped += 1
            drawn = limit
        departure[i] = drawn
    if np.any(departure - start < np.ceil(energy / p_max_mw / step_h - 1e-9)):
        raise ValueError(
            "a session cannot deliver its energy before the next arrival even at "
            "nominal power; the profile and the rating are inconsistent"
        )
    return SessionTable(
        arrival=start,
        departure=departure,
        energy_mwh=energy,
        p_max_mw=p_max_mw,
        raised_to_feasible=raised,
        capped_at_next=capped,
        source=dwell.source,
    )


def read_dwell_csv(path: Path | str, source: str) -> DwellDistribution:
    """Read the intermediate connection-time CSV.

    Columns ``arrival_hour`` (local, 0-23), ``duration_h`` (right bin edge) and
    ``probability``; every hour must carry the same bins.
    """
    frame = pd.read_csv(path)
    missing = {"arrival_hour", "duration_h", "probability"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    table = frame.pivot(index="arrival_hour", columns="duration_h", values="probability")
    if list(table.index) != list(range(24)):
        raise ValueError(f"{path} must cover arrival hours 0 to 23")
    if table.isna().any().any():
        raise ValueError(f"{path}: every arrival hour needs every duration bin")
    return DwellDistribution(
        duration_bins_h=tuple(float(c) for c in table.columns),
        probabilities=table.to_numpy(),
        source=source,
    )


def synthetic_dwell() -> DwellDistribution:
    """A stand-in until the ElaadNL file is in place. Not data.

    Evening arrivals stay overnight, daytime arrivals for a few hours -- the
    qualitative shape of home charging, with made-up numbers. Every result that
    rests on it is provisional, and the run record names it as the source.
    """
    bins = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
    rows = []
    for hour in range(24):
        if 16 <= hour or hour < 4:  # evening and night: mostly overnight
            rows.append([0.05, 0.05, 0.10, 0.20, 0.40, 0.20])
        else:  # daytime: short stays
            rows.append([0.20, 0.25, 0.30, 0.15, 0.07, 0.03])
    return DwellDistribution(
        duration_bins_h=bins, probabilities=np.array(rows), source="synthetic"
    )
