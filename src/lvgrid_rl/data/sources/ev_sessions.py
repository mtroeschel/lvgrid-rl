"""Home charging sessions from emobpy vehicle-years (M4, step 4.4a, decision D17).

D17 replaces D16. Sessions come from emobpy (Gaete-Morales et al. 2021, mobility
statistics of the German survey MiD 2017), generated offline by
``tools/emobpy/generate.py`` in a frozen environment of their own, one vehicle
per charge point, charging **only at home** at the charge point's nominal power
and at fast chargers en route on trips longer than the battery allows. The
project never imports emobpy; it reads the Parquet files that tool writes and
checks them against the SHA-256 in its manifest.

From an emobpy vehicle-year, one session is one stay at home: arrival when the
vehicle comes home, departure when it leaves, and as energy what emobpy's
``immediate`` strategy draws from the grid during the stay -- the energy the
vehicle needs before it leaves, after its trips and any charging en route.

**Energy level.** The sessions are scaled so that each charge point's annual
energy equals that of its SimBench curve in the scenario. The emobpy vehicles
keep their structure -- when they come and go, how energy is spread over the
stays -- and the grid dataset keeps its energy, so the uncontrolled grid stays
comparable with M3. The scale factor is reported per charge point.

**Time.** emobpy's clock is local wall-clock time (its weather series are in
Europe/Berlin). Arrivals and departures are converted to UTC as events; on the
two daylight-saving nights a session straddling the change is an hour longer
or shorter on the wall clock than in UTC, which is the physical reading.

Every correction is counted in :class:`SessionTable`: stays without energy
(the battery was full), sessions cut at the edges of the simulation year, and
energy reduced because a scaled session would not fit at nominal power.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SessionTable",
    "EmobpyVehicle",
    "read_emobpy_run",
    "home_sessions",
    "scale_to_annual",
]

EMOBPY_STEP_H = 0.25
"""emobpy's time step, 15 minutes."""


@dataclass(frozen=True, slots=True)
class SessionTable:
    """Charging sessions of one charge point, on simulation step indices.

    Args:
        arrival: Step at which the vehicle arrives.
        departure: Step at which it leaves (exclusive): it can charge in
            ``[arrival, departure)``.
        energy_mwh: Grid-side energy it needs by departure.
        p_max_mw: Nominal power of the charge point.
        counts: Corrections applied, by kind -- see the module docstring.
        scale: Factor applied to bring the annual energy to the target, or 1.
    """

    arrival: np.ndarray
    departure: np.ndarray
    energy_mwh: np.ndarray
    p_max_mw: float
    counts: dict[str, int] = field(default_factory=dict)
    scale: float = 1.0

    def __len__(self) -> int:
        return int(self.arrival.size)

    @property
    def total_energy_mwh(self) -> float:
        """Energy over all sessions."""
        return float(self.energy_mwh.sum())


@dataclass(frozen=True, slots=True)
class EmobpyVehicle:
    """One vehicle-year from the emobpy tool, with its manifest entry."""

    asset_id: str
    frame: pd.DataFrame
    nominal_power_kw: float
    annual_energy_mwh: float
    battery_capacity_kwh: float
    driver: str
    vehicle: tuple
    weeks: list


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_emobpy_run(
    run_dir: Path | str, manifest: Path | str | None = None
) -> dict[str, EmobpyVehicle]:
    """Read the vehicle-years of one tool run, verified against a manifest.

    Args:
        run_dir: The run's directory under ``data/raw/emobpy/``.
        manifest: The manifest to verify against; by default the run's own.
            The committed copy (``configs/ev/emobpy-*.manifest.json``) proves
            the files are the dataset the results were computed on.

    Raises:
        FileNotFoundError: if the run is missing, with the command to make it.
        ValueError: if a file's checksum differs from the manifest.
    """
    run_dir = Path(run_dir)
    manifest_path = Path(manifest) if manifest else run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.exists():
        raise FileNotFoundError(
            f"no emobpy run at {run_dir}. Reproduce it from its committed manifest:\n"
            "    cd tools/emobpy && uv sync --locked && uv run python generate.py "
            "--replay ../../configs/ev/emobpy-<run>.manifest.json"
        )
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for entry in content["charge_points"]:
        path = run_dir / entry["file"]
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"{path} has SHA-256 {digest}, the manifest {entry['sha256']}"
            )
        out[entry["asset_id"]] = EmobpyVehicle(
            asset_id=entry["asset_id"],
            frame=pd.read_parquet(path),
            nominal_power_kw=float(entry["nominal_power_kw"]),
            annual_energy_mwh=float(entry["annual_energy_mwh"]),
            battery_capacity_kwh=float(entry["battery_capacity_kwh"]),
            driver=entry["driver"],
            vehicle=tuple(entry["vehicle"]),
            weeks=entry["weeks"],
        )
    return out


def _to_step(local: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Wall-clock times to simulation steps, rounded up.

    Relative to the start of the index and not clipped: a time before the year
    gives a negative step, one after it a step past the end, so that sessions
    at the edges can be recognised and cut.
    """
    utc = (
        pd.DatetimeIndex(local)
        .tz_localize("Europe/Berlin", ambiguous=True, nonexistent="shift_forward")
        .tz_convert("UTC")
    )
    step_ns = (index[1] - index[0]).value
    offset = utc.asi8 - index[0].value
    return -(-offset // step_ns)


def home_sessions(
    frame: pd.DataFrame, index: pd.DatetimeIndex, p_max_mw: float
) -> SessionTable:
    """Sessions from one emobpy vehicle-year, on a simulation index.

    Args:
        frame: The tool's output: ``datetime_local``, ``state``,
            ``charging_point``, ``charge_grid`` (kW), at 15 minutes.
        index: The simulation index (UTC).
        p_max_mw: Nominal power of the charge point.
    """
    at_home = (frame["charging_point"].to_numpy() == "home") & (
        frame["state"].to_numpy() == "home"
    )
    edges = np.flatnonzero(np.diff(np.r_[0, at_home.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    grid_mwh = frame["charge_grid"].to_numpy(dtype=np.float64) * EMOBPY_STEP_H / 1000.0
    energy = np.array([grid_mwh[a:b].sum() for a, b in zip(starts, ends, strict=True)])

    times = frame["datetime_local"]
    step = pd.Timedelta(hours=EMOBPY_STEP_H)
    arrival = _to_step(times.iloc[starts].reset_index(drop=True), index)
    departure = _to_step((times.iloc[ends - 1] + step).reset_index(drop=True), index)

    counts = {"no_energy": 0, "outside_year": 0, "cut_at_year_edge": 0}
    keep = energy > 1e-9
    counts["no_energy"] = int((~keep).sum())
    n = len(index)
    inside = (departure > 0) & (arrival < n)
    counts["outside_year"] = int((keep & ~inside).sum())
    keep &= inside
    cut = keep & ((arrival < 0) | (departure > n))
    counts["cut_at_year_edge"] = int(cut.sum())
    arrival, departure, energy = arrival[keep], departure[keep], energy[keep]
    # A stay cut at the edge of the year keeps the energy of the part inside it,
    # pro rata -- the vehicle draws it during the stay, not all at its end.
    length = (departure - arrival).astype(np.float64)
    arrival_c, departure_c = np.clip(arrival, 0, n), np.clip(departure, 0, n)
    energy = energy * np.where(
        length > 0, (departure_c - arrival_c) / np.maximum(length, 1), 0
    )
    table = SessionTable(
        arrival=arrival_c,
        departure=departure_c,
        energy_mwh=energy,
        p_max_mw=p_max_mw,
        counts=counts,
    )
    return _fit(table, index)


def _fit(
    table: SessionTable, index: pd.DatetimeIndex, key: str = "reduced_to_fit"
) -> SessionTable:
    """Cap each session's energy at what nominal power delivers in its window."""
    step_h = (index[1] - index[0]) / pd.Timedelta("1h")
    deliverable = (table.departure - table.arrival) * step_h * table.p_max_mw
    over = table.energy_mwh > deliverable + 1e-12
    counts = dict(table.counts)
    counts[key] = counts.get(key, 0) + int(over.sum())
    return SessionTable(
        arrival=table.arrival,
        departure=table.departure,
        energy_mwh=np.minimum(table.energy_mwh, deliverable),
        p_max_mw=table.p_max_mw,
        counts=counts,
        scale=table.scale,
    )


def scale_to_annual(
    table: SessionTable, target_mwh: float, index: pd.DatetimeIndex
) -> SessionTable:
    """Scale the energies so that their sum is ``target_mwh``.

    A factor above one can push a session past what nominal power delivers in
    its window; such sessions are capped and counted (``reduced_after_scaling``),
    so the total may end slightly below the target.
    """
    total = table.total_energy_mwh
    if total <= 0.0:
        raise ValueError("no session energy to scale")
    factor = target_mwh / total
    scaled = SessionTable(
        arrival=table.arrival,
        departure=table.departure,
        energy_mwh=table.energy_mwh * factor,
        p_max_mw=table.p_max_mw,
        counts=table.counts,
        scale=factor,
    )
    return _fit(scaled, index, key="reduced_after_scaling")
