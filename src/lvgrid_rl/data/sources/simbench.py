"""Adapter for the SimBench datasets.

Produces two things from a SimBench code:

* profile tables in UTC, normalised to ``[0, 1]`` or to the rated power of the
  respective element, separately for active and reactive power,
* an asset description with category, bus, ratings and
  :class:`~lvgrid_rl.core.schemas.AssetRatings`.

On the data situation, verified while writing this module (SimBench 1.6.1, grid
``1-LV-rural1``):

===================  =========================================================
Development stage    Contents
===================  =========================================================
``--0-sw``           13 households, 4 PV, no storage, no heat pumps, no EV
``--1-sw``           adds 4 storage units and one ground-source heat pump
``--2-sw``           28 loads including heat pumps and charge points, 8 PV,
                     5 storage units
===================  =========================================================

The profile names encode the category: ``H0``/``L1``/``L2``/``G`` are household,
agricultural and commercial loads; ``Air_*`` and ``Soil_*`` are air- and
ground-source heat pumps with the bivalent operating mode in the name; ``HLS_*``
are charge points with the connection power in the name (3.7, 11.0, 22.0 kW);
``PV*`` are photovoltaic profiles and ``Storage_*`` storage profiles.

**Important limitation regarding charge points.** The ``HLS`` profiles are fixed
load curves, not a description of flexibility: arrival, departure and energy
demand per charging session cannot be reconstructed from them. The control task
needs a session model from emobpy (M5); until then charge points are
uncontrollable load.

**Reactive power.** SimBench supplies load profiles as separate ``*_pload`` and
``*_qload`` series. The reactive series are carried through as their own column
group, because they need their own resampling rule and because reference method
B3 (Q(U) characteristic per VDE-AR-N 4105) depends on them. They are not
derived from a power factor: ``H0-A_qload`` ranges from -0.26 to 1.20, so
capacitive behaviour is present in the data and must not be assumed away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from lvgrid_rl.core.schemas import AssetRatings
from lvgrid_rl.data.timebase import to_utc_index

if TYPE_CHECKING:  # pragma: no cover
    from pandapower.auxiliary import pandapowerNet

__all__ = [
    "AssetCategory",
    "AssetSpec",
    "SimBenchData",
    "load_simbench",
    "categorize",
    "ev_rated_power_kw",
    "Q_PREFIX",
    "merge_pq",
    "split_pq",
]

SIMBENCH_TIME_FORMAT = "%d.%m.%Y %H:%M"
"""Time format of the SimBench CSV files: German local time, naive here."""

Q_PREFIX = "q::"
"""Column prefix under which reactive power series are stored in a merged frame."""


class AssetCategory(StrEnum):
    """Domain category of an asset, derived from its profile name."""

    HOUSEHOLD = "household"
    COMMERCIAL = "commercial"
    AGRICULTURE = "agriculture"
    HEAT_PUMP = "heat_pump"
    EV_CHARGER = "ev_charger"
    PV = "pv"
    STORAGE = "storage"
    OTHER = "other"

    @property
    def is_controllable(self) -> bool:
        """Can the agent dispatch this asset?

        Charge points count as uncontrollable until the session model arrives
        (M5), because no admissible shift can be derived from a fixed load
        curve.
        """
        return self in (
            AssetCategory.HEAT_PUMP,
            AssetCategory.PV,
            AssetCategory.STORAGE,
        )


_CATEGORY_PATTERNS: tuple[tuple[re.Pattern[str], AssetCategory], ...] = (
    (re.compile(r"^(Air|Soil)_"), AssetCategory.HEAT_PUMP),
    (re.compile(r"^HLS_"), AssetCategory.EV_CHARGER),
    (re.compile(r"^Storage_"), AssetCategory.STORAGE),
    (re.compile(r"^(PV|WP_|Wind)"), AssetCategory.PV),
    (re.compile(r"^H0"), AssetCategory.HOUSEHOLD),
    (re.compile(r"^L\d"), AssetCategory.AGRICULTURE),
    (re.compile(r"^(G\d|BL-H|WB-H)"), AssetCategory.COMMERCIAL),
)


def categorize(profile_name: str) -> AssetCategory:
    """Map a SimBench profile name onto a category.

    >>> categorize("Air_Semi-Parallel_2")
    <AssetCategory.HEAT_PUMP: 'heat_pump'>
    >>> categorize("HLS_A_22.0")
    <AssetCategory.EV_CHARGER: 'ev_charger'>
    >>> categorize("H0-A")
    <AssetCategory.HOUSEHOLD: 'household'>
    >>> categorize("something unknown")
    <AssetCategory.OTHER: 'other'>
    """
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.match(profile_name):
            return category
    return AssetCategory.OTHER


def ev_rated_power_kw(profile_name: str) -> float | None:
    """Connection power of a charge point profile in kW, otherwise ``None``.

    >>> ev_rated_power_kw("HLS_A_22.0")
    22.0
    >>> ev_rated_power_kw("H0-A") is None
    True
    """
    m = re.match(r"^HLS_[A-Z]_([\d.]+)$", profile_name)
    return float(m.group(1)) if m else None


def merge_pq(p: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Merge active and reactive profiles into one frame for storage.

    Reactive columns are prefixed with :data:`Q_PREFIX`. Keeping them in one
    frame means the cache holds a single artefact whose content hash covers
    both, so a reactive series cannot silently drift out of step with its active
    counterpart.
    """
    renamed = q.rename(columns=lambda c: f"{Q_PREFIX}{c}")
    return pd.concat([p, renamed], axis=1)


def split_pq(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a merged frame back into active and reactive profiles."""
    q_cols = [c for c in frame.columns if c.startswith(Q_PREFIX)]
    p_cols = [c for c in frame.columns if not c.startswith(Q_PREFIX)]
    q = frame[q_cols].rename(columns=lambda c: c[len(Q_PREFIX) :])
    return frame[p_cols], q


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Description of an asset taken from the grid dataset.

    Args:
        asset_id: Unique identifier of the form ``"<table>:<index>"``.
        element_table: pandapower table (``load``, ``sgen``, ``storage``).
        element_index: Row index within that table.
        bus: Bus the asset is connected to.
        category: Domain category.
        profile_name: Name of the associated profile.
        p_nom_mw: Nominal active power as a magnitude, as held in the dataset.
        ratings: Limits in the consumer sign convention.
    """

    asset_id: str
    element_table: str
    element_index: int
    bus: int
    category: AssetCategory
    profile_name: str
    p_nom_mw: float
    ratings: AssetRatings


@dataclass(frozen=True, slots=True)
class SimBenchData:
    """Result of the adapter.

    Args:
        code: SimBench code of the grid.
        profiles: Normalised active power profiles in wide format, index in
            UTC. Columns are profile names, not assets -- several assets share
            one profile.
        q_profiles: Normalised reactive power profiles, same index. Only load
            profiles carry these; the frame is empty for grids without loads.
        assets: Asset descriptions.
        connection_point_buses: Buses carrying at least one customer-side
            element (decision D7).
    """

    code: str
    profiles: pd.DataFrame
    q_profiles: pd.DataFrame
    assets: tuple[AssetSpec, ...]
    connection_point_buses: tuple[int, ...]

    def profile_for(self, asset: AssetSpec) -> pd.Series:
        """Normalised active power profile of an asset."""
        return self.profiles[asset.profile_name]

    def q_profile_for(self, asset: AssetSpec) -> pd.Series | None:
        """Normalised reactive power profile, or ``None`` if there is none."""
        if asset.profile_name not in self.q_profiles.columns:
            return None
        return self.q_profiles[asset.profile_name]

    def merged(self) -> pd.DataFrame:
        """Active and reactive profiles in a single frame, for caching."""
        return merge_pq(self.profiles, self.q_profiles)


def _ratings_for(table: str, p_nom_mw: float, s_max_mva: float | None) -> AssetRatings:
    """Power limits in the consumer sign convention.

    SimBench stores ``sgen`` power in the generator sign convention and
    ``storage`` negative in some datasets. Conversion to the internal convention
    (``p_mw > 0`` means drawing) happens here, in exactly one place (see
    :data:`lvgrid_rl.core.units.SIGN_CONVENTION`).
    """
    p = abs(p_nom_mw)
    if table == "sgen":  # generator: negative power only
        return AssetRatings(p_min_mw=-p, p_max_mw=0.0, s_max_mva=s_max_mva)
    if table == "storage":  # bidirectional
        return AssetRatings(p_min_mw=-p, p_max_mw=p, s_max_mva=s_max_mva)
    return AssetRatings(p_min_mw=0.0, p_max_mw=p, s_max_mva=s_max_mva)


def _profile_frames(net: pandapowerNet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the SimBench profile tables into UTC frames.

    Returns:
        Active power profiles and reactive power profiles, sharing one index.
    """
    active: list[pd.DataFrame] = []
    reactive: list[pd.DataFrame] = []
    index: pd.DatetimeIndex | None = None
    for key in ("load", "renewables", "powerplants", "storage"):
        raw = net["profiles"].get(key)
        if raw is None or raw.empty:
            continue
        idx = to_utc_index(pd.to_datetime(raw["time"], format=SIMBENCH_TIME_FORMAT))
        if index is None:
            index = idx
        elif not index.equals(idx):
            raise ValueError(
                f"Profile table {key!r} has a different time axis. The tables of "
                "one SimBench grid must be synchronous."
            )
        body = raw.drop(columns="time").set_axis(idx)
        q_cols = [c for c in body.columns if c.endswith("_qload")]
        if q_cols:
            reactive.append(body[q_cols].rename(columns=lambda c: c[: -len("_qload")]))
        body = body.drop(columns=q_cols)
        body = body.rename(
            columns=lambda c: c[: -len("_pload")] if c.endswith("_pload") else c
        )
        active.append(body)
    if index is None:
        raise ValueError("Grid contains no profile tables")

    p = pd.concat(active, axis=1)
    p.index.name = "timestamp_utc"
    p = p.loc[:, ~p.columns.duplicated()]

    if reactive:
        q = pd.concat(reactive, axis=1)
        q = q.loc[:, ~q.columns.duplicated()]
    else:
        q = pd.DataFrame(index=index)
    q.index.name = "timestamp_utc"
    return p, q


def load_simbench(code: str) -> SimBenchData:
    """Load a SimBench grid and convert it into the internal schema.

    Args:
        code: SimBench code, for example ``"1-LV-rural1--2-sw"``.

    Returns:
        Profiles in UTC and asset descriptions.

    Raises:
        ValueError: if an asset references a profile missing from the profile
            tables.
    """
    import simbench

    net = simbench.get_simbench_net(code)
    profiles, q_profiles = _profile_frames(net)

    assets: list[AssetSpec] = []
    buses: set[int] = set()
    for table in (
        "load",
        "sgen",
        "gen",
        "storage",
        "asymmetric_load",
        "asymmetric_sgen",
    ):
        if table not in net or len(net[table]) == 0:
            continue
        for idx, row in net[table].iterrows():
            if not bool(row.get("in_service", True)):
                continue
            profile_name = str(row.get("profile", "")) or ""
            if profile_name and profile_name not in profiles.columns:
                raise ValueError(
                    f"{table}[{idx}] references profile {profile_name!r}, which is "
                    "missing from the profile tables"
                )
            p_nom = float(row["p_mw"])
            s_max = row.get("sn_mva")
            assets.append(
                AssetSpec(
                    asset_id=f"{table}:{idx}",
                    element_table=table,
                    element_index=int(idx),
                    bus=int(row["bus"]),
                    category=categorize(profile_name),
                    profile_name=profile_name,
                    p_nom_mw=abs(p_nom),
                    ratings=_ratings_for(
                        table, p_nom, float(s_max) if pd.notna(s_max) else None
                    ),
                )
            )
            buses.add(int(row["bus"]))

    return SimBenchData(
        code=code,
        profiles=profiles,
        q_profiles=q_profiles,
        assets=tuple(assets),
        connection_point_buses=tuple(sorted(buses)),
    )
