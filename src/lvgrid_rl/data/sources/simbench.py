"""Adapter fuer die SimBench-Datensaetze.

Liefert aus einem SimBench-Code zwei Dinge:

* eine Profiltabelle in UTC, normiert auf ``[0, 1]`` bzw. auf die
  Nennleistung des jeweiligen Elements,
* eine Anlagenbeschreibung mit Kategorie, Bus, Nennwerten und
  :class:`~lvgrid_rl.core.schemas.AssetRatings`.

Zur Datenlage, die beim Erstellen dieses Moduls geprueft wurde (SimBench 1.6.1,
Netz ``1-LV-rural1``):

===================  =========================================================
Ausbaustufe          Inhalt
===================  =========================================================
``--0-sw``           13 Haushalte, 4 PV, keine Speicher, keine WP, keine EV
``--1-sw``           zusaetzlich 4 Speicher und eine Erdreich-Waermepumpe
``--2-sw``           28 Lasten inkl. Waermepumpen und Ladepunkten, 8 PV,
                     5 Speicher
===================  =========================================================

Die Profilnamen kodieren die Kategorie: ``H0``/``L1``/``L2``/``G`` sind
Haushalts-, Landwirtschafts- und Gewerbelasten, ``Air_*`` und ``Soil_*`` sind
Luft- bzw. Erdreich-Waermepumpen mit bivalenter Betriebsweise, ``HLS_*`` sind
Ladepunkte mit der Anschlussleistung im Namen (3.7, 11.0, 22.0 kW), ``PV*``
sind Photovoltaikprofile und ``Storage_*`` Speicherprofile.

**Wichtige Einschraenkung zu den Ladepunkten.** Die ``HLS``-Profile sind feste
Lastgaenge, keine Flexibilitaetsbeschreibung: Ankunft, Abfahrt und Energiebedarf
je Ladevorgang lassen sich daraus nicht rekonstruieren. Fuer die
Regelungsaufgabe wird ab M5 ein Session-Modell aus emobpy benoetigt; bis dahin
sind die Ladepunkte unsteuerbare Last.
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

__all__ = ["AssetCategory", "AssetSpec", "SimBenchData", "load_simbench", "categorize"]

SIMBENCH_TIME_FORMAT = "%d.%m.%Y %H:%M"
"""Zeitformat der SimBench-CSV-Dateien: deutsche Ortszeit, hier naiv."""


class AssetCategory(StrEnum):
    """Fachliche Kategorie einer Anlage, abgeleitet aus dem Profilnamen."""

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
        """Kann der Agent diese Anlage stellen?

        Ladepunkte gelten bis zur Einfuehrung des Session-Modells (M5) als
        nicht steuerbar, weil aus einem festen Lastgang keine zulaessige
        Verschiebung ableitbar ist.
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
    """Ordnet einen SimBench-Profilnamen einer Kategorie zu.

    >>> categorize("Air_Semi-Parallel_2")
    <AssetCategory.HEAT_PUMP: 'heat_pump'>
    >>> categorize("HLS_A_22.0")
    <AssetCategory.EV_CHARGER: 'ev_charger'>
    >>> categorize("H0-A")
    <AssetCategory.HOUSEHOLD: 'household'>
    >>> categorize("etwas Unbekanntes")
    <AssetCategory.OTHER: 'other'>
    """
    for pattern, category in _CATEGORY_PATTERNS:
        if pattern.match(profile_name):
            return category
    return AssetCategory.OTHER


def ev_rated_power_kw(profile_name: str) -> float | None:
    """Anschlussleistung eines Ladepunktprofils in kW, sonst ``None``.

    >>> ev_rated_power_kw("HLS_A_22.0")
    22.0
    >>> ev_rated_power_kw("H0-A") is None
    True
    """
    m = re.match(r"^HLS_[A-Z]_([\d.]+)$", profile_name)
    return float(m.group(1)) if m else None


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Beschreibung einer Anlage aus dem Netzdatensatz.

    Args:
        asset_id: Eindeutige Kennung, Form ``"<tabelle>:<index>"``.
        element_table: pandapower-Tabelle (``load``, ``sgen``, ``storage``).
        element_index: Zeilenindex in dieser Tabelle.
        bus: Knoten, an dem die Anlage haengt.
        category: Fachliche Kategorie.
        profile_name: Name des zugehoerigen Profils.
        p_nom_mw: Nennwirkleistung als Betrag, wie im Netzdatensatz hinterlegt.
        ratings: Grenzen im Verbraucher-Zaehlpfeilsystem.
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
    """Ergebnis des Adapters.

    Args:
        code: SimBench-Code des Netzes.
        profiles: Normierte Profile im Wide-Format, Index in UTC. Spalten sind
            Profilnamen, nicht Anlagen -- mehrere Anlagen teilen sich ein
            Profil.
        assets: Anlagenbeschreibungen.
        connection_point_buses: Busse mit mindestens einem kundenseitigen
            Element (Entscheidung D7).
    """

    code: str
    profiles: pd.DataFrame
    assets: tuple[AssetSpec, ...]
    connection_point_buses: tuple[int, ...]

    def profile_for(self, asset: AssetSpec) -> pd.Series:
        """Normiertes Profil einer Anlage."""
        return self.profiles[asset.profile_name]


def _ratings_for(table: str, p_nom_mw: float, s_max_mva: float | None) -> AssetRatings:
    """Leistungsgrenzen im Verbraucher-Zaehlpfeilsystem.

    SimBench speichert ``sgen``-Leistungen im Erzeuger-Zaehlpfeil und
    ``storage`` je nach Datensatz negativ. Die Umrechnung auf die interne
    Konvention (``p_mw > 0`` = Bezug) passiert hier, an genau einer Stelle
    (siehe :data:`lvgrid_rl.core.units.SIGN_CONVENTION`).
    """
    p = abs(p_nom_mw)
    if table == "sgen":  # Einspeiser: nur negative Leistung
        return AssetRatings(p_min_mw=-p, p_max_mw=0.0, s_max_mva=s_max_mva)
    if table == "storage":  # bidirektional
        return AssetRatings(p_min_mw=-p, p_max_mw=p, s_max_mva=s_max_mva)
    return AssetRatings(p_min_mw=0.0, p_max_mw=p, s_max_mva=s_max_mva)


def _profile_frame(net: pandapowerNet) -> pd.DataFrame:
    """Fuehrt die SimBench-Profiltabellen zu einem UTC-Frame zusammen."""
    frames: list[pd.DataFrame] = []
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
                f"Profiltabelle {key!r} hat eine abweichende Zeitachse. Die "
                "Tabellen eines SimBench-Netzes muessen synchron sein."
            )
        body = raw.drop(columns="time").set_axis(idx)
        # Lastprofile liegen als <name>_pload / <name>_qload vor. Fuer M1 wird
        # nur die Wirkleistung uebernommen; die Blindleistung kommt mit der
        # Q-Regelung in M4 dazu und wird dann als eigene Spaltengruppe gefuehrt.
        body = body.rename(columns=lambda c: c[:-6] if c.endswith("_pload") else c)
        body = body.drop(columns=[c for c in body.columns if c.endswith("_qload")])
        frames.append(body)
    if index is None:
        raise ValueError("Netz enthaelt keine Profiltabellen")
    out = pd.concat(frames, axis=1)
    out.index.name = "timestamp_utc"
    return out.loc[:, ~out.columns.duplicated()]


def load_simbench(code: str) -> SimBenchData:
    """Laedt ein SimBench-Netz und ueberfuehrt es in das interne Schema.

    Args:
        code: SimBench-Code, zum Beispiel ``"1-LV-rural1--2-sw"``.

    Returns:
        Profile in UTC und Anlagenbeschreibungen.

    Raises:
        ValueError: wenn eine Anlage auf ein Profil verweist, das in den
            Profiltabellen fehlt.
    """
    import simbench  # lokal importiert: gehoert zum optionalen Extra "sim"

    net = simbench.get_simbench_net(code)
    profiles = _profile_frame(net)

    assets: list[AssetSpec] = []
    buses: set[int] = set()
    for table in ("load", "sgen", "gen", "storage", "asymmetric_load", "asymmetric_sgen"):
        if table not in net or len(net[table]) == 0:
            continue
        df = net[table]
        for idx, row in df.iterrows():
            if not bool(row.get("in_service", True)):
                continue
            profile_name = str(row.get("profile", "")) or ""
            if profile_name and profile_name not in profiles.columns:
                raise ValueError(
                    f"{table}[{idx}] verweist auf Profil {profile_name!r}, das in "
                    "den Profiltabellen fehlt"
                )
            category = categorize(profile_name)
            p_nom = float(row["p_mw"])
            s_max = row.get("sn_mva")
            assets.append(
                AssetSpec(
                    asset_id=f"{table}:{idx}",
                    element_table=table,
                    element_index=int(idx),
                    bus=int(row["bus"]),
                    category=category,
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
        assets=tuple(assets),
        connection_point_buses=tuple(sorted(buses)),
    )
