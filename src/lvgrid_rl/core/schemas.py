"""Kernschemata der Simulation: Zustand, exogene Eingaenge, Betriebsmittel-Ratings.

Diese Datentypen sind die Schnittstelle, auf die sich alle spaeteren Schichten
stuetzen. Sie werden in M0 zuerst festgelegt, weil jede Komponente, die vorher
entsteht, sich hinterher nach ihnen richten muesste.

Drei Gestaltungsentscheidungen, jeweils mit Begruendung aus dem
Erweiterbarkeits-Contract (§13 des Architekturdokuments):

* **Alles unveraenderlich** (``frozen=True``). Aenderungen erfolgen ueber
  :func:`dataclasses.replace`. Damit ist die Anlagendynamik zwangslaeufig eine
  reine Funktion (Invariante I2), und hypothetisches Vorausrechnen fuer einen
  spaeteren praediktiven Sicherheitsfilter ist ohne Umbau moeglich.
* **numpy-Arrays werden schreibgeschuetzt.** ``frozen=True`` schuetzt nur die
  Referenz, nicht den Inhalt. :func:`freeze_array` setzt daher
  ``flags.writeable = False``. Ohne das ist die Unveraenderlichkeit eine
  Behauptung, kein Fakt.
* **Exogene Eingaenge tragen Schranken, nicht nur Werte** (Invariante I6).
  ``bounds`` wird in M1 trivial aus den Anlagen-Ratings befuellt und zunaechst
  von nichts benutzt. Die Verrohrung existiert dann aber, und das ist der
  Zweck: Unsicherheitsmengen spaeter nachzuruesten waere ein Eingriff in jeden
  Datenpfad.

Einheiten und Vorzeichen: siehe :mod:`lvgrid_rl.core.units`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "freeze_array",
    "Interval",
    "AssetRatings",
    "ExogenousInput",
    "GridState",
    "PQWindowState",
    "PQBudgetState",
    "AssetState",
    "SystemState",
]


def freeze_array(a: np.ndarray) -> np.ndarray:
    """Gibt eine schreibgeschuetzte Sicht auf ``a`` zurueck.

    Es wird bewusst *nicht* kopiert: die Schemata sind heiss im Trainingspfad,
    und eine Kopie je Zeitschritt waere spuerbar. Der Aufrufer gibt mit dem
    Aufruf die Eigentuemerschaft am Array ab.

    >>> arr = freeze_array(np.array([1.0, 2.0]))
    >>> arr.flags.writeable
    False
    """
    view = a.view()
    view.flags.writeable = False
    return view


# ---------------------------------------------------------------------------
# Unsicherheit und Betriebsmittelgrenzen
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    """Geschlossenes Intervall ``[lo, hi]``.

    Traegt absichtlich keine Einheit im Feldnamen: die Einheit ergibt sich aus
    dem Feld, in dem das Intervall verwendet wird (z. B. ``p_bounds_mw``).
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo <= self.hi:
            raise ValueError(f"Leeres Intervall: lo={self.lo} > hi={self.hi}")

    def contains(self, x: float, tol: float = 1e-9) -> bool:
        """Liegt ``x`` im Intervall, mit Toleranz gegen Gleitkommarauschen?"""
        return self.lo - tol <= x <= self.hi + tol

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True, slots=True)
class AssetRatings:
    """Physikalische Grenzen einer Anlage am Netzanschlusspunkt.

    Grundlage der *deterministischen* Unsicherheitsmengen (§6.9, P2 des
    Architekturdokuments). Diese Werte jetzt mitzuschreiben kostet nichts; sie
    spaeter fuer alle Szenarien nachzutragen ist Handarbeit am Netzdatensatz.

    Vorzeichen nach Verbraucher-Zaehlpfeil: fuer eine PV-Anlage ist
    ``p_min_mw`` negativ (volle Einspeisung) und ``p_max_mw == 0.0``.
    """

    p_min_mw: float
    p_max_mw: float
    s_max_mva: float | None = None
    """Scheinleistungsgrenze des Wechselrichters, falls Q-Faehigkeit besteht."""
    fuse_rating_a: float | None = None
    """Nennstrom der Hausanschlusssicherung, falls bekannt."""
    contracted_p_mw: float | None = None
    """Vereinbarte Anschlussleistung, falls bekannt."""

    def __post_init__(self) -> None:
        if not self.p_min_mw <= self.p_max_mw:
            raise ValueError(
                f"p_min_mw={self.p_min_mw} > p_max_mw={self.p_max_mw}"
            )
        if self.s_max_mva is not None and self.s_max_mva < 0.0:
            raise ValueError("s_max_mva darf nicht negativ sein")

    @property
    def p_bounds(self) -> Interval:
        """Leistungsgrenzen als Intervall, fuer Unsicherheitsmengen."""
        return Interval(self.p_min_mw, self.p_max_mw)


@dataclass(frozen=True, slots=True)
class ExogenousInput:
    """Nicht steuerbare Eingangsgroessen zu einem Zeitschritt.

    Enthaelt sowohl die realisierten Werte (fuer die Simulation) als **auch**
    Schranken (fuer eine spaetere robuste Zulaessigkeitspruefung). Invariante I6
    verlangt, dass kein Datenpfad eine Instanz ohne ``bounds`` erzeugt; daher
    ist das Feld nicht optional.

    Args:
        t_index: Zeitschrittindex relativ zum Szenariofenster.
        series_ids: Namen der Zeitreihen, in derselben Reihenfolge wie
            ``realized_mw``.
        realized_mw: Realisierte Wirkleistungen, Verbraucher-Zaehlpfeil.
        bounds_mw: Array der Form ``(2, n)`` mit Unter- und Obergrenzen. In M1
            trivial aus :class:`AssetRatings` befuellt.
        ambient_temp_degc: Aussenlufttemperatur, Eingang des thermischen
            Modells und der COP-Kennlinie.
        ghi_wm2: Globalstrahlung, Eingang der PV-Potenzialrechnung.
    """

    t_index: int
    series_ids: tuple[str, ...]
    realized_mw: np.ndarray
    bounds_mw: np.ndarray
    ambient_temp_degc: float
    ghi_wm2: float

    def __post_init__(self) -> None:
        n = len(self.series_ids)
        if self.realized_mw.shape != (n,):
            raise ValueError(
                f"realized_mw hat Form {self.realized_mw.shape}, erwartet ({n},)"
            )
        if self.bounds_mw.shape != (2, n):
            raise ValueError(
                f"bounds_mw hat Form {self.bounds_mw.shape}, erwartet (2, {n}). "
                "Invariante I6: exogene Eingaenge muessen Schranken tragen."
            )
        lo, hi = self.bounds_mw
        if np.any(lo > hi):
            raise ValueError("bounds_mw: Untergrenze ueber Obergrenze")
        # Der realisierte Wert muss in den Schranken liegen, sonst ist die
        # Unsicherheitsmenge falsch parametriert und jede spaetere
        # Sicherheitsaussage auf ihrer Basis waere ungueltig.
        tol = 1e-9
        outside = (self.realized_mw < lo - tol) | (self.realized_mw > hi + tol)
        if np.any(outside):
            bad = [self.series_ids[i] for i in np.flatnonzero(outside)]
            raise ValueError(
                f"Realisierung liegt ausserhalb der Schranken fuer: {bad}"
            )
        object.__setattr__(self, "realized_mw", freeze_array(self.realized_mw))
        object.__setattr__(self, "bounds_mw", freeze_array(self.bounds_mw))

    def bound_of(self, series_id: str) -> Interval:
        """Schranken einer einzelnen Zeitreihe."""
        i = self.series_ids.index(series_id)
        return Interval(float(self.bounds_mw[0, i]), float(self.bounds_mw[1, i]))


# ---------------------------------------------------------------------------
# Netzzustand
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridState:
    """Ergebnis einer Lastflussrechnung, reduziert auf das Notwendige.

    Bewusst keine pandas-Objekte: dieser Typ wird pro Simulationsschritt
    erzeugt, und DataFrames sind dafuer zu teuer.

    ``converged=False`` ist ein gueltiger Zustand und kein Fehler. Die
    Environment behandelt Nichtkonvergenz als definiertes Ereignis
    (Reward-Strafe plus ``info``-Flag), nicht als Ausnahme.
    """

    t_index: int
    vm_pu: np.ndarray
    """Spannungsbetrag je Bus, Indizierung wie im pandapower-Netz."""
    line_loading_percent: np.ndarray
    trafo_loading_percent: np.ndarray
    p_slack_mw: float
    losses_mw: float
    converged: bool

    def __post_init__(self) -> None:
        for name in ("vm_pu", "line_loading_percent", "trafo_loading_percent"):
            object.__setattr__(self, name, freeze_array(getattr(self, name)))


# ---------------------------------------------------------------------------
# EN-50160-Zustand
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PQWindowState:
    """Laufendes 10-Minuten-Mittelungsfenster eines bewerteten Busses.

    EN 50160 bewertet 10-min-Mittelwerte, nicht Momentanwerte. Dieser Zustand
    haelt die Teilsumme des offenen Fensters; erst am Fensterende entsteht ein
    kriterienrelevanter Wert.
    """

    samples_count: int
    partial_sum_pu: float

    @property
    def mean_pu(self) -> float:
        """Mittelwert des bisher gefuellten Fensteranteils."""
        if self.samples_count == 0:
            return float("nan")
        return self.partial_sum_pu / self.samples_count


@dataclass(frozen=True, slots=True)
class PQBudgetState:
    """Verbrauchtes Verletzungsbudget des laufenden Wochenintervalls.

    EN 50160 erlaubt, dass 5 % der 10-min-Mittelwerte einer Woche ausserhalb
    ±10 % U_n liegen; bei 1008 Fenstern je Woche sind das 50 Fenster je Bus.
    Ohne diesen Zustand in der Beobachtung ist das Regelproblem nicht
    Markov'sch (§6.6 des Architekturdokuments), weshalb er Teil des
    Systemzustands ist und nicht nur eine Auswertungsgroesse.
    """

    windows_elapsed_count: int
    """Abgeschlossene 10-min-Fenster seit Wochenbeginn."""
    windows_total_count: int = 1008
    """Fenster je Wochenintervall: 7 * 24 * 6."""
    violations_k95_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Je bewerteter Bus: Anzahl Fenster ausserhalb ±10 % U_n."""
    violations_k100_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Je bewerteter Bus: Anzahl Fenster ausserhalb +10 % / -15 % U_n."""
    windows: tuple[PQWindowState, ...] = ()
    """Offene Mittelungsfenster, je bewerteter Bus."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "violations_k95_count", freeze_array(self.violations_k95_count)
        )
        object.__setattr__(
            self, "violations_k100_count", freeze_array(self.violations_k100_count)
        )

    @property
    def budget_windows_count(self) -> int:
        """Zulaessige Anzahl verletzender Fenster je Bus und Woche (K95)."""
        return int(0.05 * self.windows_total_count)

    def budget_used_frac(self) -> np.ndarray:
        """Anteil des verbrauchten K95-Budgets je Bus, kann > 1 werden."""
        return self.violations_k95_count / max(self.budget_windows_count, 1)

    def windows_remaining_count(self) -> int:
        return max(self.windows_total_count - self.windows_elapsed_count, 0)


# ---------------------------------------------------------------------------
# Anlagenzustand und Gesamtzustand
# ---------------------------------------------------------------------------


@runtime_checkable
class AssetState(Protocol):
    """Marker-Protokoll fuer Anlagenzustaende.

    Konkrete Implementierungen entstehen in M4 (``lvgrid_rl.components``) und
    muessen unveraenderliche, kopierbare Wertobjekte sein -- Invariante I2. Ein
    Zustand mit Methoden, die ``self`` veraendern, macht den spaeteren
    praediktiven Sicherheitsfilter unmoeglich, weil dieser Anlagenzustaende
    hypothetisch ueber einen Horizont fortschreiben muss.
    """

    asset_id: str


@dataclass(frozen=True, slots=True)
class SystemState:
    """Vollstaendiger Systemzustand zu einem Zeitpunkt.

    **Invariante I1:** Dieser Typ ist die einzige Quelle der Wahrheit. Die
    Beobachtung des Agenten (ab M3) ist eine *Projektion* hiervon und niemals
    selbst Zustandstraeger. Nur so kann ein spaeterer Zertifizierer mehr
    Information erhalten als der Agent -- er ist eine separate, besser
    instrumentierte Komponente (§6.9).

    ``exogenous`` enthaelt die realisierten Werte *bei* ``t_index``. Was zum
    Entscheidungszeitpunkt bekannt sein darf, regelt
    :class:`lvgrid_rl.core.information.InformationSet`; dieser Typ hier ist der
    Vollzustand fuer Simulation, Zertifizierung und Auswertung.
    """

    t_index: int
    timestamp: datetime
    """Zeitstempel in UTC. Kalenderfeatures werden daraus abgeleitet."""
    topology_id: str
    """Kennung der Schalttopologie. Eine Sicherheitsaussage gilt je Topologie."""
    grid: GridState
    assets: Mapping[str, AssetState]
    exogenous: ExogenousInput
    pq: PQBudgetState

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp muss zeitzonenbehaftet sein (UTC). Naive Zeitstempel "
                "erzeugen bei Sommerzeitwechseln stille Fehler."
            )
