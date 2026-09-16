"""Protokolle der austauschbaren Komponenten.

Hier stehen nur Schnittstellen, keine Implementierungen. Der Zweck ist, die
Vertraege festzuschreiben, die der Erweiterbarkeits-Contract (§13 des
Architekturdokuments) schuetzt -- insbesondere jene, deren Nachruestung teuer
bis unmoeglich waere:

* :class:`FlexAsset` mit reiner Dynamikfunktion (I2) und physikalischen
  Aktionsgrenzen (I4),
* :class:`PowerFlowEngine` mit hypothetischem Aufruf (I5),
* :class:`SafetyComponent` und :class:`SafetyCertifier` als Eingriffspunkte,
  die ab M0 mit Null-Implementierung existieren (I7).

Konkrete Implementierungen folgen ab M2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.schemas import (
    AssetRatings,
    AssetState,
    ExogenousInput,
    GridState,
    Interval,
    SystemState,
)

__all__ = [
    "ActionSpec",
    "Setpoint",
    "AssetOutcome",
    "FlexAsset",
    "PowerFlowEngine",
    "Verdict",
    "UncertaintySet",
    "SafetyCertifier",
    "InterventionInfo",
    "SafetyComponent",
    "NullSafetyComponent",
    "Controller",
]


# ---------------------------------------------------------------------------
# Aktionen und Setpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Aktionsraum einer Anlage, in **physikalischen** Einheiten.

    **Invariante I4.** Aktionen sind Leistungsgroessen, keine normierten
    Anteile und keine Zielwerte. Semantiken wie "SoC-Zielwert" oder "Anteil der
    Restenergie" erzeugen eine nicht boxfoermige zulaessige Menge in
    Aktionskoordinaten, waehrend die zertifizierte Zulaessigkeitsmenge in
    Einspeisungen formuliert ist. Die Normierung fuer den Agenten erfolgt im
    ``ActionMapper`` und ist dort affin und invertierbar.

    Args:
        names: Sprechende Namen der Komponenten, z. B. ``("p_mw", "q_mvar")``.
        bounds: Grenzen je Komponente.
        discrete_levels: Bei diskretisierten Varianten die Anzahl Stufen je
            Komponente, sonst ``None``. Wird fuer den Masking-Arm des
            Safe-RL-Vergleichs (§6.8) benoetigt.
    """

    names: tuple[str, ...]
    bounds: tuple[Interval, ...]
    discrete_levels: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.names) != len(self.bounds):
            raise ValueError("names und bounds muessen gleich lang sein")
        if self.discrete_levels is not None and len(self.discrete_levels) != len(
            self.names
        ):
            raise ValueError("discrete_levels muss zu names passen")

    @property
    def dim(self) -> int:
        """Dimension des Aktionsraums dieser Anlage."""
        return len(self.names)


@dataclass(frozen=True, slots=True)
class Setpoint:
    """Ausgefuehrter Arbeitspunkt einer Anlage, Verbraucher-Zaehlpfeil.

    ``clipping_info`` ist nicht optional gedacht: **jede** Begrenzung, die eine
    Anlage an der vorgeschlagenen Aktion vornimmt, wird hier gemeldet und
    nicht still vorgenommen (Invariante I4). Nur so laesst sich spaeter
    unterscheiden, ob eine Policy systematisch Unzulaessiges vorschlaegt oder
    ob ein Sicherheitsmechanismus eingegriffen hat.
    """

    asset_id: str
    p_mw: float
    q_mvar: float = 0.0
    clipping_info: Mapping[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.clipping_info is None:
            object.__setattr__(self, "clipping_info", {})

    @property
    def was_clipped(self) -> bool:
        """Wurde die vorgeschlagene Aktion begrenzt?"""
        return bool(self.clipping_info)


@dataclass(frozen=True, slots=True)
class AssetOutcome:
    """Fuer Reward und KPI relevante Folgen eines Zeitschritts.

    Die Groessen sind physikalisch und ungewichtet. Die Gewichtung passiert im
    ``RewardComposer``; der Vergleich mit Referenzverfahren erfolgt auf diesen
    Rohgroessen und nie auf dem Reward (§6.4).
    """

    curtailed_energy_mwh: float = 0.0
    unserved_energy_mwh: float = 0.0
    comfort_deviation_kh: float = 0.0
    throughput_energy_mwh: float = 0.0
    switching_count: int = 0


# ---------------------------------------------------------------------------
# Anlagen
# ---------------------------------------------------------------------------


@runtime_checkable
class FlexAsset(Protocol):
    """Steuerbare Anlage am Netz.

    **Invariante I2:** :meth:`dynamics` ist eine reine Funktion. Gleiche
    Eingabe liefert gleiche Ausgabe, und keine Eingabe wird veraendert. Ein
    praediktiver Sicherheitsfilter muss Anlagenzustaende ueber einen Horizont
    hypothetisch fortschreiben; mit zustandsmutierenden Methoden waere das nur
    durch Neuschreiben aller Anlagenmodelle nachruestbar.
    """

    asset_id: str
    bus: int
    ratings: AssetRatings

    def action_spec(self) -> ActionSpec:
        """Aktionsraum in physikalischen Einheiten."""
        ...

    def initial_state(self, rng: np.random.Generator) -> AssetState:
        """Anfangszustand, gezogen mit dem uebergebenen Generator."""
        ...

    def to_setpoint(
        self, s: AssetState, action: np.ndarray, info: InformationSet
    ) -> Setpoint:
        """Aktion in einen Arbeitspunkt abbilden.

        Begrenzungen auf den physikalisch zulaessigen Bereich sind erlaubt,
        muessen aber in ``Setpoint.clipping_info`` gemeldet werden.
        """
        ...

    def dynamics(
        self,
        s: AssetState,
        sp: Setpoint,
        x: ExogenousInput,
        g: GridState,
    ) -> tuple[AssetState, AssetOutcome]:
        """Zustandsfortschreibung als reine Funktion."""
        ...


# ---------------------------------------------------------------------------
# Netzphysik
# ---------------------------------------------------------------------------


@runtime_checkable
class PowerFlowEngine(Protocol):
    """Lastflussrechnung mit hypothetischem Aufruf.

    **Invariante I5:** :meth:`run_hypothetical` darf den Zustand des laufenden
    Netzes nicht veraendern. Rueckfalltrajektorien eines praediktiven
    Sicherheitsfilters und die Falsifikationssuche der Verifikation brauchen
    genau das.
    """

    def run(self, setpoints: Mapping[str, Setpoint], t_index: int) -> GridState:
        """Lastfluss auf dem laufenden Netz rechnen und Zustand fortschreiben."""
        ...

    def run_hypothetical(
        self, setpoints: Mapping[str, Setpoint], t_index: int
    ) -> GridState:
        """Lastfluss rechnen, ohne das laufende Netz zu veraendern."""
        ...


# ---------------------------------------------------------------------------
# Sicherheit
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Ergebnis einer Zulaessigkeitspruefung -- bewusst asymmetrisch.

    Es gibt kein ``UNSAFE``. ``NOT_CERTIFIED`` bedeutet "nicht beweisbar
    zulaessig", und das genuegt, um den Rueckfall auszuloesen. Ein
    Zertifizierer darf konservativ sein, aber nie optimistisch -- diese
    Asymmetrie ist die Grundlage jeder spaeteren Garantieaussage (§6.9).
    """

    CERTIFIED_SAFE = "certified_safe"
    NOT_CERTIFIED = "not_certified"


@dataclass(frozen=True, slots=True)
class UncertaintySet:
    """Menge der moeglichen Realisierungen der nicht steuerbaren Einspeisungen.

    Args:
        series_ids: Namen der betroffenen Zeitreihen.
        bounds_mw: Form ``(2, n)``, Unter- und Obergrenzen.
        confidence: ``None`` bei deterministischen, aus Anlagen-Ratings
            abgeleiteten Schranken -- dann gilt die Aussage ohne
            Restwahrscheinlichkeit. Bei datengetriebenen Mengen die Konfidenz
            ``1 - delta``, die dann in jeder Garantieaussage mitgenannt werden
            muss (§6.9, Baustein 2).
    """

    series_ids: tuple[str, ...]
    bounds_mw: np.ndarray
    confidence: float | None = None


@runtime_checkable
class SafetyCertifier(Protocol):
    """Prueft, ob eine Aktion nachweisbar zulaessig ist.

    Implementierungen ab M7b: robuste konvexe Restriktion, alternativ
    Linearisierung mit rigoroser Restgliedschranke. Reine Linearisierung ohne
    Restgliedschranke erfuellt dieses Protokoll ausdruecklich **nicht** -- sie
    gehoert in die heuristischen Mechanismen (§6.8).
    """

    def certify(
        self, state: SystemState, action: np.ndarray, uncertainty: UncertaintySet
    ) -> Verdict:
        """Ist die Aktion fuer alle Realisierungen der Unsicherheitsmenge zulaessig?"""
        ...


@dataclass(frozen=True, slots=True)
class InterventionInfo:
    """Protokoll eines Sicherheitseingriffs, Grundlage der Safety-KPIs (§8.3)."""

    intervened: bool
    magnitude: float = 0.0
    """Norm der Aktionsaenderung, ``||a' - a||``."""
    reason: str = ""


@runtime_checkable
class SafetyComponent(Protocol):
    """Sicherheitseingriff *innerhalb* der Environment.

    **Invariante I7.** Bewusst kein Gym-Wrapper: ein Wrapper sieht nur die
    Beobachtung, ein Zertifizierer braucht den vollen ``SystemState``. Der
    zweite Eingriffspunkt fuer Masking liegt dagegen ausserhalb der
    Environment und wird ueber ``info["action_mask"]`` bereitgestellt.
    """

    def transform(
        self, action: np.ndarray, state: SystemState, info: InformationSet
    ) -> tuple[np.ndarray, InterventionInfo]:
        """Aktion gegebenenfalls veraendern."""
        ...

    def action_mask(self, state: SystemState, info: InformationSet) -> np.ndarray | None:
        """Maske zulaessiger diskreter Aktionen, oder ``None``."""
        ...


class NullSafetyComponent:
    """Standardimplementierung: greift nicht ein, erlaubt alles.

    Existiert ab M0, damit der Eingriffspunkt von Anfang an im Ablauf liegt und
    spaeter nur ausgetauscht, nicht eingebaut werden muss.
    """

    def transform(
        self, action: np.ndarray, state: SystemState, info: InformationSet
    ) -> tuple[np.ndarray, InterventionInfo]:
        """Gibt die Aktion unveraendert zurueck."""
        return action, InterventionInfo(intervened=False)

    def action_mask(self, state: SystemState, info: InformationSet) -> np.ndarray | None:
        """Erlaubt alle Aktionen."""
        return None


# ---------------------------------------------------------------------------
# Regler
# ---------------------------------------------------------------------------


@runtime_checkable
class Controller(Protocol):
    """Gemeinsame Schnittstelle von RL-Policies und Referenzverfahren.

    Der Evaluationspfad kennt nur dieses Protokoll. Dadurch laufen
    regelbasierte Verfahren, OPF, MPC und trainierte Policies durch exakt
    dieselbe Auswertung, was den Vergleich strukturell fair macht (§7.1).

    Verfahren, die mehr als das ``InformationSet`` brauchen -- etwa
    ``mpc_oracle`` mit perfekter Vorausschau --, erhalten diesen privilegierten
    Zugriff explizit und werden im Ergebnisbericht entsprechend gekennzeichnet.
    """

    def reset(self, info: InformationSet) -> None:
        """Internen Zustand zu Episodenbeginn zuruecksetzen."""
        ...

    def act(self, info: InformationSet) -> np.ndarray:
        """Aktion fuer den aktuellen Entscheidungszeitpunkt."""
        ...
