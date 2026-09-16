"""Informationsordnung: was zum Entscheidungszeitpunkt bekannt sein darf.

Dies ist die Umsetzung von Invariante I3 (§13 des Architekturdokuments) und
zugleich die unangenehmste der sieben Invarianten, weil ihre Verletzung
unsichtbar bleibt: Ein Regler, der versehentlich die Realisierung des
kommenden Intervalls liest, lernt hervorragend und ist im Betrieb wertlos --
und der Fehler ist diffus ueber den Code verteilt.

Das Modul stellt zwei Dinge bereit:

* :class:`InformationSet` -- der Ausschnitt des Zustands, auf dem die
  Aktionsbildung arbeiten darf. Strukturell so gebaut, dass realisierte
  Zukunftswerte gar nicht hineinpassen.
* :class:`DecisionScope` -- ein Kontextmanager, der Zugriffe auf Zeitschritte
  nach dem Entscheidungszeitpunkt zur Laufzeit unterbindet. Dadurch wird die
  Invariante testbar, statt nur dokumentiert zu sein.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from lvgrid_rl.core.schemas import AssetState, Interval, PQBudgetState, freeze_array

__all__ = [
    "InformationSet",
    "DecisionScope",
    "ClairvoyanceError",
    "current_decision_horizon",
    "assert_readable",
]


class ClairvoyanceError(RuntimeError):
    """Zugriff auf Information, die zum Entscheidungszeitpunkt fehlt.

    Siehe :func:`assert_readable` und :class:`DecisionScope`.
    """


# Der Entscheidungshorizont ist Thread-lokal, damit parallele Environments in
# ``SubprocVecEnv``/``DummyVecEnv`` sich nicht gegenseitig beeinflussen.
_local = threading.local()


def current_decision_horizon() -> int | None:
    """Aktuell gueltiger Entscheidungszeitpunkt.

    Gibt ``None`` zurueck, wenn kein :class:`DecisionScope` aktiv ist.
    """
    return getattr(_local, "horizon", None)


def assert_readable(t_index: int, what: str = "Zeitreihenwert") -> None:
    """Prueft, ob ``t_index`` zum Entscheidungszeitpunkt gelesen werden darf.

    Wird von den Datenzugriffen der Szenarioschicht aufgerufen. Ausserhalb
    eines :class:`DecisionScope` ist jeder Zugriff erlaubt und die Funktion
    kehrt sofort zurueck -- Simulation, Auswertung und Zertifizierung duerfen
    den Vollzustand sehen, und der heisse Pfad soll nichts kosten.

    Innerhalb eines Scope wird der Zugriff zusaetzlich protokolliert, damit
    Tests pruefen koennen, *welche* Zeitpunkte gelesen wurden -- nicht nur,
    dass keine Ausnahme fiel.

    Raises:
        ClairvoyanceError: wenn innerhalb eines Entscheidungsfensters auf einen
            spaeteren Zeitschritt zugegriffen wird.
    """
    horizon = current_decision_horizon()
    if horizon is None:
        return
    recorder = getattr(_local, "recorder", None)
    if recorder is not None:
        recorder.append(t_index)
    if t_index > horizon:
        raise ClairvoyanceError(
            f"{what} fuer t={t_index} angefordert, erlaubt ist hoechstens "
            f"t={horizon}. Invariante I3: die Aktionsbildung darf nur auf "
            "Prognosen zugreifen, nicht auf Realisierungen des kommenden "
            "Intervalls."
        )


class DecisionScope:
    """Kontextmanager, der die Informationsordnung zur Laufzeit erzwingt.

    Umschliesst in der Environment die Schritte 1 und 2 des Ablaufs (§6.1):
    Aktionsabbildung und Setpoint-Bildung. Innerhalb des Blocks fuehrt jeder
    Zugriff auf einen Zeitschritt ``> t_decision`` zu einer
    :class:`ClairvoyanceError`.

    Zugriffe werden zusaetzlich mitgeschrieben, damit Tests pruefen koennen,
    *welche* Zeitpunkte gelesen wurden -- nicht nur, dass keine Ausnahme fiel.

    Example:
        >>> scope = DecisionScope(t_decision=10)
        >>> with scope:
        ...     assert_readable(10)
        ...     assert_readable(8)
        >>> scope.accessed
        (10, 8)
        >>> with DecisionScope(t_decision=10):
        ...     assert_readable(11)
        Traceback (most recent call last):
            ...
        lvgrid_rl.core.information.ClairvoyanceError: ...
    """

    __slots__ = ("t_decision", "_accessed", "_previous")

    def __init__(self, t_decision: int) -> None:
        self.t_decision = t_decision
        self._accessed: list[int] = []
        self._previous: int | None = None

    def __enter__(self) -> DecisionScope:
        self._previous = current_decision_horizon()
        _local.horizon = self.t_decision
        _local.recorder = self._accessed
        return self

    def __exit__(self, *exc_info: object) -> None:
        _local.horizon = self._previous
        _local.recorder = None

    @property
    def accessed(self) -> tuple[int, ...]:
        """Alle innerhalb des Blocks angeforderten Zeitschritte."""
        return tuple(self._accessed)


@contextmanager
def unrestricted() -> Iterator[None]:
    """Hebt die Informationsordnung vorruebergehend auf.

    Nur fuer Komponenten, die den Vollzustand legitim brauchen: Simulation,
    Referenzverfahren mit perfekter Vorausschau (``mpc_oracle``) und
    Auswertung. Jede Verwendung in Agentenpfaden ist ein Fehler und sollte im
    Review auffallen -- deshalb der sprechende Name.
    """
    previous = current_decision_horizon()
    _local.horizon = None
    try:
        yield
    finally:
        _local.horizon = previous


@dataclass(frozen=True, slots=True)
class InformationSet:
    """Was der Regler zum Entscheidungszeitpunkt ``t_index`` wissen darf.

    Strukturell so gewaehlt, dass Hellsichtigkeit nicht ausdrueckbar ist: es
    gibt kein Feld fuer realisierte Werte des kommenden Intervalls. Statt der
    Realisierung stehen Schranken (``exogenous_bounds_mw``) und Prognosen
    (``forecast``) zur Verfuegung.

    Der Unterschied zur Beobachtung des Agenten: das ``InformationSet`` ist das
    *Maximum* des zulaessig Wissbaren. Der ``ObservationBuilder`` (M3) waehlt
    daraus gemaess ``sensor_config`` aus und kann deutlich weniger
    weitergeben. Ein spaeterer Zertifizierer darf dagegen das volle
    ``InformationSet`` nutzen.

    Args:
        t_index: Entscheidungszeitpunkt.
        timestamp: Zeitstempel in UTC.
        measurements: Messwerte gemaess konfigurierter Sensorik, Schluessel in
            der Form ``"vm_pu/bus_17"`` oder ``"trafo_loading_percent/0"``.
        asset_states: Interne Zustaende der eigenen Anlagen.
        exogenous_bounds_mw: Schranken der nicht steuerbaren Einspeisungen
            waehrend ``[t, t + control_dt)``, Form ``(2, n)``.
        series_ids: Namen zu den Spalten von ``exogenous_bounds_mw``.
        forecast: Prognosen je Groesse, jeweils Array der Laenge ``horizon``.
            Erzeugt vom Prognosefehlermodell (§6.7); im Modus ``perfect``
            enthaelt es die Realisierung, und genau das ist dann ein
            ausgewiesener Sonderfall und kein Leck.
        pq: Verbrauchtes EN-50160-Budget. Ohne dieses Feld ist das
            Regelproblem nicht Markov'sch.
    """

    t_index: int
    timestamp: datetime
    measurements: Mapping[str, float]
    asset_states: Mapping[str, AssetState]
    series_ids: tuple[str, ...]
    exogenous_bounds_mw: np.ndarray
    forecast: Mapping[str, np.ndarray]
    pq: PQBudgetState

    def __post_init__(self) -> None:
        n = len(self.series_ids)
        if self.exogenous_bounds_mw.shape != (2, n):
            raise ValueError(
                f"exogenous_bounds_mw hat Form {self.exogenous_bounds_mw.shape}, "
                f"erwartet (2, {n})"
            )
        object.__setattr__(
            self, "exogenous_bounds_mw", freeze_array(self.exogenous_bounds_mw)
        )

    def bound_of(self, series_id: str) -> Interval:
        """Schranken einer einzelnen Zeitreihe im kommenden Intervall."""
        i = self.series_ids.index(series_id)
        return Interval(
            float(self.exogenous_bounds_mw[0, i]),
            float(self.exogenous_bounds_mw[1, i]),
        )
