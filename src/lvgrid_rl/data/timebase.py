"""Zeitbasis der Simulation: Schrittweiten und Zeitzonen.

Zwei Festlegungen, die hier zusammenlaufen und beide aus dem Spannungskriterium
folgen (Abschnitt 6.6 der Architektur):

**Zulaessige Schrittweiten.** EN 50160 bewertet 10-Minuten-Mittelwerte. Eine
Simulationsschrittweite muss 10 Minuten daher ganzzahlig teilen, sonst laesst
sich ein Bewertungsfenster nicht ohne Interpolation aus Simulationsschritten
bilden. Damit fallen die naheliegenden 15 Minuten heraus -- ausgerechnet die
Originalaufloesung der SimBench-Zeitreihen.

**Zeitzonen.** Intern gilt durchgaengig UTC. Der Grund ist nicht Ordnungsliebe,
sondern ein konkreter Fallstrick in den Quelldaten: die SimBench-Zeitachse ist
deutsche Ortszeit *mit* Sommerzeitumstellung. Naiv eingelesen ergibt das einen
Index, der weder monoton noch eindeutig ist -- am 27.03.2016 fehlen vier
Viertelstunden, am 30.10.2016 kommen vier doppelt vor. Wer das uebersieht,
resampelt auf einem kaputten Index und merkt es nie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

__all__ = [
    "PQ_WINDOW_MIN",
    "ALLOWED_SIM_DT_MIN",
    "DEFAULT_SOURCE_TZ",
    "TimeBase",
    "to_utc_index",
]

PQ_WINDOW_MIN: Final[int] = 10
"""Mittelungsintervall der EN 50160 in Minuten."""

ALLOWED_SIM_DT_MIN: Final[tuple[int, ...]] = (1, 2, 5, 10)
"""Schrittweiten, die :data:`PQ_WINDOW_MIN` ganzzahlig teilen."""

DEFAULT_SOURCE_TZ: Final[str] = "Europe/Berlin"
"""Zeitzone der deutschen Quelldatensaetze (SimBench, WPuQ, HTW, DWD)."""

WINDOWS_PER_WEEK: Final[int] = 7 * 24 * 60 // PQ_WINDOW_MIN
"""1008 Bewertungsfenster je Wochenintervall."""


def to_utc_index(
    timestamps: pd.Series | pd.DatetimeIndex,
    source_tz: str = DEFAULT_SOURCE_TZ,
) -> pd.DatetimeIndex:
    """Wandelt naive Ortszeit-Zeitstempel in einen UTC-Index.

    Die Umstellung auf Sommerzeit wird ueber ``ambiguous="infer"`` aufgeloest:
    pandas erkennt den zusammenhaengenden, aufsteigend sortierten Block der
    doppelten Stunde und ordnet ihm die richtigen UTC-Offsets zu. Das
    funktioniert, weil die Quelldaten die Wiederholung tatsaechlich enthalten
    und in der Reihenfolge ihres Auftretens speichern.

    Args:
        timestamps: Naive Zeitstempel in Ortszeit.
        source_tz: Zeitzone der Quelldaten.

    Returns:
        Streng monoton steigender, eindeutiger Index in UTC.

    Raises:
        ValueError: wenn die Umstellung nicht aufloesbar ist oder das Ergebnis
            nicht monoton oder nicht eindeutig ist. Beides deutet auf eine
            luecken- oder reihenfolgegestoerte Quelldatei hin und darf nicht
            stillschweigend weiterverarbeitet werden.
    """
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    try:
        localized = idx.tz_localize(source_tz, ambiguous="infer", nonexistent="raise")
    except Exception as exc:  # pragma: no cover - quelldatenabhaengig
        raise ValueError(
            f"Zeitstempel liessen sich nicht nach {source_tz} lokalisieren: {exc}. "
            "Ursache ist meist eine Quelldatei, in der die doppelte Stunde der "
            "Sommerzeitumstellung fehlt oder unsortiert vorliegt."
        ) from exc
    utc = localized.tz_convert("UTC")
    if not utc.is_monotonic_increasing:
        raise ValueError("UTC-Index ist nicht monoton steigend")
    if not utc.is_unique:
        raise ValueError("UTC-Index enthaelt doppelte Zeitstempel")
    return utc


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Schrittweiten der Simulation und der Regelung.

    Args:
        sim_dt_min: Simulationsschrittweite, muss in
            :data:`ALLOWED_SIM_DT_MIN` liegen.
        control_dt_min: Regelungsschrittweite, muss ein Vielfaches von
            ``sim_dt_min`` sein. Ein Versatz gegenueber dem 10-min-Raster der
            Spannungsbewertung ist ausdruecklich zulaessig und realistisch
            (Abschnitt 6.6): ein 15-min-Regeltakt auf 5-min-Physik ist ein
            gueltiger Fall.

    Example:
        >>> tb = TimeBase(sim_dt_min=5, control_dt_min=15)
        >>> tb.sim_steps_per_pq_window
        2
        >>> tb.sim_steps_per_control_step
        3
        >>> TimeBase(sim_dt_min=15, control_dt_min=15)
        Traceback (most recent call last):
            ...
        ValueError: sim_dt_min=15 ist unzulaessig...
    """

    sim_dt_min: int
    control_dt_min: int

    def __post_init__(self) -> None:
        if self.sim_dt_min not in ALLOWED_SIM_DT_MIN:
            raise ValueError(
                f"sim_dt_min={self.sim_dt_min} ist unzulaessig; erlaubt sind "
                f"{ALLOWED_SIM_DT_MIN}. Die Schrittweite muss das "
                f"{PQ_WINDOW_MIN}-min-Bewertungsintervall der EN 50160 "
                "ganzzahlig teilen."
            )
        if self.control_dt_min % self.sim_dt_min != 0:
            raise ValueError(
                f"control_dt_min={self.control_dt_min} ist kein Vielfaches von "
                f"sim_dt_min={self.sim_dt_min}"
            )

    @property
    def sim_steps_per_pq_window(self) -> int:
        """Simulationsschritte je 10-min-Bewertungsfenster."""
        return PQ_WINDOW_MIN // self.sim_dt_min

    @property
    def sim_steps_per_control_step(self) -> int:
        """Simulationsschritte je Regelentscheidung."""
        return self.control_dt_min // self.sim_dt_min

    @property
    def control_steps_per_week(self) -> int:
        """Regelentscheidungen je Wochenintervall."""
        return 7 * 24 * 60 // self.control_dt_min

    def suggested_gamma(self) -> float:
        """Diskontfaktor, dessen effektiver Horizont eine Woche abdeckt.

        Gamma ist in diesem Projekt kein freier Hyperparameter: das
        Spannungskriterium bezieht sich auf ein Wochenintervall, und ein aus
        Gewohnheit gesetztes ``0.99`` wuerde diesen Horizont schlicht nicht
        sehen (Abschnitt 6.5).

        >>> round(TimeBase(5, 15).suggested_gamma(), 5)
        0.99851
        """
        return 1.0 - 1.0 / self.control_steps_per_week

    @property
    def freq(self) -> pd.Timedelta:
        """Simulationsschrittweite als ``Timedelta``."""
        return pd.Timedelta(minutes=self.sim_dt_min)
