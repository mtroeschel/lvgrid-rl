"""Einheiten- und Vorzeichenkanon des Projekts.

Dieses Modul enthaelt keine Logik, sondern eine Festlegung. Es ist bewusst das
erste Modul des Projekts, weil zwei Klassen von Fehlern, die hier entstehen,
spaeter extrem teuer sind:

1. **Gemischte Einheiten.** pandapower arbeitet mit MW/MVar fuer Leistungen,
   aber p.u. fuer Spannungen. Wer intern zusaetzlich kW oder Volt einfuehrt,
   erzeugt versteckte Konversionen. Fuer die spaetere Zertifizierung (§6.9 der
   Architektur) ist das fatal, weil Intervallrechnungen keine unbemerkten
   Faktoren verzeihen.
2. **Gemischte Vorzeichen.** pandapower nutzt fuer ``load`` das
   Verbraucher-Zaehlpfeilsystem (``p_mw > 0`` = Bezug) und fuer ``sgen`` das
   Erzeuger-Zaehlpfeilsystem (``p_mw > 0`` = Einspeisung). Ein Batteriespeicher
   ist beides. Wenn diese Unterscheidung durch den Code wandert, kostet sie
   irgendwann Tage.

Die Festlegung wird durch ``tests/test_core.py`` erzwungen: jedes Feld eines
Schema-Datentyps mit numerischem Typ muss ein bekanntes Einheiten-Suffix tragen.

Siehe Architekturdokument §13 ("Einheitenkanon").
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Vorzeichenkanon
# ---------------------------------------------------------------------------

SIGN_CONVENTION: Final[str] = "consumer"
"""Durchgaengig Verbraucher-Zaehlpfeilsystem, **fuer alle** Anlagentypen.

``p_mw > 0`` bedeutet immer Bezug aus dem Netz, ``p_mw < 0`` immer Einspeisung
in das Netz. Daraus folgt:

===================  ===========================================
Anlage               Wertebereich
===================  ===========================================
Haushaltslast        ``p_mw >= 0``
PV-Anlage            ``p_mw <= 0``
Waermepumpe          ``p_mw >= 0``
Ladepunkt (ohne V2G) ``p_mw >= 0``
Batteriespeicher     ``p_mw > 0`` laden, ``p_mw < 0`` entladen
===================  ===========================================

Die Umrechnung auf die pandapower-Konvention passiert ausschliesslich im
Netz-Adapter (``lvgrid_rl.grid``) und nirgendwo sonst. Der Adapter ist damit
die einzige Stelle, an der ein Vorzeichenfehler entstehen kann, und sie ist
klein genug, um sie vollstaendig zu testen.
"""

# ---------------------------------------------------------------------------
# Einheitenkanon
# ---------------------------------------------------------------------------

UNIT_SUFFIXES: Final[dict[str, str]] = {
    # Leistung und Energie
    "_mw": "Wirkleistung in MW (Verbraucher-Zaehlpfeil, siehe SIGN_CONVENTION)",
    "_mvar": "Blindleistung in MVar (Verbraucher-Zaehlpfeil)",
    "_mva": "Scheinleistung in MVA",
    "_mwh": "Energie in MWh",
    "_kwh": "Energie in kWh (nur in KPI-Ausgaben, nie im Zustand)",
    # Elektrische Groessen
    "_pu": "bezogene Groesse, per unit",
    "_kv": "Spannung in kV (Nennspannungen, nicht Zustandsgroessen)",
    "_a": "Strom in A",
    "_percent": "Auslastung in Prozent (0..100, nicht 0..1)",
    # Thermik und Wetter
    "_degc": "absolute Temperatur in Grad Celsius",
    "_k": "Temperaturdifferenz in Kelvin",
    "_kh": "Komfortabweichung in Kelvinstunden",
    "_wm2": "Bestrahlungsstaerke in W/m^2",
    # Zeit
    "_s": "Dauer in Sekunden",
    "_min": "Dauer in Minuten",
    # Dimensionslos
    "_frac": "dimensionsloser Anteil im Intervall [0, 1]",
    "_count": "Anzahl (ganzzahlig)",
}
"""Zulaessige Einheiten-Suffixe fuer numerische Felder in Schema-Datentypen."""

_SUFFIXES_BY_LENGTH: Final[tuple[str, ...]] = tuple(
    sorted(UNIT_SUFFIXES, key=len, reverse=True)
)


def unit_of(field_name: str) -> str | None:
    """Gibt das Einheiten-Suffix eines Feldnamens zurueck, oder ``None``.

    Die Suche laeuft vom laengsten zum kuerzesten Suffix, damit ``_mwh`` nicht
    versehentlich als ``_mw`` erkannt wird.

    >>> unit_of("p_slack_mw")
    '_mw'
    >>> unit_of("energy_mwh")
    '_mwh'
    >>> unit_of("irgendwas")
    """
    for suffix in _SUFFIXES_BY_LENGTH:
        if field_name.endswith(suffix):
            return suffix
    return None


def describe_unit(field_name: str) -> str:
    """Beschreibung der Einheit eines Feldnamens, fuer Fehlermeldungen."""
    suffix = unit_of(field_name)
    if suffix is None:
        return f"{field_name}: keine bekannte Einheit"
    return f"{field_name}: {UNIT_SUFFIXES[suffix]}"
