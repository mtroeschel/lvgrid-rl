"""Resampling der Zeitreihen auf die einheitliche Simulationsschrittweite.

Die Aggregationsregel haengt von der physikalischen Bedeutung der Groesse ab,
nicht vom Datentyp. Ein Mittelwert ueber Leistungen erhaelt die Energie; ein
Mittelwert ueber einen binaeren Verfuegbarkeitsstatus ergibt Unsinn. Die
Zuordnung steht deshalb explizit in der Konfiguration und nicht als Default im
Code (Tabelle in Abschnitt 3.3 der Architektur).

Beim Hochrechnen (SimBench liefert 15 min, die Simulation braucht 5 min) gilt
dasselbe umgekehrt: Leistungen werden stueckweise konstant fortgeschrieben,
weil das die Energie des Ursprungsintervalls erhaelt. Lineare Interpolation
taete das nicht und wuerde zusaetzlich Rampen erfinden, die in den Daten nicht
stehen.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

from lvgrid_rl.data.timebase import TimeBase

__all__ = ["Policy", "resample_series", "resample_frame", "DEFAULT_POLICIES"]


class Policy(StrEnum):
    """Aggregationsregel einer Groesse.

    Attributes:
        MEAN: Leistungen. Beim Verdichten energieerhaltender Mittelwert, beim
            Hochrechnen stueckweise konstant.
        DIFF: Energiezaehlerstaende. Beim Verdichten Differenz ueber das
            Intervall.
        LAST: Zustandsgroessen wie Ladezustand. Endwert des Intervalls.
        MAJORITY: Binaere Groessen wie "Fahrzeug angesteckt". Mehrheitsentscheid.
        LINEAR: Stetige Umgebungsgroessen wie Temperatur.
    """

    MEAN = "mean"
    DIFF = "diff"
    LAST = "last"
    MAJORITY = "majority"
    LINEAR = "linear"


DEFAULT_POLICIES: Mapping[str, Policy] = {
    "power": Policy.MEAN,
    "energy_counter": Policy.DIFF,
    "irradiance": Policy.MEAN,
    "temperature": Policy.LINEAR,
    "availability": Policy.MAJORITY,
    "state": Policy.LAST,
}
"""Zuordnung Groessenart -> Regel, wie in ``configs/data/default.yaml``."""


def _infer_source_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Ermittelt die Schrittweite eines regelmaessigen Index.

    Raises:
        ValueError: bei unregelmaessigem Index. Das ist fast immer ein
            unbehandelter Sommerzeitwechsel (siehe
            :func:`lvgrid_rl.data.timebase.to_utc_index`) und darf nicht
            stillschweigend mit einer geschaetzten Frequenz ueberdeckt werden.
    """
    diffs = pd.Series(index).diff().dropna().unique()
    if len(diffs) != 1:
        raise ValueError(
            f"Unregelmaessiger Zeitindex, {len(diffs)} verschiedene Schrittweiten: "
            f"{sorted(pd.to_timedelta(diffs))[:5]}. Meist ein nicht nach UTC "
            "konvertierter Index mit Sommerzeitwechsel."
        )
    return pd.Timedelta(diffs[0])


def resample_series(s: pd.Series, target: pd.Timedelta, policy: Policy) -> pd.Series:
    """Bringt eine Zeitreihe auf die Zielschrittweite.

    Args:
        s: Zeitreihe mit regelmaessigem, tz-behaftetem Index.
        target: Zielschrittweite.
        policy: Aggregationsregel.

    Returns:
        Zeitreihe mit der Zielschrittweite und demselben Zeitbereich.

    Raises:
        ValueError: wenn Quell- und Zielschrittweite kein ganzzahliges
            Verhaeltnis haben. Dann waere jede Umrechnung eine Interpolation
            mit Rasterversatz, und der entstehende Fehler waere nicht
            nachvollziehbar.
    """
    source = _infer_source_freq(pd.DatetimeIndex(s.index))
    if source == target:
        return s.copy()

    if source > target:  # Hochrechnen
        if source % target != pd.Timedelta(0):
            raise ValueError(
                f"Quellschrittweite {source} ist kein Vielfaches der "
                f"Zielschrittweite {target}"
            )
        # Der letzte Quellwert deckt ein volles Quellintervall ab; der Index
        # muss deshalb ueber das letzte Original-Zeitstempel hinaus verlaengert
        # werden, sonst fehlt am Jahresende ein Teilintervall.
        end = s.index[-1] + source - target
        new_index = pd.date_range(s.index[0], end, freq=target)
        if policy in (Policy.MEAN, Policy.MAJORITY, Policy.LAST, Policy.DIFF):
            # Stueckweise konstant. Fuer MEAN erhaelt das die Energie des
            # Quellintervalls, fuer DIFF wird der Zaehlerzuwachs gleichmaessig
            # verteilt (siehe unten).
            out = s.reindex(new_index, method="ffill")
            if policy == Policy.DIFF:
                out = out / (source // target)
            return out
        if policy == Policy.LINEAR:
            return (
                s.reindex(s.index.union(new_index)).interpolate("time").reindex(new_index)
            )
        raise AssertionError(f"Unbehandelte Regel {policy}")

    # Verdichten
    if target % source != pd.Timedelta(0):
        raise ValueError(
            f"Zielschrittweite {target} ist kein Vielfaches der "
            f"Quellschrittweite {source}"
        )
    grouper = s.resample(target, label="left", closed="left")
    if policy == Policy.MEAN:
        return grouper.mean()
    if policy == Policy.DIFF:
        return grouper.sum()
    if policy == Policy.LAST:
        return grouper.last()
    if policy == Policy.LINEAR:
        return grouper.mean()
    if policy == Policy.MAJORITY:
        # Mehrheitsentscheid ueber 0/1; bei Gleichstand faellt die Entscheidung
        # zugunsten von "verfuegbar", weil eine faelschlich als nicht verfuegbar
        # gewertete Ladesaeule Flexibilitaet verschenkt, die real vorhanden war.
        return (grouper.mean() >= 0.5).astype(float)
    raise AssertionError(f"Unbehandelte Regel {policy}")


def resample_frame(
    df: pd.DataFrame,
    timebase: TimeBase,
    policies: Mapping[str, Policy],
    default: Policy = Policy.MEAN,
) -> pd.DataFrame:
    """Resampelt alle Spalten eines DataFrame gemaess Spaltenregeln.

    Args:
        df: Zeitreihen im Wide-Format, Index in UTC.
        timebase: Zielschrittweite.
        policies: Zuordnung Spaltenname -> Regel.
        default: Regel fuer Spalten ohne Eintrag.

    Returns:
        DataFrame mit der Zielschrittweite.
    """
    target = timebase.freq
    cols = {
        name: resample_series(df[name], target, policies.get(name, default))
        for name in df.columns
    }
    out = pd.DataFrame(cols)
    out.index.name = df.index.name
    return out


def energy_error(original: pd.Series, resampled: pd.Series) -> float:
    """Relativer Energiefehler einer Leistungszeitreihe nach dem Resampling.

    Die Werte sind Intervallmittelwerte der Leistung, keine Punktabtastungen.
    Die Energie ist deshalb ``sum(p) * dt`` (Rechteckregel); eine Trapezregel
    waere hier falsch, weil sie die Randintervalle halb gewichtet.

    Fuer :attr:`Policy.MEAN` muss der Fehler exakt null sein, solange die
    Zeitbereiche uebereinstimmen. Dient als Pruefgroesse in Tests und in der
    Datenvalidierung.
    """
    dt_orig = _infer_source_freq(pd.DatetimeIndex(original.index))
    dt_new = _infer_source_freq(pd.DatetimeIndex(resampled.index))
    e_orig = float(original.sum()) * dt_orig.total_seconds()
    e_new = float(resampled.sum()) * dt_new.total_seconds()
    if e_orig == 0.0:
        return abs(e_new)
    return abs(e_new - e_orig) / abs(e_orig)
