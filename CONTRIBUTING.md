# Mitwirken

## Einrichtung

```bash
git clone https://github.com/mtroeschel/lvgrid-rl.git
cd lvgrid-rl
pip install -e ".[dev]"
pytest
```

## Drei Regeln, die vor allem anderen kommen

Das sind die Stellen, an denen ein neuer Beitrag am schnellsten Schaden
anrichtet, und alle drei sind durch Tests abgesichert.

**1. Vorzeichen.** Durchgängig Verbraucher-Zählpfeilsystem, für *alle*
Anlagentypen: `p_mw > 0` ist Bezug aus dem Netz, `p_mw < 0` Einspeisung. Eine
PV-Anlage hat immer `p_mw <= 0`. Die Umrechnung auf die pandapower-Konvention
(`sgen` zählt umgekehrt) passiert ausschließlich im Netzadapter
(`lvgrid_rl.grid`) und nirgendwo sonst. Siehe
`lvgrid_rl.core.units.SIGN_CONVENTION`.

**2. Einheiten.** Jedes numerische Feld eines Schematyps trägt ein
Einheiten-Suffix aus `lvgrid_rl.core.units.UNIT_SUFFIXES`. Ein Test erzwingt
das über alle Schematypen. Neue Felder ohne Einheit müssen bewusst in
`UNITLESS_FIELDS` in `tests/test_core.py` aufgenommen werden — das soll eine
Entscheidung sein, keine Nachlässigkeit.

**3. Invarianten.** `tests/test_invariants.py` schützt die sieben Invarianten
des Erweiterbarkeits-Contracts (Abschnitt 13 in `docs/architektur.md`). Offene
Invarianten sind `xfail(strict=True)` markiert. Wird eine Komponente
implementiert und der zugehörige Test unerwartet grün, **bricht der Build** —
dann ist der Marker zu entfernen und die im Docstring genannten Prüfkriterien
sind auszuformulieren. Ein Marker darf nie entfernt werden, ohne den Test
tatsächlich zu schreiben.

## Ablauf

Ein Branch je Meilenstein, zum Beispiel `m1-datenschicht`, und ein Pull Request
auch bei Alleinarbeit. Der PR ist der Ort, an dem CI-Ergebnis, Entscheidungen
und Begründungen dokumentiert sind — für eine wissenschaftliche Arbeit ist das
der Prüfpfad, nicht Bürokratie.

Vor dem Push:

```bash
ruff check src tests
pytest -q
```

## Keine Daten im Repository

Rohdaten und abgeleitete Caches werden nicht versioniert, siehe `data/README.md`.
Reproduzierbarkeit läuft über Konfiguration, Seeds und Datenmanifest.

## KI-Unterstützung kenntlich machen

Teile dieses Projekts entstehen mit Unterstützung eines KI-Assistenten. Solche
Beiträge werden im Commit über einen `Co-authored-by`-Trailer oder in der
PR-Beschreibung ausgewiesen. Für wissenschaftliche Veröffentlichungen gelten
zusätzlich die Offenlegungsregeln der jeweiligen Zeitschrift und Institution.
