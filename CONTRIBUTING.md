# Mitwirken

## Einrichtung

```bash
git clone https://github.com/mtroeschel/lvgrid-rl.git
cd lvgrid-rl
uv sync
uv run pytest
```

`uv.lock` ist versioniert und gehört zur Reproduzierbarkeitskette. Wer eine
Abhängigkeit ändert, committet das aktualisierte Lockfile mit -- die CI läuft
mit `uv sync --locked` und bricht sonst ab.

### Wenn `uv run pytest` ein fehlendes Modul meldet

Dann läuft ein `pytest` aus dem System-PATH statt aus `.venv`: das Skript
`pytest` bringt seinen Interpreter in der Shebang-Zeile mit, und der sieht die
Pakete der Umgebung nicht. Diagnose:

```bash
uv run which pytest      # sollte auf .venv/bin/pytest zeigen
uv run python -c "import sys; print(sys.executable)"
```

Zeigt `which` auf etwas ausserhalb von `.venv`, fehlt pytest in der Umgebung.
Ursache ist dann meist, dass die Werkzeuge in `[project.optional-dependencies]`
statt in `[dependency-groups]` stehen -- uv installiert Extras nicht
standardmäßig, Dependency Groups schon.

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

## Pre-Commit-Hooks

Einmalig je Klon einrichten:

```bash
uv sync
uv run pre-commit install
```

Danach laufen vor jedem Commit genau die Pruefungen, die auch die CI laeuft.
Der Anlass fuer diese Hooks war konkret: in M0 war die CI rot, weil `ruff` nie
lokal gelaufen war -- die Tests waren gruen, der Linter hatte 62 Verstoesse.

Auf dem gesamten Bestand pruefen, etwa nach dem Klonen oder nach einem
`autoupdate`:

```bash
pre-commit run --all-files
```

Der Test-Hook ruft `uv run --frozen python -m pytest` auf und setzt damit
voraus, dass `uv` im PATH liegt. Grund: Git-Hooks laufen mit dem PATH des
aufrufenden Terminals, nicht in der aktivierten `.venv`. Ein direkter Aufruf
von `pytest` wuerde ein Skript aus dem System-PATH treffen, ein Aufruf von
`python -m pytest` scheitert auf Systemen, die nur `python3` kennen, mit
"Executable `python` not found". Wer ohne uv arbeitet, ueberspringt den Hook
einzeln:

```bash
SKIP=pytest git commit -m "..."
```

und verlaesst sich auf die CI. Die Ruff-Hooks sind davon nicht betroffen,
pre-commit verwaltet deren Umgebung selbst.

Sollte die Testsuite mit pandapower und echten Trainingslaeufen spuerbar
langsamer werden, gehoert der Hook nach `stages: [pre-push]`. In M0 liegt die
gesamte Hook-Kette bei gut einer Sekunde.

Zwei Dinge dazu. Erstens ersetzt `pre-commit` die CI nicht: die Hooks sehen nur
die gestageten Dateien und laufen in der lokalen Umgebung, waehrend die CI den
vollstaendigen Baum frisch aufsetzt. Zweitens ist die Ruff-Version in
`.pre-commit-config.yaml` und in den `dev`-Extras von `pyproject.toml` bewusst
identisch gepinnt -- laufen Hook und CI auseinander, meldet der eine, was der
andere durchlaesst, und dann vertraut man beiden nicht mehr.

Ohne Hooks vor dem Push mindestens:

```bash
ruff check src tests
ruff format --check src tests scripts
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
