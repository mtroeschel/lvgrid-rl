# lvgrid-rl

Trainingsumgebung für Reinforcement-Learning-Agenten zur autonomen Regelung von
Niederspannungsnetzen. Die Agenten regeln Spannungsbandverletzungen sowie
Leitungs- und Transformatorüberlastungen aus, indem sie PV-Einspeisung abregeln,
Wärmepumpen- und Ladelasten verschieben und Batteriespeicher steuern.

Netzsimulation mit [pandapower](https://pandapower.readthedocs.io) und
[SimBench](https://simbench.de), Agenten mit
[Stable-Baselines3](https://stable-baselines3.readthedocs.io).

**Stand: M0** — Skelett und Kernschemata. Es gibt noch keine Simulation und
keinen Agenten. Was es gibt, ist die Schnittstellenschicht, auf der alles
Weitere aufbaut, und eine geprüfte Reproduzierbarkeitskette.

Die vollständige Architektur steht in `docs/architektur.md`.

## Installation

```bash
pip install -e ".[dev]"          # M0: nur numpy + Testwerkzeuge
pip install -e ".[dev,sim,rl,config]"   # ab M1
```

Die schweren Abhängigkeiten sind absichtlich optional: das Skelett und die
Invariantentests sollen ohne pandapower und PyTorch laufen.

## Rauchtest

```bash
python scripts/train.py --seed 1 --run-dir results/smoke
pytest -v
```

`scripts/train.py` trainiert noch nichts. Es löst die Konfiguration auf, leitet
die Seeds ab und schreibt ein Run-Manifest — damit ist die
Reproduzierbarkeitskette geprüft, bevor das erste Modell existiert.

## Zwei Festlegungen, die man vor dem ersten Beitrag kennen sollte

**Vorzeichen.** Durchgängig Verbraucher-Zählpfeilsystem, für *alle*
Anlagentypen: `p_mw > 0` ist Bezug aus dem Netz, `p_mw < 0` Einspeisung. Eine
PV-Anlage hat also immer `p_mw <= 0`. Die Umrechnung auf die
pandapower-Konvention (`sgen` zählt umgekehrt) passiert ausschließlich im
Netzadapter. Siehe `lvgrid_rl.core.units.SIGN_CONVENTION`.

**Einheiten.** MW, MVar, MWh, p.u., Prozent (0–100, nicht 0–1), °C, K, W/m².
Jedes numerische Feld eines Schematyps trägt ein Einheiten-Suffix, und ein Test
erzwingt das. Der Grund ist nicht Ordnungsliebe: die später geplante
Zertifizierung rechnet mit Intervallen, und die verzeihen keine versteckten
Konversionsfaktoren.

## Aufbau

| Paket | Schicht |
|---|---|
| `core` | Schemata, Protokolle, Einheitenkanon, Informationsordnung |
| `data` | L0 Datenschicht: Quellen, Resampling, Cache, Szenarien |
| `grid` | L1 Netz und Physik |
| `components` | L2 Anlagen- und Flexibilitätsmodelle |
| `env` | L3 Gymnasium-Environment |
| `agents`, `baselines` | L4 Agenten und Referenzverfahren |
| `eval`, `viz` | L5 KPIs, Auswertung, Visualisierung |
| `experiment` | L6 Orchestrierung und Reproduzierbarkeit |

## Erweiterbarkeits-Contract

Die harten Sicherheitsgarantien (robuste konvexe Restriktion, prädiktiver
Sicherheitsfilter) sind auf M7b verschoben. Damit sie dann ohne Umbau ergänzbar
sind, gelten sieben Invarianten, jede mit einem Test in
`tests/test_invariants.py`. Die noch offenen sind als `xfail(strict=True)`
markiert: sobald die zugehörige Komponente existiert und der Test unerwartet
grün wird, **bricht der Build** und erzwingt, den Marker zu entfernen. Die
Anzahl der XFAIL-Meldungen ist der Schuldenstand des Contracts.

| | Invariante | Fällig | Stand in M0 |
|---|---|---|---|
| I1 | `SystemState` vollständig, von `Observation` getrennt | M3 | Teilaussage grün |
| I2 | Anlagendynamik als reine Funktion | M4 | offen |
| I3 | Strikte Informationsordnung | M3 | Mechanismus grün |
| I4 | Aktionen physikalisch, Normierung affin invertierbar | M3 | offen |
| I5 | Hypothetischer Lastfluss ohne Seiteneffekt | M2 | offen |
| I6 | Exogene Eingänge tragen Schranken, Ratings gepflegt | M1 | Schemaebene grün |
| I7 | Zwei Sicherheits-Eingriffspunkte vorhanden | M3 | Null-Implementierung grün |

Der Grund für den Aufwand: Modularitätsversprechen verfallen still. Sechs
Monate ohne Prüfung, und irgendeine sinnvolle Abkürzung hat die Erweiterbarkeit
aufgebraucht, ohne dass es auffällt.

## Nächste Schritte

* **M1** Datenschicht: SimBench-Adapter, Resampling auf `sim_dt ∈ {1,2,5,10}`
  min, Validierung, Parquet-Cache mit Manifest.
* **M2** Netz: Loader, Asset-Mapping, `PowerFlowEngine` inklusive
  hypothetischem Aufruf, `PQAggregator`, EN-50160-Evaluator — und die
  P4-Vorabprüfung: existiert in den geplanten Ausbauszenarien überhaupt eine
  zulässige Rückfallaktion?

## Lizenz

MIT, siehe [`LICENSE`](LICENSE).

Für die Datensätze gilt das nicht: deren Bedingungen unterscheiden sich je
Quelle. Das Repository enthält daher keine Rohdaten, sondern Bezugsskripte und
Manifeste, siehe [`data/README.md`](data/README.md).

## Zitation

Siehe [`CITATION.cff`](CITATION.cff).
