# Systemarchitektur: RL-Trainingsumgebung für die autonome Regelung von Niederspannungsnetzen

**Version:** 0.5 — Zertifizierung als spätere Ausbaustufe; Erweiterbarkeits-Contract in §13
**Stack:** Python ≥ 3.11 · pandapower + simbench · Gymnasium · Stable-Baselines3 (+ sb3-contrib) · PyTorch · Hydra/OmegaConf · pandas/pyarrow · Optuna · MLflow oder TensorBoard

---

## 1. Zielsetzung und Designprinzipien

### 1.1 Regelungsaufgabe

Ein (oder mehrere) RL-Agenten regeln ein NS-Netz so, dass

- das Spannungsband nach dem **Perzentilkriterium der EN 50160** eingehalten wird — bewertet auf 10-min-Mittelwerten und Wochenintervallen, nicht je Zeitschritt (Umsetzung siehe §6.6),
- Leitungen und der Ortsnetztransformator nicht überlastet werden,

und zwar über die Stellgrößen

| Aktor | Stellgröße | Typische Freiheitsgrade |
|---|---|---|
| PV-Anlage | P-Abregelung, Q-Bereitstellung | `p_curtail ∈ [0,1]`, `q ∈ [-Q_max, Q_max]` |
| Batteriespeicher | Lade-/Entladeleistung, optional Q | `p ∈ [-P_max, P_max]`, SoC-Dynamik |
| Wärmepumpe | Sperren/Freigeben, Leistungsbegrenzung | binär, diskret (SG-Ready) oder kontinuierlich |
| EV-Ladepunkt | Ladeleistung je Ladevorgang | `p ∈ [0, P_max]` bzw. `{0, 4.2 kW, P_max}` |

Nebenbedingungen und Komfortziele (gelieferte Ladeenergie bis Abfahrt, Gebäudetemperatur, Speicherzustand) sind Teil der Zielfunktion, nicht Teil der Netzrestriktion — genau das macht die Aufgabe als Sequenzentscheidungsproblem interessant.

### 1.2 Designprinzipien

1. **Strikte Schichtentrennung.** Daten → Szenario → Netzmodell → Anlagenmodelle → Gym-Environment → Agent → Auswertung. Jede Schicht ist gegen eine Alternative austauschbar, ohne die darüberliegenden zu ändern.
2. **Konfiguration statt Code.** Jedes Experiment ist vollständig durch eine YAML-Konfiguration (Hydra) beschrieben. Kein Experiment wird durch Editieren von Quellcode definiert.
3. **Alles, was Zufall ist, ist geseedet und protokolliert.** Ein Run ist durch `(code_commit, config_hash, data_manifest_hash, seed)` eindeutig bestimmt.
4. **Referenzverfahren und RL-Policies teilen dasselbe Controller-Interface.** Dadurch läuft die Evaluation für regelbasierte Verfahren, OPF, MPC und RL über exakt denselben Pfad.
5. **Der Agent sieht nur, was ein realer Regler sehen könnte.** Beobachtbarkeit ist konfigurierbar (Vollzustand als Forschungs-Oberschranke vs. realistische Messstellen).
6. **Simulationsschrittweite und Regelungsschrittweite sind getrennt.** `sim_dt ∈ {1, 2, 5, 10} min`, `control_dt = k · sim_dt`. Die Einschränkung der zulässigen Werte folgt aus dem 10-min-Mittelungsintervall der EN 50160 (§6.6).
8. **Der Agent ist eine nicht vertrauenswürdige Komponente.** Die Netzintegrität wird nicht von der Policy garantiert, sondern von einem separaten, verifizierten Zertifizierer, den der Agent nicht beeinflussen, parametrieren oder umgehen kann (§6.9). Alles, was der Agent tut, ist Optimierung innerhalb einer nachweislich zulässigen Menge.
7. **Der Einzelagent ist ein Spezialfall, kein Sonderfall.** Die Env hält intern immer eine Struktur je Aktor; der zentrale Einzelagent ist eine Sicht über eine Partition, die alle Aktoren enthält. Multi-Agenten-Betrieb ist damit später eine Konfigurations- und keine Umbauaufgabe (§6.3).

---

## 2. Gesamtarchitektur

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L6  Experiment & Orchestrierung                                             │
│     Hydra-Configs · Run-Registry · Sweeps (Optuna) · Seed-Management        │
├──────────────────────────────────────────────────────────────────────────────┤
│ L5  Auswertung & Visualisierung                                             │
│     KPI-Engine · statistische Aggregation (IQM/Bootstrap) · Plots · Replay  │
├──────────────────────────────────────────────────────────────────────────────┤
│ L4  Agenten & Referenzverfahren        (gemeinsames Controller-Interface)   │
│     SB3-Factory · Policies/Feature-Extraktoren · Baselines · OPF · MPC      │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  Gymnasium-Environment                                                   │
│     ObsBuilder · ActionMapper · RewardComposer · EpisodeSampler · Wrappers  │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  Anlagen- & Flexibilitätsmodelle                                         │
│     PV · BESS · Wärmepumpe (+ therm. Speicher) · EVSE (+ Session-Modell)    │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  Netz & Physik                                                           │
│     simbench/pandapower-Loader · Asset-Mapping · Power-Flow · Netz-Metriken │
├──────────────────────────────────────────────────────────────────────────────┤
│ L0  Datenschicht                                                            │
│     Quellen-Adapter · Resampling · Validierung · Parquet-Cache · Manifest   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Repository-Layout

```
lvgrid-rl/
├── pyproject.toml            # uv/poetry, Lockfile eingecheckt
├── Dockerfile
├── configs/
│   ├── config.yaml           # Hydra-Root
│   ├── grid/                 # simbench_1-LV-rural1, semiurb4, urban6, custom_*
│   ├── scenario/             # Ausbaugrade PV/WP/EV/BESS, Wetterjahr
│   ├── data/                 # Quellenauswahl, sim_dt, Resampling-Policy
│   ├── env/                  # obs_space, action_space, reward, episode
│   ├── agent/                # ppo, sac, td3, tqc, recurrent_ppo, gnn_ppo, ...
│   ├── baseline/             # do_nothing, qu_droop, en14a, opf, mpc_oracle, ...
│   └── experiment/           # zusammengesetzte Studien
├── src/lvgrid_rl/
│   ├── data/
│   ├── grid/
│   ├── components/
│   ├── env/
│   ├── agents/
│   ├── baselines/
│   ├── experiment/
│   ├── eval/
│   └── viz/
├── scripts/                  # prepare_data.py, train.py, evaluate.py, report.py
├── tests/
└── results/                  # Run-Artefakte (nicht versioniert, DVC-tauglich)
```

---

## 3. L0 — Datenschicht

### 3.1 Aufgaben

1. **Beschaffung** (`data/sources/*.py`): ein Adapter je Datenquelle, einheitliche Rückgabe als „langes" DataFrame mit `(timestamp, series_id, quantity, value, unit)`.
2. **Validierung** (`data/schema.py`, pandera): Zeitstempel monoton, UTC + Zeitzonenkennung, Lücken quantifiziert, Einheiten geprüft, Plausibilitätsgrenzen.
3. **Lückenbehandlung** (`data/gapfill.py`): kurze Lücken (< 3 Schritte) linear, längere durch gleiche Tageszeit des vorangegangenen gleichen Wochentags; jede Ersetzung wird in einer Maske protokolliert und geht in die Datenqualitäts-KPIs ein.
4. **Resampling auf `sim_dt`** (`data/resample.py`) — der kritische Schritt, siehe 3.3.
5. **Normierung**: alle Profile werden als p.u.-Profile (bezogen auf Jahresmaximum oder Jahresenergie) gespeichert und erst im Szenario mit anlagenspezifischen Skalierungsfaktoren belegt.
6. **Cache & Manifest** (`data/cache.py`): Ergebnis wird als partitioniertes Parquet abgelegt, Schlüssel = Hash über `(quelle, version, zeitraum, sim_dt, resampling_policy, gapfill_policy)`. `manifest.json` enthält SHA-256 jeder Rohdatei plus Pipeline-Version.

### 3.2 Nutzbare Zeitreihen (Recherchestand)

**Primär / Default**

- **SimBench-Zeitreihen.** Konsistente Jahreszeitreihen für Last, Erzeugung und Speicher in **15-min-Auflösung für ein vollständiges Jahr**, direkt über `simbench.get_simbench_net()` bzw. die CSV-Tabellen `LoadProfile`, `RESProfile`, `PowerPlantProfile`, `StorageProfile` verfügbar. Der Datensatz enthält u. a. Haushalts- und Gewerbeprofile (an SLP-Klassen angelehnt), acht PV-Profile, Speicherprofile sowie **fünf Wärmepumpen-Profile (`WP-1` … `WP-5`)** mit modellierten bivalenten Betriebsweisen. Vorteil: Netz und Zeitreihen sind aufeinander abgestimmt und über `net["profiles"]` bereits verknüpft. Nachteil: 15 min ist die feinste Auflösung, und **EV-Ladeprofile fehlen**.
  https://simbench.de/en/download/datasets/ · https://simbench.readthedocs.io

**Haushaltslast hochaufgelöst**

- **HTW Berlin, „Repräsentative elektrische Lastprofile für Wohngebäude"** — 74 Einfamilienhaushalte, Jahresprofil auf **1-Sekunden-Basis** (synthetisiert aus 15-min-Smart-Meter-Messungen des IZES-Datensatzes und 1-s-Messungen aus ADRES-Concept), inkl. phasenaufgelöster Wirk- und Blindleistung. Ideal, um bei `sim_dt = 1 min` realistische Gleichzeitigkeits- und Spitzeneffekte zu erhalten.
  https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/
- **WPuQ (ISFH Hameln)** — 38 Einfamilienhäuser in Niedersachsen, 2018–2020, **Originalauflösung 10 s**, bereitgestellt in 10 s / 1 min / 15 min / 60 min. Enthält getrennt **Haushaltslast und Wärmepumpenlast je Gebäude**, zusätzlich die Ortsnetzstation mit der Aggregation von 68 Haushalten sowie Wetterdaten. Für dieses Projekt der wertvollste Datensatz, weil Haushalt und WP separat vorliegen und ein realer Aggregationsmaßstab mitgeliefert wird.
  DOI 10.5281/zenodo.5642902 · Paper: Sci Data 9, 56 (2022) · Code: https://github.com/ISFH/WPuQ
- **OPSD Household Data (CoSSMic, Konstanz)** — 11 Haushalte in Süddeutschland, Messungen zunächst in 3-min-, später in **1-min-Intervallen**, zusätzlich als 15- und 60-min-Aggregat veröffentlicht. Enthält je Haushalt Teilmessungen inkl. PV, teilweise Wärmepumpe und EV-Ladung.
  https://data.open-power-system-data.org/household_data/

**Wärmepumpen**

- **WPuQ** (siehe oben) — Messwerte der WP-Last, 10 s.
- **HEAPO** — 1.408 reale Haushalte mit Wärmepumpe im Kanton Zürich, Smart-Meter-Daten in **15-min- und Tagesauflösung**, 2018-11 bis 2024-03, mit Haushalts-Metadaten, Wetterdaten von 8 Stationen und 410 Vor-Ort-Protokollen. Gut für die Streuung über viele Anlagen.
- **when2heat (OPSD)** — synthetische Zeitreihen für Wärmebedarf und **COP** (Luft/Erdreich/Grundwasser × Fußboden/Radiator/Warmwasser) für 28 europäische Länder, **stündlich**. Nicht als Lastprofil geeignet, aber als Quelle für COP-Kennlinien und Jahresgang des Wärmebedarfs, der dann über Innentagesprofile auf `sim_dt` disaggregiert wird.
- **SimBench `WP-1..WP-5`** — 15 min, konsistent zum Netzdatensatz.

**Elektromobilität**

Hier ist entscheidend, dass der Agent *Flexibilität* braucht, nicht nur ein Lastprofil: pro Ladevorgang Ankunftszeit, Abfahrtszeit, Energiebedarf, maximale Ladeleistung.

- **emobpy (DIW Berlin)** — Python-Werkzeug, das aus der Erhebung *Mobilität in Deutschland* vier Zeitreihen je Fahrzeug generiert: Mobilität/Standort, Fahrstromverbrauch, **Netzverfügbarkeit** und Netzladebedarf, standardmäßig in **15-min-Schritten** (Schrittweite konfigurierbar). Das ist die sauberste Quelle für die hier benötigte Flexibilitätsbeschreibung.
  https://emobpy.readthedocs.io · Sci Data 8, 152 (2021)
- **ElaadNL Open Data** — normierte Profile und Verteilungen für **Ankunftszeit, Anschlussdauer, Energiebedarf und Ladekurven**, getrennt für privat, Arbeitsplatz und öffentlich, abgeleitet aus großen Mengen realer Ladevorgänge; zusätzlich ein offener Datensatz mit 10.000 zufälligen Ladevorgängen. Gut zur Kalibrierung bzw. Validierung des Session-Generators.
  https://platform.elaad.io / https://elaad.nl/en/open-datasets/
- **ACN-Data (Caltech/JPL)** — Ladevorgänge auf Session-Ebene, Arbeitsplatz-Kontext, US; als zweites, unabhängiges Verhaltensmuster für Generalisierungstests.

**PV und Wetter**

- **DWD Open Data / CDC** — Stationsmessungen der Globalstrahlung und weiterer Größen in **10-min-Auflösung** (Produkt `OBS_DEU_PT10M_RAD-G`), zusätzlich Lufttemperatur, Feuchte, Wind in 10-min. Über `wetterdienst` oder direkt vom Open-Data-Server beziehbar. 10 min ist für `sim_dt ≥ 10 min` direkt nutzbar; für 1 min ist Interpolation nötig und sollte als Limitation dokumentiert werden.
  https://opendata.dwd.de/climate_environment/CDC/
- **PVGIS / CAMS Radiation Service** — satellitenbasierte Einstrahlung, u. a. mit 15-min- bzw. 1-min-Produkten; über `pvlib.iotools` anbindbar.
- **PV-Modellierung** mit `pvlib` (Transposition, Temperaturmodell, Wechselrichter) statt fertiger Erzeugungsprofile — erlaubt es, Ausrichtung und Verschattung je Anlage zu variieren und damit Szenarienvielfalt zu erzeugen.

**Referenz / Fallback**

- **BDEW-Standardlastprofile** über `demandlib` (H0/H0-dyn, G0…), 15 min — als Referenzfall und für Sensitivitätsstudien „SLP vs. Messdaten".
- **Marktstammdatenregister (MaStR)** — reale Anlagen- und Leistungsverteilungen zur Parametrierung der Ausbau-Szenarien (nicht Zeitreihen).

### 3.3 Resampling auf eine einheitliche Schrittweite

Zielschrittweite `sim_dt ∈ {1, 2, 5, 10} min`, Default 5 min. **15 min ist bewusst nicht zulässig:** das EN-50160-Kriterium bewertet 10-min-Mittelwerte, und 15 min teilt 10 min nicht — ein PQ-Fenster wäre nicht ohne Interpolation aus Simulationsschritten bildbar (§6.6). Pro Signaltyp wird eine eigene Aggregationsregel hinterlegt:

| Signaltyp | Beim Downsampling | Beim Upsampling |
|---|---|---|
| Leistung (Last, Erzeugung) | Mittelwert (energieerhaltend) | stückweise konstant, alternativ energieerhaltende Interpolation |
| Energiezählerstand | Differenz über Intervall | monotone Interpolation, dann Differenz |
| Globalstrahlung | Mittelwert | Clear-Sky-Index-Interpolation (nicht linear auf GHI!) |
| Temperatur | Mittelwert | linear |
| Binäre Verfügbarkeit (EV angesteckt) | Mehrheitsentscheid + Mindestdauer | stückweise konstant |
| EV-Ladevorgänge | **nicht resampeln** — Events bleiben Events, Ankunft/Abfahrt wird auf das Zeitraster gerundet, Energiebedarf bleibt erhalten | – |
| SoC/Zustände | Endwert des Intervalls | linear |

Zwei Punkte, die früh festgelegt und dokumentiert werden sollten:

- **Mittelung glättet Spitzen.** Eine 10-min-Mittelung reduziert die für Überlastung relevanten Extrema deutlich; für das Spannungskriterium ist die Mittelung dagegen normativ *gewollt*, weil EN 50160 selbst auf 10-min-Mittelwerten definiert ist. Empfehlung: Training bei `sim_dt = 5 min` (Rechenzeit, 2 Schritte je PQ-Fenster), **finale Evaluation zusätzlich bei `sim_dt = 1 min`** mit identischer Policy. Die thermischen Grenzwerte werden bei 1 min spürbar strenger, das Spannungskriterium kaum — dieser Unterschied ist ein Ergebnis und sollte berichtet werden.
- **Zeitzonen und Sommerzeit.** Intern durchgehend UTC, Kalenderfeatures (Tageszeit, Wochentag) aus lokaler Zeit abgeleitet. Schaltsekunden/DST-Artefakte in Rohdaten werden beim Import bereinigt und im Manifest vermerkt.

### 3.4 Szenario-Builder

`data/scenario_builder.py` erzeugt aus Netz + Datenkatalog ein reproduzierbares **Szenario**:

- Durchdringungsgrade: Anteil Haushalte mit PV / WP / EV / BESS, Leistungsklassen, Verteilung über die Stränge.
- Zuordnung: welche Zeitreihe (`series_id`) gehört zu welchem `pandapower`-Element, mit welchem Skalierungsfaktor.
- Ziehung erfolgt über einen dedizierten `numpy.random.Generator` mit eigenem `scenario_seed` → identische Szenarien über alle Agenten hinweg vergleichbar.
- Ausgabe: `ScenarioSpec` (JSON, versioniert) + vorgerechnete Profilmatrix als Parquet. **Wichtig:** Der Szenario-Seed ist vom Trainings-Seed getrennt, damit „5 Seeds" nicht versehentlich „5 verschiedene Netze" bedeutet.

---

## 4. L1 — Netz und Physik

```
grid/
├── loader.py        # simbench-Netze, eigene Netze, Netzvarianten
├── asset_mapping.py # ScenarioSpec -> pandapower-Elemente, Index-Caches
├── powerflow.py     # PowerFlowEngine (Wrapper mit Warmstart & Fehlerbehandlung)
└── metrics.py       # GridState -> Verletzungsmetriken
```

**PowerFlowEngine.** Eine dünne Fassade über `pp.runpp`, die (a) numba aktiviert, (b) Warmstart über `init="results"` nutzt, (c) Nicht-Konvergenz abfängt und als definierten Zustand behandelt (Reward-Penalty + `info["pf_diverged"]`, nicht Crash), (d) optional gegen einen alternativen Backend-Solver getauscht werden kann und (e) **hypothetisch** aufgerufen werden kann: `run(net_snapshot, setpoints) -> GridState` ohne Seiteneffekt auf das laufende Netz. Punkt (e) wird zunächst nur vom Schatten-Lastfluss (§6.8) genutzt, ist aber die Voraussetzung dafür, dass später ein prädiktiver Sicherheitsfilter Rückfalltrajektorien durchrechnen kann (§13, I5). Der Power Flow ist der Laufzeit-Engpass: Ein LV-Netz mit ~100 Knoten braucht wenige Millisekunden, aber bei 10⁶–10⁷ Schritten Training summiert sich das. Deshalb:

- Index-Arrays (Bus-, Line-, Trafo-Indizes) einmal beim Reset auflösen, nicht per Schritt über pandas-Lookups.
- Direkte Schreibzugriffe auf `net.load["p_mw"].values` statt `ConstControl`-Objekte im RL-Loop (pandapowers `control`/`timeseries`-Module sind für Studien gedacht, nicht für Millionen Steps).
- `SubprocVecEnv` mit 8–16 Workern, `OMP_NUM_THREADS=1` je Worker.
- Optionaler **Surrogat-Modus**: ein trainiertes neuronales Netz oder eine linearisierte Sensitivitätsmatrix (∂U/∂P, ∂U/∂Q) als schneller Ersatz für Vortraining, mit anschließendem Finetuning auf dem exakten AC-Load-Flow. Als separate `PowerFlowEngine`-Implementierung, damit dieselbe Env beides nutzen kann.

**GridState.** Ein schlanker Dataclass/NamedTuple mit `vm_pu`, `line_loading_percent`, `trafo_loading_percent`, `p_slack`, `losses`. Keine DataFrames im Hot Path.

**metrics.py** liefert pro Schritt die momentanen Größen:
`max_overvoltage`, `max_undervoltage`, `sum_voltage_violation_pu`, `n_line_overloads`, `max_line_loading`, `trafo_loading`, `overload_integral_MVA·h`, `losses_kWh`.

**pq.py — `PQAggregator`.** Die Spannungsbewertung erfolgt *nicht* auf diesen Momentanwerten, sondern auf 10-min-Mittelwerten. Der Aggregator hält je bewertetem Bus einen Ringpuffer über das laufende Fenster, bildet am Fensterende den Mittelwert und führt den Wochen-Budgetzähler. Details und Begründung in §6.6. Bewertet werden nur Busse mit `bus_role == "connection_point"` (Netzanschlusspunkte) — EN 50160 gilt am Übergabepunkt, nicht an Kabelverteilern oder Muffen.

**Ableitung der Anschlusspunkt-Menge** (`grid/connection_points.py`): Ein Bus ist Anschlusspunkt, wenn **mindestens ein kundenseitiges Element** daran hängt — also die Vereinigung über

```python
CP_ELEMENTS = ("load", "sgen", "gen", "storage", "asymmetric_load", "asymmetric_sgen")
```

Die Erzeuger sind dabei nicht optional: ein PV-Einspeiser oder ein Batteriespeicher ohne Last am selben Bus ist ebenso ein Übergabepunkt wie ein Hausanschluss, und gerade solche Knoten liegen bei Überspannungsproblemen häufig am Strangende, also genau dort, wo das Kriterium bindend wird. Eine reine `load`-Betrachtung würde die kritischsten Knoten systematisch ausblenden.

Zwei Feinheiten:

- Nicht jedes `sgen`-Element ist ein Kunde. Aggregierte Ersatzeinspeiser oder Modellierungshilfen (z. B. ein `sgen` zur Nachbildung eines unterlagerten Netzausschnitts) müssen ausgeschlossen werden. Umsetzung über eine Ausnahmeliste in `configs/grid/*`, nicht über Heuristik auf Elementnamen.
- `in_service == False` und Elemente mit `p_mw == 0` über das ganze Jahr werden ausgeschlossen, sonst verwässern nie belastete Knoten die `pass_rate`.

Die resultierende Liste wird je Netz einmal manuell geprüft und dann in der Grid-Config fixiert (mit Hash, damit eine stille Änderung die KPI nicht unbemerkt verschiebt). Zusätzlich wird in der Auswertung immer die konservative Variante „alle Busse" mitberechnet — die Differenz beider Zahlen ist die Sensitivität gegenüber dieser Modellierungsentscheidung und gehört in den Ergebnisbericht.

---

## 5. L2 — Anlagen- und Flexibilitätsmodelle

Alle Aktoren implementieren ein gemeinsames Protokoll. Das ist der Schlüssel dafür, dass Netz-, Aktor- und Agentenkonfiguration unabhängig variiert werden können.

```python
class FlexAsset(Protocol):
    asset_id: str
    bus: int
    ratings: AssetRatings  # S_max, P_max, Sicherungsnennstrom, vereinbarte Leistung

    def action_spec(self) -> ActionSpec: ...  # Box/Discrete in PHYSIKALISCHEN Einheiten
    def obs_spec(self) -> ObsSpec: ...
    def initial_state(self, rng: Generator, window: ScenarioWindow) -> AssetState: ...

    # --- reine Funktionen, kein self-Zustand ------------------------------
    def to_setpoint(
        self, s: AssetState, action: np.ndarray, x: ExogenousInput
    ) -> Setpoint:
        # Aktion -> (p_mw, q_mvar). Affin und invertierbar; jede Begrenzung
        # wird in Setpoint.clipping_info gemeldet, nicht still vorgenommen.
        ...

    def dynamics(
        self, s: AssetState, sp: Setpoint, x: ExogenousInput, g: GridState
    ) -> tuple[AssetState, AssetOutcome]:
        # Zustandsfortschreibung als reine Funktion: gleiche Eingabe -> gleiche
        # Ausgabe, kein Seiteneffekt. Erlaubt hypothetisches Vorausrechnen.
        ...
```

Drei Festlegungen im Protokoll, die zunächst wie Formalismus aussehen, aber die spätere Zertifizierung offenhalten (Begründung je Punkt in §13):

- **`dynamics` ist eine reine Funktion**, `AssetState` ein kopierbares Wertobjekt. Ein prädiktiver Sicherheitsfilter muss Anlagenzustände über einen Horizont hypothetisch fortschreiben; mit zustandsmutierenden Methoden ist das nicht möglich, ohne die Modelle neu zu schreiben.
- **Aktionen sind physikalische Leistungsgrößen**, nicht normierte Anteile oder Zielwerte. Ein SoC-Zielwert oder „Anteil der Restenergie" erzeugt eine nicht-boxförmige zulässige Menge in Aktionskoordinaten; die zertifizierte Zulässigkeitsmenge ist dagegen in Einspeisungen formuliert. Die Normierung für den Agenten passiert im `ActionMapper` und ist dort affin und invertierbar.
- **`ratings` wird von Anfang an geführt.** Sicherungsnennstrom, vereinbarte Anschlussleistung und `S_max` sind die Grundlage der deterministischen Unsicherheitsmengen (§6.9, P2). Sie jetzt mitzuschreiben kostet nichts; sie später für alle Szenarien nachzutragen ist Handarbeit.

`AssetOutcome` transportiert die für den Reward benötigten Größen (abgeregelte Energie, nicht gelieferte Ladeenergie, Komfortabweichung in K·h, Speicherdurchsatz, Schalthandlungen).

**PV (`components/pv.py`).** Potenzialprofil aus Daten; Aktion = Abregelungsfaktor und/oder Blindleistung innerhalb des Wechselrichter-Apparatediagramms (`S_max`-Kreis, `cosφ`-Grenzen). Kuratierte Betriebsarten: `p_only`, `q_only`, `pq`.

**BESS (`components/bess.py`).** SoC-Integration mit getrennten Lade-/Entladewirkungsgraden, `SoC ∈ [soc_min, soc_max]`, C-Rate-Grenzen, optional Selbstentladung und ein einfaches Degradationsmodell (Durchsatz- oder Zyklen-basiert) als Kostenterm. Aktion wird auf den physikalisch zulässigen Bereich projiziert — die Projektion wird protokolliert (`info["action_clipped"]`), damit man erkennt, ob die Policy systematisch Unzulässiges vorschlägt.

**Wärmepumpe (`components/heatpump.py`).** Nicht als reines Lastprofil, sondern als **verschiebbare Last mit Zustand**: Wärmebedarf aus Daten, COP(T_außen, T_vorlauf) aus when2heat-Kennlinien, und ein Speicher — je nach gewünschter Modelltiefe

- **Stufe 1 — festgelegt für M4: Pufferspeicher als Energiereservoir** mit Verlusten, zulässigem Temperaturband und COP(T_außen, T_vorlauf) aus when2heat-Kennlinien,
- Stufe 2: 1R1C-/2R2C-Gebäudemodell (thermische Masse) mit Komfortband für die Raumtemperatur,
- Stufe 3: SG-Ready-Stufen (Sperre, Normal, Einschaltempfehlung, Anlaufbefehl) als diskreter Aktionsraum,

plus Mindestlauf- und Mindeststillstandszeiten und eine maximale Sperrdauer. Komfortverletzung = Integral der Bandunterschreitung.

**EV-Ladepunkt (`components/evse.py`).** Session-basiert: der Szenario-Builder erzeugt je Ladepunkt eine Sequenz `(t_arrival, t_departure, E_demand, P_max, soc_arrival)`. Zwischen Ankunft und Abfahrt ist die Ladeleistung steuerbar; bei Abfahrt wird die nicht gelieferte Energie als Penalty verbucht. Optional V2G als Erweiterung (negative Leistung), zunächst deaktiviert.

---

## 6. L3 — Gymnasium-Environment

```
env/
├── lv_grid_env.py   # LVGridEnv(gymnasium.Env)
├── obs.py           # ObservationBuilder
├── actions.py       # ActionMapper (flach / je Asset / hierarchisch)
├── reward.py         # RewardComposer + Terme
├── episodes.py      # EpisodeSampler (Zeitfenster-Ziehung)
└── wrappers.py      # Normalisierung, Action-Masking, Safety-Fallback, Logging
```

### 6.1 Ablauf eines Schritts

```
step(a):
  1. ActionMapper: a (Agenten-Raum) -> Setpoints je Asset
  2. für jedes Asset: apply()  -> p/q ins net schreiben
  3. für die k Simulationsschritte innerhalb eines control_dt:
       exogene Profile bei t setzen (unsteuerbare Last, PV-Potenzial, T_außen)
       PowerFlowEngine.run(net) -> GridState
       metrics.evaluate(GridState)
       für jedes Asset: advance()
       Teil-Rewards akkumulieren
  4. RewardComposer: gewichtete Summe, Einzelterme in info[]
  5. ObservationBuilder: Beobachtung für t+1
  6. terminated / truncated bestimmen
```

Die Trennung `control_dt = k · sim_dt` erlaubt es, Agenten mit 15-min-Takt auf einer 1-min-Physik zu testen, ohne die Env zu ändern.

**Informationsordnung.** Schritt 1 und 2 dürfen ausschließlich auf Größen zugreifen, die zum Entscheidungszeitpunkt bekannt sind — also auf den Zustand bei `t` und auf Prognosen, nicht auf die Realisierung der exogenen Einspeisungen während `[t, t+control_dt)`. Diese Trennung wird durch ein explizites `InformationSet(t)` erzwungen, das `ActionMapper` und Policy übergeben bekommen, statt ihnen den vollen Szenario-Zugriff zu geben. Der Grund ist nicht Pedanterie: eine implizite Hellsichtigkeit ist im Training unsichtbar, macht aber jede spätere Sicherheitsaussage wertlos, weil ein realer Regler diese Information nicht hat. Ein solcher Fehler ist diffus über den Code verteilt und praktisch nicht nachträglich zu beseitigen.

### 6.2 Observation

Der `ObservationBuilder` setzt die Beobachtung aus deklarativ konfigurierten **Feature-Gruppen** zusammen:

| Gruppe | Inhalt |
|---|---|
| `time` | sin/cos von Tageszeit und Jahreszeit, Wochentag-Flag |
| `measurements` | `vm_pu` an ausgewählten Messknoten, Trafo-Auslastung, Strangströme — Auswahl über `sensor_config` |
| `asset_state` | SoC je BESS, thermischer Zustand je WP, angesteckt/Restenergie/Restzeit je EVSE |
| `local_power` | P/Q der eigenen Anlage und des Hausanschlusses |
| `pq_budget` | verbrauchter Anteil des zulässigen 10-min-Fenster-Budgets, Restdauer des Wochenintervalls, aktuelle Teil-Mittelung im offenen Fenster — **zwingend erforderlich**, siehe §6.6 |
| `forecast` | PV-Potenzial, Last, Außentemperatur und EV-Abfahrten der nächsten H Schritte; Fehlermodell nach §6.7 |
| `history` | Ringpuffer der letzten L Schritte (alternativ über `RecurrentPPO`) |

`sensor_config` ist ein wichtiger Forschungsschalter: `full_state` (alle Knotenspannungen — unrealistisch, aber als Oberschranke) vs. `realistic` (Ortsnetzstation + Strangendmessungen + lokale Hausanschlusswerte). Beobachtungsnormierung über `VecNormalize` oder feste physikalische Skalen; feste Skalen sind für Reproduzierbarkeit und für den Vergleich mit Baselines vorzuziehen.

### 6.3 Aktionsraum

Drei über Konfiguration wählbare Modi, alle über denselben `ActionMapper`:

1. **Flach zentral** — ein `Box`-Vektor über alle Aktoren. Einfachster Einstieg, skaliert schlecht.
2. **Parametrisiert je Assettyp** — der Agent gibt pro Assettyp einen Parametervektor aus, der mit typspezifischem Encoder auf alle Instanzen angewandt wird (Parameter-Sharing → Invarianz gegenüber Anzahl der Anlagen).
3. **Multi-Agent** — je Aktor oder je Strang ein Agent, Env stellt zusätzlich ein PettingZoo-kompatibles Interface bereit; mit SB3 zunächst als „independent learners".

**Festlegung:** Start mit Modus 1 (M3), Wechsel auf Modus 2 ab M4; Modus 3 bleibt offen. Damit der Multi-Agenten-Fall später nicht nachgebaut werden muss, gelten ab M3 vier Regeln:

1. Die Env hält intern **immer** einen `AssetGraph` mit stabiler Aktor-Ordnung und einer `agent_partition`-Zuordnung. Der Einzelagent ist ein `FlatView` über eine Partition, die alle Aktoren enthält — es gibt keinen zweiten Codepfad.
2. Beobachtungs-, Aktions- und Rewardbausteine werden **je Aktor bzw. je Partition** berechnet und erst am Ende aggregiert. Keine Logik, die einen flachen Vektor fester Länge voraussetzt (das betrifft insbesondere Reward-Terme, die über Aktoren summieren).
3. Keine gelernte globale Beobachtungsnormierung über den flachen Vektor (`VecNormalize` auf Observations), sondern feste physikalische Skalen je Feature. Das ist für Multi-Agent notwendig und für die Reproduzierbarkeit ohnehin besser.
4. Ein `PettingZooAdapter` (ParallelEnv) existiert ab M3, auch unbenutzt, zusammen mit einem **Regressionstest**: Einzelagenten-Env und Multi-Agenten-Env mit einer einzigen Partition müssen bei identischen Aktionen bitidentische Trajektorien liefern. Dieser Test ist der einzige zuverlässige Schutz davor, dass die beiden Pfade über Monate auseinanderlaufen.

Der **Safety-Layer** ist nicht ein Gym-Wrapper, sondern eine Komponente *innerhalb* der Env, weil verschiedene Safe-RL-Ansätze an unterschiedlichen Stellen der Pipeline angreifen und weil ein Wrapper nur die Beobachtung sieht — ein Zertifizierer braucht den vollen Systemzustand. Die Architektur hält deshalb ab M3 zwei Eingriffspunkte offen, beide mit einer Null-Implementierung als Default:

- `env.safety: SafetyComponent` — wird in Schritt 2 des Ablaufs (§6.1) aufgerufen, vor dem Schreiben der Setpoints, und erhält `SystemState`;
- `info["action_mask"]` — wird von der Env immer bereitgestellt (bei `none`: alles zulässig), sodass eine maskierende Policy außerhalb der Env sie nutzen kann, ohne dass die Env geändert werden muss.

Taxonomie und Vergleich der heuristischen Mechanismen in §6.8, Zielarchitektur mit harter Garantie in §6.9, Invarianten in §13.

### 6.4 Reward

Terme werden zwei Kategorien zugeordnet: **`objective`** (zu minimierende Kosten) und **`constraint`** (einzuhaltende Grenzen). Diese Trennung ist die Voraussetzung dafür, dass feste Gewichtung und Lagrange-Variante aus derselben Konfiguration laufen.

```yaml
reward:
  mode: fixed_weights           # fixed_weights | lagrangian
  objective:
    pv_curtailment:    {weight: -1.0,  unit: kWh}
    ev_unserved:       {weight: -5.0,  unit: kWh, at: departure}
    hp_comfort:        {weight: -2.0,  unit: Kh}
    bess_degradation:  {weight: -0.2,  unit: kWh_throughput}
    action_smoothness: {weight: -0.05}
    grid_losses:       {weight: -0.1,  unit: kWh}
  constraint:
    en50160_k95:       {weight: -10.0, limit: 0.05, shaping: potential}
    en50160_k100:      {weight: -50.0, limit: 0.0}
    thermal_overload:  {weight: -10.0, limit: 0.0, form: hinge, limit_pct: 100}
  normalization: fixed_scale
```

**Modus `fixed_weights`** (Startkonfiguration): alle Terme werden skalarisiert addiert, `limit` dient nur der Definition des Straffreibereichs.

**Modus `lagrangian`** (vorgesehene Alternative): nur `objective`-Terme bilden den Reward, jeder `constraint`-Term wird zusätzlich als Kostensignal in `info["cost/<name>"]` geführt. Ein `LagrangianCallback` schätzt episodenweise das erwartete Kostenmaß `J_c` und aktualisiert die Multiplikatoren per dualem Gradientenaufstieg `λ ← [λ + η·(J_c − d)]₊`; ein `LagrangianRewardWrapper` bildet `r_eff = r_obj − Σ λ_i·c_i`. Der praktische Vorteil in diesem Projekt: die Grenzwerte `d` stehen in physikalischen bzw. normativen Einheiten. Für EN 50160 ist `d = 0.05` **exakt das Normkriterium** — man muss kein Strafgewicht raten, sondern nennt die einzuhaltende Grenze. Für dieses Problem passt die Constrained-Formulierung damit ungewöhnlich gut, weshalb sie nicht nur als Fußnote, sondern als vollwertiger zweiter Modus vorgesehen ist.

Zwei Fallstricke, die man vor der Implementierung kennen sollte:

- Ein sich änderndes λ macht den Reward **nicht-stationär**. Bei Off-Policy-Verfahren enthält der Replay-Buffer Rewards, die mit veraltetem λ berechnet wurden. Sauberer Weg: Kosten getrennt im Buffer ablegen und `r_eff` beim Sampling neu berechnen (angepasster `ReplayBuffer`). Pragmatischer Weg: λ deutlich langsamer aktualisieren als die Policy lernt. Empfehlung: die Lagrange-Variante zuerst mit PPO validieren, dann auf SAC/TQC übertragen.
- Das Dualproblem oszilliert leicht. λ braucht eine Obergrenze, `J_c` einen EMA-geglätteten Schätzer, und η sollte deutlich kleiner sein als die Policy-Lernrate.

Jeder Term wird **separat in `info["reward/<term>"]`** zurückgegeben. Ohne diese Dekomposition lässt sich später nicht mehr rekonstruieren, welcher Term das Lernverhalten dominiert hat. Unabhängig vom Reward werden die physikalischen KPIs ungewichtet erfasst: **der Vergleich mit Referenzverfahren erfolgt ausschließlich auf KPIs, niemals auf dem Reward** — sonst gewinnt trivial die Policy, deren Zielfunktion man selbst definiert hat.

### 6.5 Episoden

`EpisodeSampler` zieht Zeitfenster aus dem Datenjahr:

- **Zeitliche Splits** — `train`: z. B. Jan–Sep, `val`: Okt, `test`: Nov–Dez plus kuratierte Extremwochen (höchste PV-Einspeisung, kältester Zeitraum mit WP-Spitze, Wochenende mit hoher EV-Gleichzeitigkeit). Splits werden fest in der Konfiguration verankert. Wichtig: die Splits liegen auf **Wochengrenzen**, weil das Bewertungsintervall eine Woche ist.
- **Episodenlänge und Budget-Randomisierung** — hier entsteht eine Spannung: das Kriterium bezieht sich auf eine ganze Woche, Episoden über eine Woche sind aber lang und teuer. Gewählte Lösung: Episoden von 2–7 Tagen mit **randomisiertem Anfangszustand des Budgets** — beim Reset werden verbrauchtes Fensterbudget und Restdauer der laufenden Woche zufällig gezogen. Der Agent sieht dadurch alle Budget-Regime (viel Luft, knapp, schon überschritten) ohne Wochen-Episoden. Der Evaluationssatz enthält dagegen **vollständige Kalenderwochen mit Budget-Startwert null**, damit die berichtete KPI normkonform ist.
- **Diskontierung ist keine freie Wahl.** Der effektive Horizont `1/(1−γ)` muss den Kriterienhorizont abdecken. Bei `control_dt = 15 min` entspricht eine Woche 672 Entscheidungen → `γ ≈ 0.997`; bei `control_dt = 5 min` sind es 2016 → `γ ≈ 0.999`. Die Konfigurationsvalidierung leitet ein konsistentes `γ` aus `control_dt` und Episodenlänge ab und warnt bei Abweichung. Ein aus Gewohnheit gesetztes `γ = 0.99` würde den Wochenhorizont schlicht nicht sehen.
- **Curriculum** (optional) — Start mit moderatem Ausbaugrad, Steigerung über den Trainingsverlauf; als eigene Sampler-Implementierung, damit abschaltbar.
- **Fester Evaluationssatz** — eine unveränderliche Liste von (Startwoche, Szenario-Seed)-Paaren, gegen die *alle* Agenten und *alle* Baselines evaluiert werden.
- `truncated = True` am Fensterende; `terminated` nur bei irreparablem Zustand (Power-Flow-Divergenz).

### 6.6 Umsetzung des EN-50160-Perzentilkriteriums

Das ist die Entscheidung mit den weitreichendsten Folgen, deshalb hier zusammengefasst statt über die Schichten verstreut.

**Das Kriterium.** Zwei verschachtelte Bedingungen, jeweils auf **10-min-Mittelwerten des Effektivwerts** über ein **Wochenintervall**:

- **K95** — 95 % der 10-min-Mittelwerte jeder Woche innerhalb ±10 % U_n
- **K100** — alle 10-min-Mittelwerte innerhalb +10 % / −15 % U_n (Niederspannung)

Bei 1008 Zehn-Minuten-Fenstern je Woche erlaubt K95 also **bis zu 50 Fenster außerhalb ±10 %** pro Bus und Woche. Bewertungsort ist der Netzanschlusspunkt.

**Konsequenz 1 — Schrittweite.** 15 min teilt 10 min nicht, daher `sim_dt ∈ {1, 2, 5, 10} min` (§3.3). `control_dt` bleibt ein Vielfaches von `sim_dt` und darf gegenüber dem PQ-Raster versetzt sein — ein 15-min-Regeltakt auf 5-min-Physik ist zulässig und realistisch, die Fensterausrichtung wird aber protokolliert, damit Artefakte erkennbar bleiben.

**Konsequenz 2 — der Reward ist nicht mehr schrittweise definierbar.** Ob eine Überschreitung „zählt", hängt von der Verteilung der gesamten Woche ab. Das Problem ist ohne Zusatzzustand nicht Markov'sch und das Kostensignal ist terminal. Drei Bausteine lösen das:

1. **`PQAggregator`** bildet die 10-min-Mittelwerte. Kriterien-relevante Ereignisse entstehen nur an Fensterenden, nicht in jedem Schritt.
2. **Budget-Zustand in der Beobachtung.** Verbrauchte Fenster, Restdauer der Woche und der laufende Teilmittelwert gehen als Feature-Gruppe `pq_budget` in die Observation ein. Ohne diese Größen kann eine optimale Policy nicht existieren — der Agent könnte nicht unterscheiden, ob eine Überschreitung noch im Budget liegt oder es sprengt. Bei vielen Bussen wird verdichtet: Maximum der Budget-Ausnutzung, Anzahl Busse über 80 % Budget, Anzahl bereits verletzender Busse.
3. **Potentialbasiertes Shaping.** Damit Credit Assignment über ~672–2016 Schritte funktioniert, wird die terminale Kostenfunktion durch ein Potential `Φ(verbrauchtes Budget, Überschreitungstiefe, Restzeit)` ergänzt: `r_shaped = r + γ·Φ(s′) − Φ(s)`. Potentialbasiertes Shaping (Ng et al.) verändert die optimale Policy nicht, liefert aber ein dichtes Signal. Zusätzlich ein klein gewichteter direkter Tiefenterm über `|ΔU|` mit Totband als Anlaufhilfe, der in einer Ablation abgeschaltet werden kann — er ist streng genommen kriterienfremd und könnte den Agenten konservativer machen als die Norm verlangt.

K100 wird als harte Nebenbedingung mit hohem Gewicht je Ereignis behandelt.

**Konsequenz 3 — Episoden und γ:** siehe §6.5.

**Konsequenz 4 — die Referenzverfahren ändern ihre mathematische Form.** B7–B9 optimieren nicht mehr gegen eine Box-Restriktion, sondern gegen eine **Kardinalitätsbedingung** („höchstens 50 Fenster außerhalb je Bus und Woche"). Das ist gemischt-ganzzahlig: eine Binärvariable je Fenster und Bus plus Kardinalitätsrestriktion. Drei praktikable Wege:

- (a) **MILP mit Big-M** auf dem linearisierten Netzmodell — für einen Wochenhorizont und wenige Dutzend bewertete Busse lösbar, aber nicht billig;
- (b) **CVaR-Relaxation** als konvexe Approximation (beschränkt den Mittelwert des schlechtesten 5 %-Quantils statt seiner Anzahl) — konservativ, schnell;
- (c) **sequenzielle Lockerung**: Box-Restriktion iterativ aufweichen, bis das Budget ausgeschöpft ist.

Empfehlung: (b) für M5, (a) für die Endauswertung von `mpc_oracle`. Entscheidend ist nur, dass die Referenzverfahren gegen **dasselbe** Kriterium optimieren, gegen das anschließend bewertet wird.

**Konsequenz 5 — Baselines müssen mitgetunt werden.** Die Totbänder und Steigungen von B3/B4 und die Schwellwerte von B5/B6 sind implizit für ein hartes ±10 %-Kriterium ausgelegt. Unter K95 ist die optimale Parametrierung eine andere, weil Toleranz zulässig ist. Jede Baseline erhält daher eine eigene, dokumentierte Parametersuche gegen dieselbe Ziel-KPI auf dem Validierungszeitraum. Ohne das gewinnt der RL-Agent gegen eine absichtlich schlecht eingestellte Referenz — der häufigste methodische Fehler in der RL-Energieliteratur, und derjenige, der Ergebnisse am schnellsten angreifbar macht.

### 6.7 Prognosefehlermodell

`env/forecast.py` mit vier Modi:

| Modus | Beschreibung |
|---|---|
| `perfect` | exakte Zukunftswerte — Sonderfall und Oberschranke, zum Quantifizieren des Prognosewerts |
| `persistence` | naive Fortschreibung, für PV auf dem Clear-Sky-Index |
| `synthetic` | realisierungskonsistentes Fehlermodell (**Default**) |
| `model` | echtes Prognosemodell (z. B. Gradient Boosting auf Wetterfeatures) — Ausbaustufe |

Die zentrale Eigenschaft des `synthetic`-Modus: die Prognose wird **aus der Realisierung erzeugt** (`forecast = realisation ⊕ error`), nicht unabhängig gezogen. Andernfalls sind Prognose und Realisierung statistisch inkonsistent und die Policy kann den Fehler teilweise invertieren — ein Leck, das im Training wie Können aussieht und in der Evaluation verschwindet.

Fehlerstruktur je Größe:

- **PV** — multiplikativer Fehler auf dem Clear-Sky-Index, AR(1)-korreliert über den Horizont, heteroskedastisch (σ wächst mit dem Horizont h und mit der Bewölkungsvariabilität; bei klarem Himmel nahezu fehlerfrei, bei wechselnder Bewölkung groß).
- **Last** — additiver Fehler mit tageszeitabhängigem σ(h).
- **Wärmebedarf** — Fehler auf der Außentemperaturprognose, propagiert durch das thermische Modell; dadurch automatisch zeitlich korreliert.
- **EV** — der dominierende Unsicherheitsterm. Abfahrtszeit und Energiebedarf sind Verteilungen, keine Punktwerte. Der Agent erhält eine Punktprognose plus optional ein Streuungsmaß; die Realisierung weicht davon systematisch ab.

`forecast_seed` ist von Trainings- und Szenario-Seed getrennt, sodass dieselbe Episode reproduzierbar dieselbe Prognose erhält — sonst sind zwei Runs mit gleichem Trainings-Seed nicht vergleichbar. Die σ(h)-Parameter werden gegen Literaturwerte kalibriert und stehen in der Konfiguration, nicht im Code.

---

### 6.8 Safe-RL-Mechanismen als vergleichbare Komponente

> **Einordnung nach v0.4:** Die in diesem Abschnitt beschriebenen Mechanismen sind **heuristisch** — sie reduzieren Verletzungen, garantieren aber nichts. Sie bleiben als Vergleichsarme im Experiment, weil die wissenschaftlich interessante Frage nun lautet: *Was kostet die Garantie gegenüber der Heuristik?* Die Zielarchitektur für den Betrieb ist der zertifizierte Shield in §6.9.

Die Skizze aus v0.1 („Projektion oder Fallback, zuschaltbar") hätte einen Vergleich verschiedener Safe-RL-Umsetzungen **nicht** erlaubt. Der Grund: Masking, Projektion und Ersetzung greifen an drei verschiedenen Stellen an und stellen verschiedene Anforderungen, die nicht hinter einem gemeinsamen Schalter verschwinden. Der Layer wird daher in drei orthogonale Achsen zerlegt.

#### Achse 1 — `FeasibilityModel`: Woher kommt die Zulässigkeitsinformation?

Das ist der eigentliche Knackpunkt, und er wird in der Literatur oft übergangen: **ob eine Aktion das Spannungsband verletzt, weiß man erst nach dem Lastfluss.** Jeder Mechanismus, der *vor* der Ausführung eingreift, braucht also ein Vorhersagemodell:

| Implementierung | Beschreibung | Kosten |
|---|---|---|
| `sensitivity` | linearisierte Sensitivitäten ∂U/∂P, ∂U/∂Q und ∂I/∂P aus pandapower am aktuellen Arbeitspunkt | sehr günstig, am Arbeitspunkt genau, bei großen Aktionen ungenau |
| `learned` | neuronales Netz, trainiert auf (Zustand, Aktion) → Grenzwertverletzung | günstig, Genauigkeit unklar, Trainingsaufwand |
| `exact` | echter AC-Lastfluss je Kandidatenaktion | teuer; **kein Zertifizierer** — ein exakter Lastfluss bewertet *einen* Punkt und sagt nichts über die Unsicherheitsmenge der unsteuerbaren Einspeisungen im folgenden Intervall aus. Nur als Referenz und für die Fehlerdiagnose nutzbar |

Die Trennung dieser Achse von Achse 2 ist der Kern des Vergleichsrahmens: nur so lässt sich unterscheiden, ob ein Mechanismus schlecht ist oder nur sein Zulässigkeitsmodell.

#### Achse 2 — `SafetyMechanism`: Wie wird eingegriffen?

```python
class SafetyMechanism(Protocol):
    requires_discrete: bool
    def mask(self, obs, state) -> np.ndarray | None:      # nur Masking
    def transform(self, a, obs, state) -> tuple[np.ndarray, InterventionInfo]:
```

| Modus | Eingriffspunkt | Voraussetzungen | Umsetzung |
|---|---|---|---|
| `none` | – | – | Referenzarm des Vergleichs |
| `mask` | **vor** dem Sampling, auf der Politikverteilung | **diskreter** Aktionsraum; Maske muss vor dem Lastfluss bekannt sein | `MaskablePPO` (sb3-contrib); Maske aus `FeasibilityModel` über alle diskreten Stufen |
| `project` | **nach** dem Sampling, kontinuierlich | konvexe (linearisierte) zulässige Menge | QP: `min ‖a′ − a‖²` unter linearisierten Restriktionen (`qpsolvers`/`osqp`) |
| `replace` | nach dem Sampling, binär | ein vertrauenswürdiger Fallback-Regler | Verletzungsprädiktion → Aktion von B3/B4 (Q(U)/P(U)-Droop) ausführen |
| `safety_layer` | nach dem Sampling, gelernt | vorab trainiertes lineares Kostenmodell | Ansatz nach Dalal et al.: analytische Einzelrestriktions-Korrektur |

#### Achse 3 — `LearningCoupling`: Was sieht der Lernalgorithmus?

Diese Achse ist in der Praxis folgenreicher als die Wahl des Mechanismus selbst und wird häufig implizit und inkonsistent entschieden:

| Modus | Im Buffer/Rollout landet | Wirkung |
|---|---|---|
| `store_proposed` | die **vorgeschlagene** Aktion `a` | Policy lernt nicht, dass sie korrigiert wurde; Gradienten sind gegenüber der ausgeführten Dynamik verzerrt |
| `store_executed` | die **ausgeführte** Aktion `a′` | für Off-Policy-Q-Lernen konsistent, verändert aber die Verhaltensdichte (Importance-Sampling-Annahmen bei PPO verletzt) |
| `differentiable` | `a′`, Gradienten fließen durch die Projektion | theoretisch am saubersten, erfordert differenzierbare QP-Schicht (`cvxpylayers`) und deutlich mehr Rechenzeit |
| `penalty_only` | `a` plus Strafterm für den Eingriff | Mechanismus wirkt nur zur Laufzeit, nicht im Lernsignal |

Konfigurationsseitig ist damit jeder Vergleich ein Kreuzprodukt statt eines Forks:

```yaml
safety:
  feasibility: sensitivity     # none | sensitivity | learned | exact
  mechanism:   project         # none | mask | project | replace | safety_layer
  coupling:    store_executed  # store_proposed | store_executed | differentiable | penalty_only
  intervention_penalty: 0.0
```

#### Was gemessen werden muss, damit der Vergleich etwas aussagt

Die reine Verletzungsrate genügt nicht — sie vermischt die Güte des Mechanismus mit der Güte des Zulässigkeitsmodells und mit der Konservativität. `eval/safety_kpis.py` erfasst deshalb:

- **`intervention_rate`** — Anteil der Schritte mit veränderter Aktion
- **`intervention_magnitude`** — `‖a′ − a‖` (Verteilung, nicht nur Mittelwert)
- **`residual_violation_rate`** — Verletzungen **nach** dem exakten Lastfluss. Für heuristische Mechanismen eine KPI; für den zertifizierten Shield (§6.9) eine Testzusicherung mit Sollwert null. Ein Mechanismus mit linearisiertem Modell garantiert nichts; diese Zahl sagt, wie viel von der behaupteten Sicherheit übrig bleibt.
- **`false_positive_rate`** — Eingriffe, die unnötig waren (der exakte Lastfluss zeigt: die Originalaktion war zulässig). Direktes Maß für Überkonservativität.
- **`false_negative_rate`** — kein Eingriff, aber Verletzung. Direktes Maß für Modellfehler.
- **`conservatism_cost`** — der den Eingriffen zurechenbare Anteil an Abregelung, nicht gelieferter Ladeenergie und Komfortverletzung
- **`decision_time_ms`** — QP-Lösungszeit ist nicht vernachlässigbar und ist genau die Größe, mit der man gegen `opf_myopic` argumentiert
- **Lerneffekt** — Sample-Effizienz und Endperformance mit/ohne Mechanismus, plus Verletzungen **während** des Trainings (der eigentliche Zweck von Safe RL)

Die beiden Fehlerraten erfordern einen **Schatten-Lastfluss**: pro Schritt wird zusätzlich die unveränderte Originalaktion exakt gerechnet. Das verdoppelt die Kosten und wird daher nur auf dem Evaluationssatz und auf Stichproben im Training aktiviert (`safety.shadow_powerflow: eval_only`). Ohne diese Größen ist ein Vergleich verschiedener Safe-RL-Verfahren nicht mehr als eine Sammlung von Endergebnissen.

#### Drei Einschränkungen, die man vorab akzeptieren muss

1. **Masking zwingt zu diskreten Aktionen und zu PPO.** In SB3 existiert Action-Masking nur über `MaskablePPO`; es gibt kein maskiertes SAC/TD3. Ein Vergleich über alle Mechanismen hinweg muss den Algorithmus also auf PPO festlegen, sonst vermischt man Mechanismus- und Algorithmuseffekt. Zusätzlich braucht der Masking-Arm eine Diskretisierung von PV-Abregelung, BESS- und Ladeleistung. Damit die Diskretisierung nicht zum zweiten Confounder wird, erhält die Env den Schalter `action.discretization: none | k_levels`, und der Masking-Arm wird gegen einen **ebenfalls diskretisierten** Projektions- und Ersetzungsarm verglichen. Empfohlener Studienaufbau: erst Mechanismusvergleich bei fixiertem PPO und fixierter Diskretisierung, dann Mechanismus × Algorithmus nur für `project`/`replace`.
2. **Der Rechenaufwand der Maske skaliert mit der Aktionsraumgröße.** Die Maske muss für jede diskrete Stufe jeder Anlage eine Zulässigkeitsaussage treffen. Mit linearisierten Sensitivitäten ist das eine Matrixmultiplikation und billig; mit `exact` ist es undurchführbar. Der `exact`-Modus ist deshalb nur für `replace` und für die Fehlerdiagnose vorgesehen.
3. **Unter EN 50160 ist „sicher" nicht per Schritt definiert** — und das ist die wichtigste Wechselwirkung mit Entscheidung D2. K95 ist ein Wochenbudget, kein Momentankriterium; ein aktionsbasierter Mechanismus kann es gar nicht direkt durchsetzen und muss gegen ein strengeres Surrogat arbeiten (das harte ±10 %-Band). Er ist damit **strukturell konservativer als die Norm verlangt**. Umgekehrt ist **K100 (−15 %, alle Fenster) ein echtes hartes Kriterium** und das natürliche Ziel eines aktionsbasierten Mechanismus.

Daraus ergibt sich eine saubere Arbeitsteilung, die selbst ein Ergebnis der Arbeit sein kann:

> **K100 über den Safety-Mechanismus (harte Grenze), K95 über die Lagrange-Formulierung (Budgetkriterium).** Diese Hybridvariante ist der interessanteste Arm der Studie, weil sie jedes der beiden Normkriterien mit dem Verfahren adressiert, dessen Garantieform dazu passt — Momentangarantie gegen Momentankriterium, Erwartungswertgarantie gegen Perzentilkriterium.

Der Vergleichsrahmen erlaubt damit vier Kategorien in einem Kreuzprodukt: ungeschützt, aktionsbasiert (drei Varianten × drei Kopplungen), Lagrange, hybrid.

---

### 6.9 Zertifizierte Sicherheit (Zielarchitektur)

Anforderung: eine belastbare Garantie, keine Sicherheitsheuristik. Das ist erreichbar — aber nur als **bedingte** Garantie, deren Voraussetzungen vollständig offengelegt werden müssen. In kritischer Infrastruktur liegt der Wert genau in dieser Explizitheit, nicht in einer unqualifizierten Zusage.

#### Was garantiert wird — und was nicht

**Garantiert (Zielaussage):**

> Für jeden Zustand innerhalb der zertifizierten Betriebshülle und für **jede** Realisierung der unsteuerbaren Einspeisungen innerhalb der spezifizierten Unsicherheitsmenge gilt: die ausgeführte Aktion führt zu einer Lösung der AC-Lastflussgleichungen, die alle Spannungs- und Betriebsmittelgrenzen (K100, thermische Grenzen) einhält — und es existiert zu jedem künftigen Zeitpunkt eine ebenfalls zulässige Fortsetzung.

**Nicht garantiert**, und das muss im Ergebnisbericht stehen:

- Gültigkeit gegenüber der Realität. Die Garantie ist modellbezogen. Modellfehler (Impedanzen, Topologie, Schieflast, Messfehler) liegen außerhalb.
- Kundenkomfort. Der Shield darf im Grenzfall PV vollständig abregeln und Wärmepumpen sowie Ladepunkte sperren. Netzintegrität ist hart beschränkt, Komfort ist weiches Ziel — diese Hierarchie ist für kritische Infrastruktur die richtige, aber sie ist eine Entscheidung und keine Naturkonstante.
- Sicherheit bei Netzfehlern und Topologieänderungen. Die Garantie gilt je Topologie; Schalthandlungen erfordern eine erneute Zertifizierung.

#### Vier Voraussetzungen (P1–P4)

| | Voraussetzung | Wie sie hergestellt wird | Wenn sie verletzt ist |
|---|---|---|---|
| **P1** | Zulässigkeitsaussage ist *hinreichend*, nicht approximativ | konvexe Restriktion statt Linearisierung (unten) | keine Garantie, nur Heuristik |
| **P2** | Unsicherheitsmenge der unsteuerbaren Einspeisungen ist zertifiziert | physikalische Schranken oder Conformal Prediction | Garantie gilt nur für nachträglich als „typisch\" erklärte Fälle |
| **P3** | Rekursive Zulässigkeit — es existiert immer eine zulässige Fortsetzung | prädiktiver Sicherheitsfilter mit terminaler Rückfallmenge | Agent kann sich in Zustände ohne zulässige Aktion manövrieren |
| **P4** | Rückfallaktion ist selbst zulässig | **offline je Netz × Szenario zu verifizieren** | **kein Regler kann etwas garantieren — das Netz ist unzureichend** |

P4 ist der Punkt, der über das Projekt hinausweist: wenn die Grundlast allein (PV vollständig abgeregelt, alle flexiblen Lasten gesperrt) in einem Szenario schon Unterspannung erzeugt, existiert keine sichere Rückfallaktion. Dann ist nicht die Regelung das Problem, sondern der Netzausbau. Die Verifikation von P4 ist deshalb kein Vorspiel, sondern ein Ergebnis: **die zertifizierte Betriebshülle je Netz und Ausbaugrad** sagt, bis wohin autonome Regelung Netzausbau ersetzen kann und wo nicht.

#### Baustein 1 — Zertifizierte AC-Zulässigkeit statt Linearisierung

Der passende mathematische Gegenstand ist die **konvexe Restriktion** des Lastfluss-Zulässigkeitsbereichs: sie identifiziert eine konvexe Teilmenge der Leistungseinspeisungen, in der die Existenz einer Lastflusslösung garantiert ist und diese die Betriebsrestriktionen erfüllt. Entscheidend ist der Unterschied zu den verbreiteten Relaxierungen: im Gegensatz zu konvexen Relaxierungen liefert die konvexe Restriktion eine hinreichende Bedingung für Lastflusszulässigkeit und ist besonders für Probleme mit Unsicherheit in Erzeugung und Nachfrage geeignet. Eine Relaxierung ist eine äußere Approximation und für Garantien wertlos; nur eine innere Approximation trägt eine hinreichende Aussage. Das Verfahren führt auf konvexe quadratische Restriktionen, die einen für den praktischen Netzbetrieb hinreichend großen Bereich abdecken.

Für P2 existiert die passende Erweiterung: die **robuste konvexe Restriktion** ist eine konvexe innere Approximation des nichtkonvexen zulässigen Bereichs des AC-OPF-Problems, die Unsicherheit in den Leistungseinspeisungen berücksichtigt, mit dem Anspruch, die Betriebsgrenzen für alle Unsicherheitsrealisierungen innerhalb einer spezifizierten Unsicherheitsmenge einzuhalten.

Damit wird der `project`-Mechanismus aus §6.8 von einer Heuristik zu einem beweisbaren Verfahren: die Projektion erfolgt nicht auf eine Linearisierung, sondern auf die robuste konvexe Restriktion. Das ist ein konvexes (SOCP/QCQP) Problem und pro Regelschritt lösbar.

*Rückfalloption, falls die konvexe Restriktion für die NS-Netzgröße zu aufwendig wird:* Linearisierung **plus rigoroser Restgliedschranke** (Intervallarithmetik auf dem Taylor-Rest, mit nach außen gerichteter Rundung). Weniger scharf, aber ebenfalls hinreichend. Reine Linearisierung ohne Restgliedschranke ist es nicht.

**Literatur:** Lee, Nguyen, Dvijotham, Turitsyn, *Convex Restriction of Power Flow Feasibility Sets*, IEEE Trans. Control of Network Systems 6(3), 2019 (arXiv:1803.00818) · Lee, Turitsyn, Molzahn, Roald, *Robust AC Optimal Power Flow with Robust Convex Restriction*.

#### Baustein 2 — Zertifizierte Unsicherheitsmengen

Die Garantie ist nur so gut wie die Menge, über die sie quantifiziert. Zwei Wege, die unterschiedliche Aussagen liefern und beide berichtet werden sollten:

- **Deterministisch / physikalisch** — Hausanschlusssicherung, vereinbarte Anschlussleistung, Wechselrichter-`S_max`, maximale WP-Leistung, maximale Ladepunktleistung. Ergibt eine *deterministische* Garantie ohne Restwahrscheinlichkeit, ist aber sehr konservativ (die Gleichzeitigkeit aller Anschlüsse am Maximum ist praktisch unmöglich).
- **Datengetrieben mit endlicher Stichprobengarantie** — Conformal Prediction oder Szenarioansatz auf den Zeitreihen aus §3.2, liefert Mengen mit Konfidenz 1−δ. Schärfer, aber die Garantie wird probabilistisch und δ muss genannt werden.

Empfehlung: die deterministische Variante als Hauptaussage („für kritische Infrastruktur"), die datengetriebene als Quantifizierung des Preises der Strenge. Die Differenz zwischen beiden — wieviel Abregelung die Worst-Case-Annahme kostet — ist eine der interessantesten Zahlen der Arbeit.

#### Baustein 3 — Rekursive Zulässigkeit

Einschritt-Sicherheit genügt nicht. Alle Aktoren außer PV haben Zustand: ein jetzt zulässiges Laden kann den Speicher füllen, sodass später keine Abwärtsflexibilität mehr existiert; ein gesperrtes EV muss vor Abfahrt laden. Das passende Werkzeug ist der **prädiktive Sicherheitsfilter**: er prüft in Echtzeit die Sicherheit einer vorgeschlagenen lernbasierten Eingabe, indem er für den nächsten Schritt eine sichere Rückfalltrajektorie in Richtung einer bekannten sicheren Menge sucht; die Möglichkeit, die potenziell unsichere Eingabe nötigenfalls zu verändern, stellt Sicherheit für alle künftigen Zeitpunkte her. Die Struktur passt genau zum Vorhaben: jeder RL-Algorithmus, der das ursprüngliche System geregelt hätte, kann stattdessen auf das sichere System angewendet werden, was eine zertifiziert sichere RL-Anwendung ergibt — der Filter greift minimalinvasiv ein und filtert Eingaben nur dann, wenn Sicherheit nicht garantiert werden kann.

Konkretisierung für dieses System:

- **Terminale Rückfallmenge**: Zustände, aus denen die Rückfallregel (PV vollständig abgeregelt, flexible Lasten auf Minimum, BESS frei zur Netzstützung) für alle Unsicherheitsrealisierungen zulässig bleibt. Ihre Nichtleere ist P4.
- **Rückfalltrajektorie** über Horizont N (typisch 2–8 h, mindestens so lang, dass ein voller Speicher abgebaut und ein EV-Ladevorgang abgeschlossen werden kann), gerechnet auf dem robusten konvexen Restriktionsmodell.
- Die Aktion des Agenten wird zertifiziert, **wenn** eine solche Trajektorie existiert; andernfalls wird der erste Schritt der Rückfalltrajektorie ausgeführt.

**Literatur:** Wabersich & Zeilinger, *Linear Model Predictive Safety Certification for Learning-Based Control*, CDC 2018 (arXiv:1803.08552) · Wabersich & Zeilinger, *A predictive safety filter for learning-based control of constrained nonlinear dynamical systems*, Automatica 129:109597, 2021 · Wabersich et al., *Data-Driven Safety Filters*, IEEE Control Systems Magazine, 2023 (Überblick).

#### Baustein 4 — Verifikation der Implementierung

Die Garantie lebt im Code, nicht im Aufsatz. Vier Maßnahmen, ohne die die Aussage nichts wert ist:

1. **Der Shield ist ein eigenes Modul mit eigenem Test-Suite.** Er teilt keinen Zustand mit dem Agenten, bekommt keine von der Policy erzeugten Parameter und ist im Training **und** in der Evaluation aktiv (sonst entsteht eine Train/Deploy-Diskrepanz, die die Garantie im Betrieb sofort entwertet).
2. **Falsifikationssuche als CI-Job.** Gezielte adversariale Suche über (Zustand, Aktion, Unsicherheitsrealisierung) nach einem als zertifiziert-sicher eingestuften Fall, der im exakten AC-Lastfluss eine Grenze verletzt. Jeder Treffer ist ein **Defekt des Zertifizierers**, keine tolerierbare Statistik.
3. **`residual_violation_rate` wird zur Testzusicherung.** In §8.3 ist sie eine KPI; für den zertifizierten Arm ist sie eine Assertion mit Sollwert exakt null. Ein Wert > 0 bricht den Build.
4. **Gleitkomma-Korrektheit.** Intervall- und Restgliedrechnungen brauchen nach außen gerichtete Rundung. Eine Garantie, die an der letzten Stelle kippt, ist keine.

#### Konsequenzen für die Architektur

- `FeasibilityModel` wird zu **`SafetyCertifier`** mit *zweiwertiger, asymmetrischer* Antwort:

```python
class SafetyCertifier(Protocol):
    def certify(
        self, state: SystemState, action: np.ndarray, uncertainty: UncertaintySet
    ) -> Verdict:
        # Verdict: CERTIFIED_SAFE oder NOT_CERTIFIED.
        # Niemals "unsafe" behaupten: NOT_CERTIFIED heisst "nicht beweisbar",
        # und das genuegt fuer den Rueckfall.
        ...
```

  Diese Asymmetrie ist wesentlich: der Zertifizierer darf konservativ sein, aber nie optimistisch.
- `safety.mechanism` erhält `certified_shield` als einzigen Modus, der die Garantie trägt. `mask`/`project`/`replace` mit Linearisierung bleiben als heuristische Vergleichsarme (§6.8).
- `safety.coupling` wird für den zertifizierten Arm auf `store_executed` festgelegt: der Agent lernt auf dem *sicheren System*, wie es der prädiktive Sicherheitsfilter vorsieht.
- **Grenzwerte des Shields = K100 (+10 %/−15 %)**, nicht ±10 %. Würde der Shield ±10 % hart erzwingen, wäre er unnötig konservativ und das K95-Budget wäre trivial ungenutzt. Arbeitsteilung: Shield garantiert K100 hart, die Lagrange-Formulierung verwaltet das K95-Budget (§6.4, §6.6).
- **Rechenzeit ist ein Risiko, nicht nur eine KPI.** Wenn Zertifizierung pro Schritt so teuer wird wie ein MPC-Schritt, verliert das Argument „RL ist zur Laufzeit billiger als MPC" seine Substanz. `certification_time_ms` gehört daher zu den Kernergebnissen und nicht in den Anhang. Gegenmaßnahmen: Vorberechnung der Restriktionskoeffizienten je Arbeitspunkt-Cluster, Warmstart des SOCP, reduzierte Zertifizierung nur auf den kritischen Strängen.

#### Zwei Einschränkungen, die den Modellrahmen berühren

1. **Symmetrie.** Die verfügbaren Formulierungen der konvexen Restriktion sind für das symmetrische Dreiphasen- bzw. Einphasen-Äquivalent aufgestellt. Niederspannungsnetze sind real unsymmetrisch, und die Datenlage gibt das her (HTW-Berlin-Profile sind phasenaufgelöst). Eine Zertifizierung für das unsymmetrische Vierleitersystem ist erheblich aufwendiger und in der Literatur schwächer abgedeckt. **Festlegung:** der zertifizierte Arm arbeitet auf dem symmetrischen Modell; Schieflast wird als separate Sensitivitätsstudie *ohne* Garantie gefahren. Diese Einschränkung ist offenzulegen, weil Schieflast in NS-Netzen für Spannungsprobleme relevant ist.
2. **Beobachtbarkeit.** Der Zertifizierer braucht den Systemzustand. Mit `sensor_config: realistic` (§6.2) ist der Zustand nur teilweise beobachtet — dann ist eine harte Garantie nur möglich, wenn die Zustandsschätzung selbst eine beschränkte, zertifizierte Fehlerschranke hat, die in die Unsicherheitsmenge eingeht. Drei Wege: (i) der Zertifizierer erhält den Vollzustand (idealisiert, klar zu kennzeichnen); (ii) Intervall-Zustandsschätzung aus begrenzten Messungen (rigoros, deutlich aufwendiger); (iii) eine explizit benannte Annahme über die Messinfrastruktur. Wichtig ist, dass **der Shield mehr Information bekommen darf als der Agent** — er ist eine separate, besser instrumentierte, vertrauenswürdige Komponente. Aber die dafür nötige Messausstattung ist eine Aussage über die Realisierbarkeit und muss beziffert werden.

---

## 7. L4 — Agenten und Referenzverfahren

### 7.1 Gemeinsames Controller-Interface

```python
class Controller(Protocol):
    def reset(self, obs, info) -> None: ...
    def act(self, obs, info) -> np.ndarray: ...
```

RL-Policies, regelbasierte Verfahren, OPF und MPC implementieren dasselbe Interface. `eval/runner.py` kennt nur `Controller` — dadurch ist der Vergleich strukturell fair. Verfahren, die mehr als die Agenten-Beobachtung brauchen (OPF braucht das Netzmodell, MPC die Prognose), erhalten diesen Zugriff über `info["privileged"]`, und dieser privilegierte Zugriff wird im Ergebnisbericht explizit ausgewiesen.

### 7.2 RL-Agenten

`agents/factory.py` baut aus einer YAML-Spezifikation ein SB3-Modell:

```yaml
agent:
  algo: sac                 # ppo | a2c | sac | td3 | ddpg | dqn
                            # sb3-contrib: tqc | qrdqn | recurrent_ppo | trpo | crossq | ars
  policy: MultiInputPolicy
  policy_kwargs:
    net_arch: [256, 256]
    activation_fn: torch.nn.ReLU
    features_extractor: asset_encoder   # mlp | asset_encoder | gnn | attention
  hyperparams: {learning_rate: 3.0e-4, batch_size: 256, train_freq: 1, ...}
```

Eigene Feature-Extraktoren (`agents/extractors/`):

- `MlpExtractor` — Baseline auf flacher Beobachtung.
- `AssetSetEncoder` — typspezifische Encoder + Pooling über Anlagen gleichen Typs; erlaubt Transfer auf Netze mit anderer Anlagenzahl.
- `GraphExtractor` — Message Passing auf der Netztopologie (torch-geometric); besonders passend, weil Spannungskopplungen entlang der Stränge lokal sind.
- `AttentionExtractor` — Transformer-Encoder über Anlagen-Token.

Kombiniert mit den drei Aktionsraum-Modi und der Wahl `MLP / Recurrent / Graph` ergibt sich die gewünschte einfache Vergleichbarkeit verschiedener Agentenarchitekturen: eine Zeile in der Konfiguration.

Hyperparametersuche über Optuna (`experiment/sweep.py`), mit `TPESampler` und `MedianPruner`, Suchraum je Algorithmus deklarativ definiert. RL Zoo3 kann als Vorlage für sinnvolle Suchräume dienen.

### 7.3 Referenzverfahren

Diese sind für die Arbeit genauso wichtig wie die Agenten selbst, weil sie die Skala definieren, auf der RL-Ergebnisse gelesen werden.

| # | Verfahren | Rolle | Umsetzung |
|---|---|---|---|
| B0 | `do_nothing` | untere Schranke; ungeregeltes Netz | trivial |
| B1 | `random` | Sanity-Check gegen degenerierte Policies | trivial |
| B2 | `fixed_cap` | starre Einspeisebegrenzung (z. B. 70 % / 60 % der Anlagenleistung) | trivial |
| B3 | `cosphi_p` / `q_u_droop` | lokale Blindleistungsregelung nach VDE-AR-N 4105 | Kennlinien-Parametrierung |
| B4 | `p_u_droop` | spannungsabhängige Wirkleistungsabregelung | Kennlinie mit Totband und Zeitkonstante |
| B5 | `en14a_dimming` | Leistungsreduktion steuerbarer Verbrauchseinrichtungen (§14a EnWG, Netzanschlusspunkt auf 4,2 kW) | regelbasiert, praxisnaher Referenzfall |
| B6 | `greedy_local` | PV-Überschussladen für BESS/EV, „so spät wie nötig" laden | heuristisch |
| B7 | `opf_myopic` | zentraler AC-OPF je Zeitschritt (`pp.runopp`) | zentral, keine Vorausschau |
| B8 | `mpc_forecast` | Mehrschritt-Optimierung mit realistischer Prognose (Persistenz oder Day-Ahead-Modell) | Pyomo/linopy + linearisiertes Netz, iterativ gegen AC |
| B9 | `mpc_oracle` | Mehrschritt-Optimierung mit perfekter Vorausschau | **obere Schranke**; quantifiziert, wieviel vom Optimum RL erreicht |

B7–B9 nutzen ein reduziertes, linearisiertes Netzmodell (LinDistFlow oder Sensitivitätsmatrizen aus pandapower) für die Optimierung und werden anschließend im vollen AC-Modell nachgerechnet — so bleibt die physikalische Bewertung für alle Verfahren identisch. B9 ist der wichtigste Referenzwert: „RL erreicht X % der Oracle-Performance bei Y % der Rechenzeit" ist eine belastbare Aussage, „RL ist besser als ungeregelt" ist keine.

Zur Formulierung der Perzentilrestriktion in B7–B9 und zur notwendigen Eigenparametrierung von B3–B6 siehe §6.6, Konsequenzen 4 und 5.

---

## 8. L5 — Speicherung, Auswertung, Visualisierung

### 8.1 Run-Layout

```
results/<experiment>/<run_id>/
├── config_resolved.yaml     # vollständig aufgelöste Hydra-Config
├── manifest.json            # code_commit, data_manifest_hash, seeds, Umgebung
├── env_spec.json            # Observation/Action-Space-Definition (maschinenlesbar)
├── requirements.lock
├── checkpoints/             # model_<steps>.zip, vecnormalize.pkl
├── tb/                      # TensorBoard-Events
├── metrics/
│   ├── train_scalars.parquet   # Schrittweise Trainingsskalare
│   └── episode_summary.parquet # eine Zeile je Episode: KPIs + Reward-Terme
├── traces/
│   └── eval_ep<k>.parquet   # Zeitreihe je Eval-Episode: Aktionen, Zustände, Verletzungen
└── eval/
    ├── kpi_table.parquet
    └── report.html
```

Zwei Granularitäten sind bewusst getrennt: **Skalare** werden immer geschrieben (klein), **Traces** nur für Evaluationsepisoden und stichprobenartig im Training (sonst explodiert das Datenvolumen). Format durchgehend Parquet; eine einzelne lange Trace-Tabelle mit Spalten `(run_id, episode, t, entity, quantity, value)` ist analysefreundlicher als viele Einzeldateien.

`run_id = sha256(code_commit ‖ config_hash ‖ data_manifest_hash ‖ seed)[:12]`. Eine SQLite- oder MLflow-Registry (`experiment/registry.py`) indexiert alle Runs, damit Auswertungen über Studien hinweg per Query laufen.

### 8.2 Reproduzierbarkeit — konkrete Maßnahmen

- **Seeds getrennt nach Zweck:** `scenario_seed`, `episode_seed`, `train_seed`, `eval_seed`. Worker-Seeds = `base + worker_idx`, deterministisch abgeleitet.
- `stable_baselines3.common.utils.set_random_seed(seed)` plus explizites Seeding von `env.reset(seed=...)`, `env.action_space.seed(...)`.
- `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`. Für kleine NS-Netze ist CPU-Training oft schneller *und* leichter determinierbar als GPU.
- **Umgebung eingefroren:** Lockfile + Dockerfile; `pip freeze` im Manifest. pandapower- und simbench-Versionen unbedingt pinnen, da Profilzuordnungen und Solver-Defaults zwischen Versionen abweichen können.
- **Daten eingefroren:** SHA-256 je Rohdatei, versionierte Preprocessing-Pipeline, Cache-Schlüssel enthält die Policy-Parameter. Optional DVC für die Rohdatenablage.
- **Reproduktionstest als CI-Job:** ein kurzer Run (z. B. 20 k Steps) wird zweimal mit gleichem Seed ausgeführt; die Trainingsskalare müssen bitidentisch sein. Dieser Test findet Reproduzierbarkeitslecks zuverlässiger als jede Dokumentation.
- Bekannte Restrisiken dokumentieren: Solver-Nichtdeterminismus bei sparse LU, `SubprocVecEnv`-Reihenfolge bei asynchronen Rollouts (mit `PPO` unkritisch, bei Off-Policy-Replay relevant).

### 8.3 KPI-Engine

`eval/kpis.py` berechnet aus Traces:

**Netzqualität — Leit-KPI normkonform**
- **`en50160_pass_rate`** — Anteil der (Bus, Woche)-Paare, die K95 *und* K100 erfüllen. Headline-Kennzahl.
- **`p95_vm_pu`** je Bus und Woche (95-%-Perzentil der 10-min-Mittelwerte) sowie der Worst-Bus-Wert — eine *kontinuierliche* Größe, unverzichtbar, weil `pass_rate` binär ist und Verbesserungen unterhalb der Schwelle sonst unsichtbar bleiben.
- **`budget_utilisation`** — verbrauchter Anteil der 50 zulässigen Fenster. Zeigt, wie knapp eine Lösung ist: 0,98 und 1,02 sind in der `pass_rate` ein Unterschied wie Tag und Nacht, operativ aber fast identisch. Beide Kennzahlen zusammen zu berichten verhindert Fehlschlüsse.
- Sekundär, zur Vergleichbarkeit mit Literatur, die ein hartes Band verwendet: Anzahl/Dauer von ±10-%-Verletzungen je Zeitschritt, `max|ΔU|`, `∫|ΔU| dt` in pu·h
- Überlastdauer und Überlastintegral für Leitungen und Trafo in MVA·h, Anzahl betroffener Betriebsmittel

**Kosten der Regelung**
- abgeregelte PV-Energie in kWh und % des Potenzials
- nicht gelieferte EV-Ladeenergie bei Abfahrt (kWh, Anteil betroffener Ladevorgänge)
- WP-Komfortverletzung in K·h, maximale Sperrdauer
- BESS-Durchsatz, Äquivalentzyklen
- Netzverluste, Schalthandlungen/Aktionsvarianz

**Lernverhalten**
- Lernkurven (Return, Einzel-Reward-Terme, Verletzungsrate) über Steps
- Sample-Effizienz (Steps bis Schwellwert-KPI)
- Stabilität über Seeds
- Rechenzeit je Entscheidung (relevant für den Vergleich mit MPC/OPF)

**Sicherheitsmechanismen** (nur wenn `safety.mechanism != none`)
- Eingriffsrate, Eingriffsstärke, Residual-Verletzungsrate
- False-Positive- und False-Negative-Rate gegen den Schatten-Lastfluss
- den Eingriffen zurechenbare Konservativitätskosten
- Verletzungen während des Trainings (nicht nur nach Konvergenz)
- Rechenzeit je Entscheidung, für den zertifizierten Arm getrennt als `certification_time_ms`
- für den zertifizierten Arm zusätzlich: Größe der zertifizierten Betriebshülle je Netz × Ausbaugrad (P4), Treffer der Falsifikationssuche (Sollwert null)

Definitionen und Begründung in §6.8 und §6.9.

**Statistische Auswertung.** Mindestens 5, besser 10 Seeds je Konfiguration. Aggregation über **IQM mit Bootstrap-Konfidenzintervallen** und Performance-Profiles (Vorgehen nach `rliable`) statt Mittelwert ± Standardabweichung — bei RL sind Verteilungen über Seeds regelmäßig schiefe und mehrgipflige Verteilungen, bei denen der Mittelwert irreführt. Zusätzlich paarweise Vergleiche gegen Baselines auf dem festen Evaluationssatz (gepaart, gleiche Episoden).

### 8.4 Visualisierung

`viz/` mit zwei Ebenen:

**Statische Plots** (matplotlib, für Publikation)
- Lernkurven mit Seed-Konfidenzband
- Balken-/Box-Plots der KPIs: RL-Varianten vs. B0–B9
- Spannungs-Heatmap (Knoten × Zeit) je Evaluationsepisode, mit markierten Verletzungen
- Jahresdauerlinien der Knotenspannungen und Betriebsmittelauslastungen
- Pareto-Darstellung Abregelungsenergie vs. Verletzungsenergie — die Kernabbildung für „was kostet Netzsicherheit"
- Netzplan über `pandapower.plotting`, Knoten gefärbt nach Verletzungshäufigkeit
- Aktionsprofile: Was tut der Agent über den Tag? (essenziell für Interpretierbarkeit)

**Interaktives Dashboard** (Streamlit + Plotly)
- Episoden-Replay: Zeitschieber, synchron Spannungen, Auslastungen, Aktionen, Reward-Terme
- Run-Vergleich per Auswahl aus der Registry
- Reward-Dekomposition über Zeit

Berichte werden über `scripts/report.py` aus den Parquet-Dateien regeneriert, nie manuell aus Notebooks zusammenkopiert.

---

## 9. L6 — Experiment-Orchestrierung

```
scripts/prepare_data.py   # Rohdaten -> validierter, resampelter Cache + Manifest
scripts/build_scenario.py # Netz + Ausbaugrad -> ScenarioSpec
scripts/train.py          # Hydra-Entry-Point: ein Run
scripts/sweep.py          # Optuna / Multirun über Seeds & Agentenvarianten
scripts/evaluate.py       # Controller (RL-Checkpoint oder Baseline) auf Eval-Satz
scripts/report.py         # Studie -> KPI-Tabellen, Plots, HTML-Bericht
```

Eine Studie ist eine YAML-Datei, die das Kreuzprodukt aufspannt, z. B.:

```yaml
# configs/experiment/study_agent_arch.yaml
grid:      [simbench_1-LV-rural1, simbench_1-LV-semiurb4, simbench_1-LV-urban6]
scenario:  [pv60_ev40_hp30]
agent:     [ppo_mlp, sac_mlp, tqc_mlp, recurrent_ppo, sac_gnn]
seed:      [1, 2, 3, 4, 5]
baseline:  [do_nothing, q_u_droop, en14a_dimming, opf_myopic, mpc_oracle]
```

Callbacks (`experiment/callbacks.py`): periodische Evaluation auf dem Validierungssatz, Checkpointing, KPI-Logging, Early-Stopping-Kandidat, Trace-Sampling. Wichtig: Modellauswahl (`best_model`) erfolgt auf dem **Validierungs**-Zeitraum, Berichtszahlen auf dem **Test**-Zeitraum. Sonst ist das Ergebnis optimistisch verzerrt.

---

## 10. Verwandte Arbeiten, an denen man sich orientieren (oder anlehnen) kann

- **OPF-Gym** (Digitalized Energy Systems, Uni Oldenburg) — gymnasium-kompatibles Framework für OPF-Probleme, das **pandapower für die Netzmodellierung und SimBench-Netze samt Zeitreihen** nutzt; enthält fünf Benchmark-Umgebungen, u. a. Spannungshaltung mit Blindleistung und Maximierung der EE-Einspeisung. Am nächsten an diesem Vorhaben; lohnt eine genaue Sichtung, ggf. als Basis statt Neuentwicklung.
  https://github.com/Digitalized-Energy-Systems/opfgym · https://opf-gym.readthedocs.io
- **PowerGridworld** (NREL) — leichtgewichtiges, modulares Framework für Multi-Agent-Gym-Umgebungen im Energiesystemkontext, mit pandapower-Lastflussmodell; die Komponenten-/Multi-Agent-Struktur ist eine gute Referenz für L2/L3.
- **Gym-ANM** — RL-Umgebungen für Active Network Management in Verteilnetzen.
- **PowerGym** (Siemens/Princeton) — Volt-Var-Control auf IEEE-Testnetzen (OpenDSS-basiert).
- **CommonPower** — Safe-RL im Smart-Grid-Kontext mit Pyomo-basiertem Modell und Prognose-Schnittstelle; interessant für den Safety-Layer und B8/B9.
- **CityLearn** — Demand Response / Gebäudeenergie, gute Vorlage für Benchmark-Disziplin und Evaluationsprotokoll.

---

## 11. Umsetzungs-Roadmap

| Meilenstein | Inhalt | Abschlusskriterium |
|---|---|---|
| **M0** Skelett | Repo, pyproject, Hydra-Grundgerüst, CI, Test-Setup | `pytest` läuft, leerer Run erzeugt Run-Ordner mit Manifest |
| **M1** Daten | SimBench-Adapter + Resampling + Validierung + Cache | 1 Jahr Profile bei `sim_dt = 10/5/2/1 min` reproduzierbar erzeugt, Manifest-Hash stabil |
| **M2** Netz | Loader, Asset-Mapping, PowerFlowEngine (inkl. hypothetischem Aufruf), Metriken, `PQAggregator` + EN-50160-Evaluator, **P4-Vorabprüfung** | Quasi-statische Jahressimulation ohne Regelung; `en50160_pass_rate` des ungeregelten Netzes je Szenario dokumentiert (= Referenzwert B0); Laufzeit je Step gemessen; **je Szenario dokumentiert, ob die Rückfallaktion zulässig ist** (siehe unten) |
| **M3** Minimal-Env | nur PV-Abregelung, flacher Aktionsraum, `pq_budget`-Features, Reward mit 2 Termen, `PettingZooAdapter` + Äquivalenztest | `check_env` besteht; PPO verbessert `en50160_pass_rate` gegenüber B0; Baselines B0/B2/B3 laufen mit eigener Parametrierung |
| **M4** Vollständige Aktorik | BESS, WP mit Puffermodell, EVSE mit Sessions | alle Aktoren in einer Episode aktiv; Aktionsprojektionen und Komfortverletzungen werden korrekt gezählt |
| **M5** Evaluation | KPI-Engine, fester Eval-Satz, Referenzverfahren B5–B9 | KPI-Tabelle RL vs. alle Baselines inkl. `mpc_oracle`; Reproduktionstest in CI grün |
| **M6** Agentenvergleich | Agenten-Factory, Feature-Extraktoren, Optuna-Sweeps | Studie über ≥ 4 Agentenarchitekturen × 5 Seeds mit IQM-Konfidenzintervallen |
| **M7a** Safe RL heuristisch | `FeasibilityModel` (sensitivity), Mechanismen `project`/`replace`/`mask`, Schatten-Lastfluss, Safety-KPIs | Mechanismusvergleich bei fixiertem PPO inkl. False-Positive/-Negative-Raten |
| **M7b** Zertifizierter Shield (eigenes Teilprojekt) | robuste konvexe Restriktion, zertifizierte Unsicherheitsmengen, prädiktiver Sicherheitsfilter, Falsifikations-CI | P4 je Netz × Ausbaugrad verifiziert (zertifizierte Betriebshülle dokumentiert); `residual_violation_rate == 0` als Build-Bedingung; Kosten der Garantie gegenüber M7a beziffert |
| **M8** Ausbau | Multi-Agent, Generalisierung über Netze/Szenarien, 1-min-Validierung | Transferergebnisse: Training auf Netz A, Test auf Netz B |

---

**Warum die P4-Vorabprüfung nach M2 gehört.** Die Frage, ob eine sichere Rückfallaktion überhaupt existiert (§6.9, P4), ist eine reine Zeitreihensimulation: PV vollständig abgeregelt, flexible Lasten auf Minimum, nur Grundlast — und dann prüfen, ob die Grenzen für alle Szenarien eingehalten werden. Das kostet in M2 praktisch nichts und beantwortet vorab, ob eine harte Garantie in den geplanten Ausbauszenarien erreichbar ist. Fällt die Prüfung negativ aus, ist das kein Rückschlag, sondern die Information, dass in diesem Szenario kein Regelverfahren den Netzausbau ersetzen kann — und sie sollte vorliegen, bevor Aufwand in M7b fließt.

## 12. Decision Record

Stand nach der ersten Abstimmungsrunde. Jede Entscheidung mit ihrer wesentlichen architektonischen Folge.

| # | Frage | Entscheidung | Folge im Dokument |
|---|---|---|---|
| D1 | Ein Agent oder mehrere? | **Einzelagent zuerst**, Multi-Agent offenhalten | §1.2 (Prinzip 7), §6.3 mit vier verbindlichen Regeln und Äquivalenztest ab M3 |
| D2 | Spannungskriterium | **EN-50160-Perzentilkriterium** (K95 auf 10-min-Mittelwerten je Woche, K100 mit −15 %) | §6.6 komplett neu; betrifft §3.3, §4, §6.2, §6.4, §6.5, §7.3, §8.3 |
| D3 | Wärmepumpenmodell | **Stufe 1: Pufferspeicher** für M4, `ThermalModel`-Protokoll lässt 1R1C später zu | §5 |
| D4 | Blindleistung | Q als **Aktionsoption implementiert**, Hauptstudie P-dominiert; Q bleibt für B3 zwingend nötig | §5 (`pv.mode ∈ {p_only, q_only, pq}`, Default `p_only`) |
| D5 | Reward-Gewichtung | **Feste Gewichte zuerst**, Constrained RL mit Lagrange-Multiplikator als vollwertige Alternative | §6.4 mit `mode: fixed_weights \| lagrangian`, Kategorien `objective`/`constraint` |
| D6 | Prognose | **Fehlermodell mit `perfect` als Sonderfall** | §6.7 |
| D7 | Anschlusspunkte | Busse mit **Last- *oder* Erzeuger-/Speicherelement**; Ausnahmeliste für Ersatzeinspeiser; „alle Busse" als Sensitivitätsvariante | §4 |
| D8 | Safety-Layer | Nicht ein Schalter, sondern **drei orthogonale Achsen** (Zulässigkeitsmodell × Mechanismus × Lernkopplung) plus eigene KPIs | §6.8, Roadmap M7a |
| D9 | Garantieniveau | **Harte, bedingte Garantie** als Zielbild über robuste konvexe Restriktion + prädiktiven Sicherheitsfilter; Agent gilt als nicht vertrauenswürdig | §6.9, §1.2 (Prinzip 8) |
| D10 | Zeitpunkt der Zertifizierung | **Später, als eigenes Teilprojekt (M7b).** M3–M6 liefern ohne Shield verwertbare Ergebnisse. Die Erweiterbarkeit wird nicht behauptet, sondern durch **sieben getestete Invarianten** gesichert | §13, Roadmap M2/M7b |

### 12.1 Was D2 tatsächlich kostet

Der Wechsel vom harten Band zum Perzentilkriterium ist normativ korrekt und methodisch anspruchsvoller, als es aussieht. Vier Punkte, die bei der Umsetzung erfahrungsgemäß Zeit kosten:

1. **`sim_dt = 15 min` fällt weg.** Das war in v0.1 noch die Default-Annahme und die natürliche Schrittweite der SimBench-Zeitreihen. Ab jetzt ist 5 min Default, was bedeutet: SimBench-Profile müssen von 15 min auf 5 min **hoch**gerechnet werden. Das ist eine Interpolation und erzeugt keine echte Information — die Spitzenglättung der Originaldaten bleibt erhalten. Für Szenarien, in denen Kurzzeitdynamik relevant ist, müssen die 1-min-fähigen Quellen (WPuQ, HTW Berlin) verwendet werden. Diese Abhängigkeit zwischen Spannungskriterium und Datenauswahl war in v0.1 nicht sichtbar und ist der wichtigste Punkt dieser Revision.
2. **Der Reward ist terminal und nicht Markov'sch** ohne Budget-Zustand. Der Agent *muss* wissen, wie viel Toleranz noch übrig ist. Wer das vergisst, trainiert ein Problem mit verstecktem Zustand und wundert sich über instabiles Lernen.
3. **γ ist an den Wochenhorizont gebunden.** `γ = 0.99` ist bei 15-min-Regeltakt ein effektiver Horizont von ~25 Stunden — das Kriterium wäre unsichtbar.
4. **Die Referenzverfahren werden gemischt-ganzzahlig.** Das betrifft vor allem `mpc_oracle`, den wichtigsten Vergleichswert. Die CVaR-Relaxation ist der pragmatische Einstieg.

### 12.2 Neu geöffnete Fragen

1. ~~Welche Busse sind Netzanschlusspunkte?~~ **Entschieden (D7):** Vereinigung der Busse mit Last-, Erzeuger- oder Speicherelement, siehe §4. Offen bleibt nur die netzspezifische Ausnahmeliste für Ersatzeinspeiser — das ist Handarbeit je Netzdatensatz und Teil von M2.
2. **Gilt das Budget je Bus oder netzweit?** Normkonform ist je Anschlusspunkt. Das macht die KPI streng (der schlechteste Bus entscheidet) und den Budget-Zustand hochdimensional. Vorschlag: KPI je Bus, Beobachtung verdichtet (§6.6).
3. **K100 mit −15 % ist asymmetrisch.** Bei PV-dominierten Überspannungsproblemen ist die untere Grenze selten bindend, bei WP- und EV-dominierten Unterspannungsproblemen sehr wohl. Die Asymmetrie sollte in der Szenarienauswahl bewusst adressiert werden (mindestens ein unterspannungsdominiertes Szenario).
4. **Wie wird die Lagrange-Variante evaluiert?** Sie optimiert eine andere Zielfunktion als die Fixgewicht-Variante. Fairer Vergleich nur über die physikalischen KPIs und die Pareto-Darstellung Abregelung vs. `en50160_pass_rate`.
5. **Confounding im Safe-RL-Vergleich.** Der Masking-Arm erzwingt diskrete Aktionen und PPO (§6.8, Einschränkung 1). Offen ist, ob die Studie die Diskretisierung für *alle* Arme übernimmt (sauberer Vergleich, aber ein anderes Problem als das kontinuierliche) oder zwei getrennte Vergleichsebenen fährt. Vorschlag: zwei Ebenen, weil die kontinuierliche Formulierung die realistischere ist und nicht für die Vergleichbarkeit eines Mechanismus geopfert werden sollte.
6. ~~Ist `residual_violation_rate > 0` akzeptabel?~~ **Entschieden (D9): nein.** Zielarchitektur ist der zertifizierte Shield (§6.9). Neu offen sind daraus drei Punkte:
   - **Symmetrieannahme** — der zertifizierte Arm rechnet symmetrisch, Schieflast bleibt ohne Garantie (§6.9). Ist das für die Zielaussage der Arbeit tragfähig, oder muss die unsymmetrische Zertifizierung Teil des Vorhabens werden? Letzteres wäre ein eigenes Arbeitspaket von erheblichem Umfang.
   - **Messausstattung** — eine harte Garantie bei realistischer Sensorik erfordert eine zertifizierte Fehlerschranke der Zustandsschätzung. Welche Messinfrastruktur darf angenommen werden, und ist diese Annahme praktisch vertretbar?
   - **Umfang von M7b** — robuste konvexe Restriktion plus prädiktiver Sicherheitsfilter plus Falsifikationsinfrastruktur ist realistisch ein eigenes Teilprojekt, nicht ein Meilenstein neben anderen. Der Aufbau sollte so geschnitten werden, dass M3–M6 auch ohne fertigen Shield verwertbare Ergebnisse liefern (heuristische Arme, Referenzverfahren, Agentenvergleich) und M7b additiv ist.
7. **Kalibrierung der Prognosefehler** — belastbare σ(h)-Werte für PV und Last auf Einzelanlagen- bzw. Hausanschlussebene (nicht Portfolioebene, dort ist der Fehler durch Ausgleichseffekte viel kleiner). Ein eigener kleiner Rechercheschritt vor M6.

---

## 13. Erweiterbarkeits-Contract für die spätere Zertifizierung

Entscheidung D10 verschiebt die Zertifizierung, verlangt aber, dass die Architektur sie später aufnehmen kann. Diese Zusage ist nur belastbar, wenn sie geprüft wird: **Modularitätsversprechen verfallen still.** Sechs Monate ohne Test, und irgendeine Abkürzung hat die Erweiterbarkeit aufgebraucht, ohne dass es jemand gemerkt hätte.

Die folgenden sieben Invarianten sind deshalb jeweils mit einem Test in `tests/test_invariants.py` verbunden, der ab M2 in der CI läuft. Die letzte Spalte ist die eigentliche Begründung: alle sieben sind jetzt billig und später teuer bis unmöglich.

| | Invariante | Warum sie für M7b nötig ist | Test | Kosten der Nachrüstung |
|---|---|---|---|---|
| **I1** | `SystemState` ist vollständig und von `Observation` getrennt; der `ObservationBuilder` ist eine reine Projektion davon | Der Zertifizierer braucht den vollen Zustand, der Agent darf weniger sehen (§6.9). Wenn Beobachtung und Zustand dasselbe Objekt sind, ist das nicht trennbar | Roundtrip-Test: aus `SystemState` lässt sich jede Feature-Gruppe rekonstruieren; `Observation` hat keine Setter | hoch — betrifft jede Env-Komponente |
| **I2** | Anlagendynamik als **reine Funktion** über kopierbaren `AssetState` | Der prädiktive Sicherheitsfilter muss Zustände hypothetisch über einen Horizont fortschreiben | Determinismus- und Seiteneffektfreiheits-Test: zweimaliger Aufruf mit gleicher Eingabe liefert identische Ausgabe; Eingabeobjekte bleiben unverändert | **sehr hoch** — Neuschreiben aller Anlagenmodelle |
| **I3** | Strikte Informationsordnung: Aktionsbildung nutzt nur `InformationSet(t)` | Eine Garantie auf Basis von Information, die der reale Regler nicht hat, ist keine | Zugriffs-Spion auf dem Szenario-Objekt: jeder Zugriff auf Zeitpunkte `> t` innerhalb der Aktionsbildung schlägt fehl | **sehr hoch** — Fehler ist diffus verteilt und kaum auffindbar |
| **I4** | Aktionen in physikalischen Einheiten, Normierung affin und invertierbar, jede Begrenzung explizit gemeldet | Die zertifizierte Zulässigkeitsmenge ist in Einspeisungen formuliert; Projektion braucht boxförmige Aktionskoordinaten | Invertierbarkeitstest über Zufallsaktionen; Test, dass kein Anlagenmodell still begrenzt | mittel bis hoch — ändert Aktionsraum und damit alle Trainingsergebnisse |
| **I5** | `PowerFlowEngine.run` ist hypothetisch aufrufbar, ohne das laufende Netz zu verändern | Rückfalltrajektorien und Falsifikationssuche brauchen genau das | Snapshot-Vergleich: nach hypothetischem Aufruf ist das Live-Netz bitidentisch | mittel |
| **I6** | Exogene Eingänge werden als `ExogenousInput` mit Feldern `realized` **und** `bounds` geführt; `ratings` je Anlage sind gepflegt | Unsicherheitsmengen (P2) brauchen Schranken, nicht Punktwerte. `bounds` wird zunächst trivial aus `ratings` befüllt und von nichts benutzt — die Verrohrung existiert aber | Schema-Test: kein Datenpfad liefert `ExogenousInput` ohne `bounds`; jede Anlage hat vollständige `ratings` | mittel; die `ratings` nachzutragen ist reine Handarbeit über alle Szenarien |
| **I7** | Zwei Sicherheits-Eingriffspunkte existieren mit Null-Implementierung: `env.safety` und `info["action_mask"]` | Der Shield sitzt innerhalb der Env (braucht Zustand), Masking außerhalb (braucht Verteilung). Beide Pfade müssen vorhanden sein | Test, dass eine Dummy-`SafetyComponent`, die Aktionen halbiert, tatsächlich greift und die Maske bei `none` alles erlaubt | niedrig bis mittel |

Zusätzlich zwei Punkte, die keine Invariante sind, aber dieselbe Funktion haben:

- **Phasenmodell als Konfigurationsschalter.** `grid.phase_model: balanced | unbalanced` ab M2, Default `balanced`. Die verfügbaren Zertifizierungsformulierungen sind symmetrisch (§6.9); wird Schieflast fest verdrahtet, ist der zertifizierte Arm später nicht lauffähig, wird Symmetrie fest verdrahtet, ist die Schieflast-Sensitivitätsstudie nicht möglich.
- **Einheitenkanon.** Eine einzige interne Repräsentation (SI oder durchgängig p.u.), im `SystemState`-Schema dokumentiert und per Test geprüft. pandapower mischt MW und p.u.; Intervallrechnungen in der Zertifizierung verzeihen keine versteckten Konversionen.

### 13.1 Was der Contract *nicht* leistet

Er hält die Struktur offen, nicht die Ergebnisse. Wenn M7b später den Aktionsraum einschränkt — und das ist der Zweck eines Shields —, ändern sich die Trainingsergebnisse aus M3–M6 und müssen erneut gerechnet werden. Das ist kein Architekturfehler, sondern die Natur der Sache: der Agent optimiert dann in einer anderen, kleineren Menge. Die Invarianten sichern zu, dass dieser erneute Durchlauf eine Konfigurationsänderung ist und keine Neuimplementierung.

Ebenfalls nicht abgedeckt: falls sich in M7b herausstellt, dass die konvexe Restriktion für die vorliegenden Netzgrößen zu langsam ist, kann eine Modellreduktion nötig werden (Aggregation von Strängen). Das berührt L1 und ist mit den Invarianten allein nicht abzufangen — es lohnt sich daher, in M2 nebenbei die Laufzeit eines SOCP auf einem der Zielnetze einmal zu messen, um diese Unsicherheit früh zu beziffern.
