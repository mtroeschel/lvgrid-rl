# Daten

Dieses Verzeichnis enthält **keine** Daten. Rohdaten und abgeleitete Caches
sind in `.gitignore` ausgeschlossen, weil die Weitergabebedingungen je Quelle
unterschiedlich sind und weil versionierte Binärdaten die Repository-Historie
unbrauchbar machen.

Reproduzierbarkeit läuft stattdessen über drei Dinge: die Bezugsskripte in
`scripts/prepare_data.py` (ab M1), die Konfiguration der Aufbereitung in
`configs/data/`, und ein `manifest.json` mit SHA-256 je Rohdatei. Wer dieselben
Rohdaten bezieht, erhält denselben Manifest-Hash und damit denselben
Verarbeitungsstand.

## Quellen

| Quelle | Auflösung | Verwendung | Bezug |
|---|---|---|---|
| SimBench | 15 min, 1 Jahr | Primärquelle M1–M4; Last, PV, Speicher, WP-1…WP-5 | https://simbench.de/en/download/datasets/ |
| WPuQ (ISFH) | 10 s / 1 / 15 / 60 min | ab M5; Haushalt und Wärmepumpe getrennt gemessen, plus Ortsnetzstation | DOI 10.5281/zenodo.5642902 |
| HTW Berlin | 1 s | ab M5; hochaufgelöste Haushaltslast, phasenaufgelöst | https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/ |
| emobpy | 15 min, konfigurierbar | EV-Verfügbarkeit und Ladebedarf | https://emobpy.readthedocs.io |
| ElaadNL | Verteilungen | Kalibrierung des EV-Session-Generators | https://elaad.nl/en/open-datasets/ |
| when2heat | 1 h | COP-Kennlinien und Jahresgang Wärmebedarf | https://data.open-power-system-data.org/when2heat/ |
| DWD Open Data | 10 min | Globalstrahlung, Temperatur | https://opendata.dwd.de/climate_environment/CDC/ |

## Zu klären vor der Veröffentlichung

Ob der abgeleitete 5-min-Cache weitergegeben werden darf, unterscheidet sich je
Quelle. Bis das je Datensatz geprüft ist, gilt: Dritte beziehen die Rohdaten
selbst, und der Manifest-Hash belegt, dass es dieselben sind.
