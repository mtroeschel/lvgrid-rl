"""Aufbereitungs-Cache und Datenmanifest.

Der Cache liegt als Parquet unter einem Schluessel, der aus allen Groessen
gebildet wird, die das Ergebnis beeinflussen: Quelle und Version, Zeitraum,
Schrittweite, Resampling-Regeln und Pipeline-Version. Dieselbe Eingabe ergibt
denselben Schluessel, eine geaenderte Regel einen anderen -- ein stiller
Treffer auf veralteten Daten ist damit ausgeschlossen.

Das Manifest haelt zusaetzlich einen SHA-256 ueber den *Inhalt* der
aufbereiteten Profile. Das ist der Nachweis, auf den sich die
Reproduzierbarkeitskette stuetzt (Abschnitt 8.2): wer dieselben Rohdaten
bezieht und dieselbe Konfiguration verwendet, erhaelt denselben Hash. Da das
Repository bewusst keine Daten enthaelt (``data/README.md``), ist dieser Hash
der Beleg, dass Dritte tatsaechlich mit denselben Zeitreihen rechnen.

Fuer SimBench gibt es keine heruntergeladene Rohdatei, die sich hashen liesse --
die Daten kommen aus dem installierten Paket. Gehasht wird deshalb der
materialisierte Profil-Frame, und die Paketversion geht in den Schluessel ein.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["PIPELINE_VERSION", "CacheKey", "DataManifest", "ProfileCache"]

PIPELINE_VERSION = "1"
"""Bei jeder Aenderung an der Aufbereitungslogik erhoehen.

Die Version geht in den Cache-Schluessel ein. Ohne sie wuerde eine korrigierte
Resampling-Regel auf bereits zwischengespeicherten Ergebnissen unbemerkt
wirkungslos bleiben.
"""


def frame_hash(df: pd.DataFrame) -> str:
    """SHA-256 ueber Inhalt, Spaltennamen und Zeitachse eines DataFrame.

    Bewusst nicht ueber die Parquet-Datei: deren Bytes haengen von
    Bibliotheksversion und Kompression ab, der Inhalt nicht.
    """
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode("utf-8"))
    h.update(str(df.index[0]).encode("utf-8"))
    h.update(str(df.index[-1]).encode("utf-8"))
    h.update(str(len(df.index)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Alle Groessen, die das Aufbereitungsergebnis bestimmen.

    Args:
        source: Name der Quelle, etwa ``"simbench"``.
        source_version: Version des Quellpakets oder des Datensatzes.
        dataset: Bezeichner innerhalb der Quelle, etwa der SimBench-Code.
        sim_dt_min: Zielschrittweite.
        policies: Angewandte Resampling-Regeln.
        pipeline_version: :data:`PIPELINE_VERSION`.
    """

    source: str
    source_version: str
    dataset: str
    sim_dt_min: int
    policies: Mapping[str, str] = field(default_factory=dict)
    pipeline_version: str = PIPELINE_VERSION

    def digest(self) -> str:
        """Zwoelfstelliger Schluessel, geeignet als Dateiname."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def filename(self) -> str:
        """Dateiname des Cache-Eintrags."""
        safe = self.dataset.replace("/", "_")
        return f"{self.source}_{safe}_{self.sim_dt_min}min_{self.digest()}.parquet"


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Nachweis ueber die verwendeten Daten.

    Args:
        key: Der Cache-Schluessel.
        content_hash: SHA-256 ueber den aufbereiteten Profil-Frame.
        n_rows: Anzahl Zeitschritte.
        n_columns: Anzahl Profilspalten.
        start_utc: Erster Zeitstempel.
        end_utc: Letzter Zeitstempel.
        notes: Freitext, etwa dokumentierte Annahmen der Aufbereitung.
    """

    key: CacheKey
    content_hash: str
    n_rows: int
    n_columns: int
    start_utc: str
    end_utc: str
    notes: str = ""

    def to_json(self) -> str:
        """Manifest als eingeruecktes JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @property
    def short_hash(self) -> str:
        """Erste zwoelf Stellen des Inhalts-Hash, fuer Logausgaben."""
        return self.content_hash[:12]


class ProfileCache:
    """Parquet-Cache fuer aufbereitete Profile.

    Args:
        root: Verzeichnis des Caches. Wird bei Bedarf angelegt und ist in
            ``.gitignore`` ausgeschlossen.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: CacheKey) -> Path:
        """Pfad des Cache-Eintrags zu einem Schluessel."""
        return self.root / key.filename()

    def manifest_path_for(self, key: CacheKey) -> Path:
        """Pfad des zugehoerigen Manifests."""
        return self.path_for(key).with_suffix(".manifest.json")

    def has(self, key: CacheKey) -> bool:
        """Liegt ein Eintrag zu diesem Schluessel vor?"""
        return self.path_for(key).exists()

    def store(
        self, key: CacheKey, profiles: pd.DataFrame, notes: str = ""
    ) -> DataManifest:
        """Legt Profile und Manifest ab.

        Returns:
            Das geschriebene Manifest.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        profiles.to_parquet(self.path_for(key), compression="zstd")
        manifest = DataManifest(
            key=key,
            content_hash=frame_hash(profiles),
            n_rows=len(profiles.index),
            n_columns=len(profiles.columns),
            start_utc=str(profiles.index[0]),
            end_utc=str(profiles.index[-1]),
            notes=notes,
        )
        self.manifest_path_for(key).write_text(manifest.to_json(), encoding="utf-8")
        return manifest

    def load(self, key: CacheKey) -> tuple[pd.DataFrame, DataManifest]:
        """Laedt Profile und Manifest und prueft den Inhalts-Hash.

        Raises:
            FileNotFoundError: wenn kein Eintrag vorliegt.
            ValueError: wenn der Inhalts-Hash nicht zum Manifest passt. Das
                bedeutet, dass die Datei nach dem Schreiben veraendert wurde,
                und entwertet jede darauf gestuetzte Reproduzierbarkeitsaussage.
        """
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"Kein Cache-Eintrag unter {path}")
        profiles = pd.read_parquet(path)
        raw: dict[str, Any] = json.loads(
            self.manifest_path_for(key).read_text(encoding="utf-8")
        )
        manifest = DataManifest(
            key=CacheKey(**raw["key"]),
            content_hash=raw["content_hash"],
            n_rows=raw["n_rows"],
            n_columns=raw["n_columns"],
            start_utc=raw["start_utc"],
            end_utc=raw["end_utc"],
            notes=raw.get("notes", ""),
        )
        actual = frame_hash(profiles)
        if actual != manifest.content_hash:
            raise ValueError(
                f"Inhalts-Hash von {path.name} weicht vom Manifest ab "
                f"({actual[:12]} statt {manifest.short_hash})."
            )
        return profiles, manifest
