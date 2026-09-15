"""Reproduzierbarkeit: Seeds, Konfigurationshash und Run-Manifest.

Ein Run ist durch ``(code_commit, config_hash, data_manifest_hash, seed)``
eindeutig bestimmt (§8.2 des Architekturdokuments). Dieses Modul erzeugt diese
Kennung und das zugehoerige Manifest.

Zwei Punkte, die in der Praxis den Unterschied machen:

* **Seeds sind nach Zweck getrennt.** Szenario, Episodenziehung, Training,
  Evaluation und Prognosefehler bekommen je einen eigenen, deterministisch
  abgeleiteten Seed. Sonst bedeutet "fuenf Seeds" versehentlich "fuenf
  verschiedene Netze", und der Vergleich zwischen Agenten ist entwertet.
* **Der Konfigurationshash ist kanonisch.** Gleiche Konfiguration mit anderer
  Schluesselreihenfolge muss denselben Hash ergeben, sonst zerfaellt die
  Run-Registry in Duplikate.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "SeedSet",
    "canonical_json",
    "config_hash",
    "compute_run_id",
    "RunManifest",
    "git_commit",
]

# Feste Reihenfolge der Seed-Zwecke. Anhaengen ist unkritisch, Umsortieren
# aendert alle abgeleiteten Seeds und damit die Reproduzierbarkeit
# bestehender Runs -- daher niemals umsortieren.
_SEED_PURPOSES: tuple[str, ...] = (
    "scenario",
    "episode",
    "train",
    "eval",
    "forecast",
)


@dataclass(frozen=True, slots=True)
class SeedSet:
    """Nach Zweck getrennte Seeds, deterministisch aus einem Basis-Seed.

    Die Ableitung nutzt :class:`numpy.random.SeedSequence`, deren
    ``spawn``-Mechanismus statistisch unabhaengige Kindsequenzen garantiert --
    im Gegensatz zu naiven Konstruktionen wie ``base + 1``, ``base + 2``, bei
    denen benachbarte Seeds korrelierte Streams erzeugen koennen.

    Example:
        >>> seeds = SeedSet.from_base(42)
        >>> seeds.scenario == SeedSet.from_base(42).scenario
        True
        >>> seeds.scenario == seeds.train
        False
    """

    base: int
    scenario: int
    episode: int
    train: int
    eval: int
    forecast: int

    @classmethod
    def from_base(cls, base: int) -> SeedSet:
        """Leitet alle Zweck-Seeds aus einem Basis-Seed ab."""
        children = np.random.SeedSequence(base).spawn(len(_SEED_PURPOSES))
        values = {
            purpose: int(child.generate_state(1, dtype=np.uint32)[0])
            for purpose, child in zip(_SEED_PURPOSES, children, strict=True)
        }
        return cls(base=base, **values)

    def generator(self, purpose: str) -> np.random.Generator:
        """Generator fuer einen Zweck.

        Raises:
            KeyError: bei unbekanntem Zweck -- Tippfehler sollen auffallen und
                nicht stillschweigend einen Default-Generator liefern.
        """
        if purpose not in _SEED_PURPOSES:
            raise KeyError(
                f"Unbekannter Seed-Zweck {purpose!r}. Bekannt: {_SEED_PURPOSES}"
            )
        return np.random.default_rng(getattr(self, purpose))

    def worker_generator(self, purpose: str, worker_index: int) -> np.random.Generator:
        """Generator fuer einen parallelen Environment-Worker.

        Jeder Worker braucht einen eigenen, aber reproduzierbaren Stream.
        """
        seq = np.random.SeedSequence([getattr(self, purpose), worker_index])
        return np.random.default_rng(seq)


def canonical_json(obj: Any) -> str:
    """Kanonische JSON-Darstellung: sortierte Schluessel, kompakte Trenner.

    >>> canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    True
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash der aufgeloesten Konfiguration, unabhaengig von der Reihenfolge."""
    return _sha256(canonical_json(config))


def git_commit(repo_root: Path | None = None) -> str:
    """Aktueller Commit, oder ``"unknown"`` ausserhalb eines Repositories.

    Ein unbekannter Commit ist kein Fehler -- Runs sollen auch aus einem
    Arbeitsverzeichnis heraus startbar sein --, wird aber im Manifest sichtbar
    und disqualifiziert den Run fuer Veroeffentlichungszwecke.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def compute_run_id(
    code_commit: str, config_hash_: str, data_manifest_hash: str, seed: int
) -> str:
    """Zwoelfstellige Run-Kennung aus den vier bestimmenden Groessen."""
    return _sha256(
        canonical_json([code_commit, config_hash_, data_manifest_hash, seed])
    )[:12]


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Vollstaendige Beschreibung eines Trainings- oder Evaluationslaufs.

    Wird als ``manifest.json`` in das Run-Verzeichnis geschrieben und ist die
    Grundlage der Run-Registry. Alles, was einen Run beeinflusst und nicht in
    der Konfiguration steht, gehoert hier hinein.
    """

    run_id: str
    code_commit: str
    config_hash: str
    data_manifest_hash: str
    seeds: SeedSet
    created_at: str
    python_version: str
    platform: str
    package_versions: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        config: Mapping[str, Any],
        data_manifest_hash: str,
        base_seed: int,
        package_versions: Mapping[str, str] | None = None,
        repo_root: Path | None = None,
        notes: str = "",
    ) -> RunManifest:
        """Erzeugt ein Manifest aus Konfiguration und Umgebung."""
        cfg_hash = config_hash(config)
        commit = git_commit(repo_root)
        return cls(
            run_id=compute_run_id(commit, cfg_hash, data_manifest_hash, base_seed),
            code_commit=commit,
            config_hash=cfg_hash,
            data_manifest_hash=data_manifest_hash,
            seeds=SeedSet.from_base(base_seed),
            created_at=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            package_versions=dict(package_versions or {}),
            notes=notes,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, run_dir: Path) -> Path:
        """Schreibt ``manifest.json`` und gibt den Pfad zurueck."""
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path
