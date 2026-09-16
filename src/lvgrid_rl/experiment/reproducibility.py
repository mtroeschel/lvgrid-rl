"""Reproducibility: seeds, configuration hash and run manifest.

A run is uniquely determined by ``(code_commit, config_hash, data_manifest_hash,
seed)`` (section 8.2 of the architecture document). This module produces that
identifier and the accompanying manifest.

Two points make the difference in practice:

* **Seeds are separated by purpose.** Scenario, episode sampling, training,
  evaluation and forecast error each get their own deterministically derived
  seed. Otherwise "five seeds" accidentally means "five different grids", and
  the comparison between agents is void.
* **The configuration hash is canonical.** The same configuration with a
  different key order must give the same hash, or the run registry decays into
  duplicates.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SeedSet",
    "canonical_json",
    "config_hash",
    "compute_run_id",
    "RunManifest",
    "git_commit",
]

# Fixed order of the seed purposes. Appending is harmless; reordering changes
# every derived seed and therefore the reproducibility of existing runs -- so
# never reorder.
_SEED_PURPOSES: tuple[str, ...] = (
    "scenario",
    "episode",
    "train",
    "eval",
    "forecast",
)


@dataclass(frozen=True, slots=True)
class SeedSet:
    """Purpose-separated seeds, derived deterministically from a base seed.

    Derivation uses :class:`numpy.random.SeedSequence`, whose ``spawn``
    mechanism guarantees statistically independent child sequences -- unlike
    naive constructions such as ``base + 1``, ``base + 2``, where adjacent seeds
    can produce correlated streams.

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
        """Derive all purpose seeds from a base seed."""
        children = np.random.SeedSequence(base).spawn(len(_SEED_PURPOSES))
        values = {
            purpose: int(child.generate_state(1, dtype=np.uint32)[0])
            for purpose, child in zip(_SEED_PURPOSES, children, strict=True)
        }
        return cls(base=base, **values)

    def generator(self, purpose: str) -> np.random.Generator:
        """Generator for one purpose.

        Raises:
            KeyError: on an unknown purpose -- typos should surface rather than
                silently yield a default generator.
        """
        if purpose not in _SEED_PURPOSES:
            raise KeyError(f"Unknown seed purpose {purpose!r}. Known: {_SEED_PURPOSES}")
        return np.random.default_rng(getattr(self, purpose))

    def worker_generator(self, purpose: str, worker_index: int) -> np.random.Generator:
        """Generator for a parallel environment worker.

        Each worker needs its own stream, but a reproducible one.
        """
        seq = np.random.SeedSequence([getattr(self, purpose), worker_index])
        return np.random.default_rng(seq)


def canonical_json(obj: Any) -> str:
    """Canonical JSON representation: sorted keys, compact separators.

    >>> canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    True
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash of the resolved configuration, independent of key order."""
    return _sha256(canonical_json(config))


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit, or ``"unknown"`` outside a repository.

    An unknown commit is not an error -- runs should be startable from a working
    directory -- but it becomes visible in the manifest and disqualifies the run
    for publication purposes.
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
    """Twelve-character run identifier from the four determining quantities."""
    return _sha256(canonical_json([code_commit, config_hash_, data_manifest_hash, seed]))[
        :12
    ]


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Complete description of a training or evaluation run.

    Written as ``manifest.json`` into the run directory and the basis of the run
    registry. Everything that influences a run and is not in the configuration
    belongs here.
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
        """Build a manifest from configuration and environment."""
        cfg_hash = config_hash(config)
        commit = git_commit(repo_root)
        return cls(
            run_id=compute_run_id(commit, cfg_hash, data_manifest_hash, base_seed),
            code_commit=commit,
            config_hash=cfg_hash,
            data_manifest_hash=data_manifest_hash,
            seeds=SeedSet.from_base(base_seed),
            created_at=datetime.now(UTC).isoformat(),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            package_versions=dict(package_versions or {}),
            notes=notes,
        )

    def to_json(self) -> str:
        """Manifest as indented JSON with sorted keys."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, run_dir: Path) -> Path:
        """Write ``manifest.json`` and return its path."""
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path
