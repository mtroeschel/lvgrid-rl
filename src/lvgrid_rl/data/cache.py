"""Preparation cache and data manifest.

The cache is stored as Parquet under a key formed from everything that
influences the result: source and version, period, step size, resampling rules
and pipeline version. The same input yields the same key, a changed rule a
different one -- a silent hit on stale data is therefore impossible.

The manifest additionally holds a SHA-256 over the *content* of the prepared
profiles. This is the evidence the reproducibility chain rests on (section 8.2):
whoever obtains the same raw data and uses the same configuration gets the same
hash. Since the repository deliberately contains no data (``data/README.md``),
that hash is the proof that third parties really compute with the same series.

For SimBench there is no downloaded raw file to hash -- the data comes from the
installed package. What is hashed is therefore the materialised profile frame,
and the package version enters the key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PIPELINE_VERSION",
    "frame_hash",
    "CacheKey",
    "DataManifest",
    "ProfileCache",
]

PIPELINE_VERSION = "2"
"""Increment on every change to the preparation logic.

The version enters the cache key. Without it, a corrected resampling rule would
silently have no effect on already cached results.

Version 2 adds the reactive power profiles, which changes the set of columns and
therefore the content hash.
"""


def frame_hash(df: pd.DataFrame) -> str:
    """SHA-256 over content, column names and time axis of a frame.

    Deliberately not over the Parquet file: its bytes depend on library version
    and compression, the content does not.
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
    """Everything that determines the preparation result.

    Args:
        source: Name of the source, for example ``"simbench"``.
        source_version: Version of the source package or dataset.
        dataset: Identifier within the source, e.g. the SimBench code.
        sim_dt_min: Target step size.
        policies: Resampling rules applied.
        pipeline_version: :data:`PIPELINE_VERSION`.
    """

    source: str
    source_version: str
    dataset: str
    sim_dt_min: int
    policies: Mapping[str, str] = field(default_factory=dict)
    pipeline_version: str = PIPELINE_VERSION

    def digest(self) -> str:
        """Twelve-character key, usable as a file name."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def filename(self) -> str:
        """File name of the cache entry."""
        safe = self.dataset.replace("/", "_")
        return f"{self.source}_{safe}_{self.sim_dt_min}min_{self.digest()}.parquet"


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Evidence about the data that was used.

    Args:
        key: The cache key.
        content_hash: SHA-256 over the prepared profile frame.
        n_rows: Number of time steps.
        n_columns: Number of profile columns.
        start_utc: First timestamp.
        end_utc: Last timestamp.
        notes: Free text, e.g. documented assumptions of the preparation.
    """

    key: CacheKey
    content_hash: str
    n_rows: int
    n_columns: int
    start_utc: str
    end_utc: str
    notes: str = ""

    def to_json(self) -> str:
        """Manifest as indented JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @property
    def short_hash(self) -> str:
        """First twelve characters of the content hash, for log output."""
        return self.content_hash[:12]


class ProfileCache:
    """Parquet cache for prepared profiles.

    Args:
        root: Cache directory. Created on demand and excluded in
            ``.gitignore``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: CacheKey) -> Path:
        """Path of the cache entry for a key."""
        return self.root / key.filename()

    def manifest_path_for(self, key: CacheKey) -> Path:
        """Path of the associated manifest."""
        return self.path_for(key).with_suffix(".manifest.json")

    def has(self, key: CacheKey) -> bool:
        """Is there an entry for this key?"""
        return self.path_for(key).exists()

    def store(
        self, key: CacheKey, profiles: pd.DataFrame, notes: str = ""
    ) -> DataManifest:
        """Store profiles and manifest.

        Returns:
            The manifest that was written.
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
        """Load profiles and manifest and verify the content hash.

        Raises:
            FileNotFoundError: if there is no entry.
            ValueError: if the content hash does not match the manifest. That
                means the file was altered after writing, which voids every
                reproducibility statement resting on it.
        """
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"No cache entry at {path}")
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
                f"Content hash of {path.name} differs from the manifest "
                f"({actual[:12]} instead of {manifest.short_hash})."
            )
        return profiles, manifest
