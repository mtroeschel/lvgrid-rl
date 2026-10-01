#!/usr/bin/env python3
"""Download the when2heat data package and verify it.

    uv run python scripts/fetch_when2heat.py

Writes ``data/raw/when2heat/when2heat-<version>.csv`` (about 330 MB) and checks
its SHA-256 against the value recorded in
``lvgrid_rl.data.sources.when2heat``. The raw data stays out of the repository
(``data/README.md``); licence CC-BY 4.0, cite Ruhnau, Hirth, Praktiknjo (2019).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from lvgrid_rl.data.sources.when2heat import (
    WHEN2HEAT_SHA256,
    WHEN2HEAT_URL,
    WHEN2HEAT_VERSION,
    _sha256,
    read_cop,
)


def main() -> None:
    """Download if missing, verify, and report the repaired hours."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/when2heat"))
    args = parser.parse_args()

    target = args.raw_dir / f"when2heat-{WHEN2HEAT_VERSION}.csv"
    if not target.exists():
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {WHEN2HEAT_URL}")
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(WHEN2HEAT_URL, partial)  # noqa: S310 - fixed https URL
        partial.rename(target)
    digest = _sha256(target)
    if digest != WHEN2HEAT_SHA256:
        raise SystemExit(f"checksum mismatch: {digest} != {WHEN2HEAT_SHA256}")
    cop = read_cop(target, verify=False)
    print(f"verified: {target} ({WHEN2HEAT_VERSION})")
    print(f"  period UTC: {cop.frame.index[0]} -> {cop.frame.index[-1]}")
    print(f"  filled autumn daylight-saving hours: {len(cop.filled_utc)}")


if __name__ == "__main__":
    main()
