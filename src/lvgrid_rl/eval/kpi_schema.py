"""Versioning of the KPI definitions in result files.

Deliberately free of dependencies, so that the aggregation scripts can check a
file without importing the grid and environment stack.
"""

from __future__ import annotations

__all__ = ["KPI_SCHEMA", "check_kpi_schema"]

KPI_SCHEMA: int = 2
"""Version of the KPI definitions in result files.

1 -- unstamped files up to M4.0: ``overload_cost`` is three times the percent-hour
integral / 100 at 5 on 15 minutes (docs/results/m3.md, correction note).
2 -- ``overload_cost`` is the integral itself.

Every script that writes a result file stamps it, and every script that combines
files checks the stamp. A table mixing the two would be off by a factor of three
in one column and look entirely plausible.
"""


def check_kpi_schema(payload: dict, source: object) -> None:
    """Refuse a result file whose KPI definitions differ from the current ones.

    Raises:
        ValueError: if the file carries another schema, or none (schema 1).
    """
    found = payload.get("kpi_schema", 1)
    if found != KPI_SCHEMA:
        raise ValueError(
            f"{source} has KPI schema {found}, the code {KPI_SCHEMA}. Schema 1 "
            "files (up to M4.0) state overload_cost three times the integral; "
            "divide by the inner steps per control step (3) or recompute, and "
            "do not mix the two in one table."
        )
