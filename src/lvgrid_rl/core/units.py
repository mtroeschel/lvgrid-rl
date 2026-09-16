"""Unit and sign conventions of the project.

This module contains no logic, only a commitment. It is deliberately the first
module of the project, because two classes of mistake that originate here are
extremely expensive later:

1. **Mixed units.** pandapower uses MW/MVar for power but per-unit for voltage.
   Introducing kW or volts on top of that creates hidden conversions. For the
   certification planned later (section 6.9 of the architecture) this is fatal,
   because interval arithmetic does not forgive unnoticed factors.
2. **Mixed sign conventions.** pandapower uses the consumer reference direction
   for ``load`` (``p_mw > 0`` means consumption) and the generator reference
   direction for ``sgen`` (``p_mw > 0`` means generation). A battery is both. If
   that distinction travels through the codebase, it will eventually cost days.

The commitment is enforced by ``tests/test_core.py``: every numeric field of a
schema type must carry a known unit suffix.

See architecture document, section 13 ("unit convention").
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------

SIGN_CONVENTION: Final[str] = "consumer"
"""Consumer reference direction throughout, **for all** asset types.

``p_mw > 0`` always means drawing power from the grid, ``p_mw < 0`` always means
feeding into the grid. It follows that:

===================  ===========================================
Asset                Value range
===================  ===========================================
Household load       ``p_mw >= 0``
PV system            ``p_mw <= 0``
Heat pump            ``p_mw >= 0``
Charge point (no V2G) ``p_mw >= 0``
Battery storage      ``p_mw > 0`` charging, ``p_mw < 0`` discharging
===================  ===========================================

Conversion to the pandapower convention happens exclusively in the grid adapter
(``lvgrid_rl.grid``) and nowhere else. That adapter is therefore the only place
where a sign error can arise, and it is small enough to test exhaustively.
"""

# ---------------------------------------------------------------------------
# Unit convention
# ---------------------------------------------------------------------------

UNIT_SUFFIXES: Final[dict[str, str]] = {
    # Power and energy
    "_mw": "active power in MW (consumer reference, see SIGN_CONVENTION)",
    "_mvar": "reactive power in MVar (consumer reference)",
    "_mva": "apparent power in MVA",
    "_mwh": "energy in MWh",
    "_kwh": "energy in kWh (KPI output only, never in state)",
    # Electrical quantities
    "_pu": "per-unit quantity",
    "_kv": "voltage in kV (nominal values, not state quantities)",
    "_a": "current in A",
    "_percent": "loading in percent (0..100, not 0..1)",
    # Thermal and weather
    "_degc": "absolute temperature in degrees Celsius",
    "_k": "temperature difference in kelvin",
    "_kh": "comfort deviation in kelvin-hours",
    "_wm2": "irradiance in W/m^2",
    # Time
    "_s": "duration in seconds",
    "_min": "duration in minutes",
    # Dimensionless
    "_frac": "dimensionless fraction in the interval [0, 1]",
    "_count": "count (integer)",
}
"""Permitted unit suffixes for numeric fields of schema types."""

_SUFFIXES_BY_LENGTH: Final[tuple[str, ...]] = tuple(
    sorted(UNIT_SUFFIXES, key=len, reverse=True)
)


def unit_of(field_name: str) -> str | None:
    """Return the unit suffix of a field name, or ``None``.

    The search runs from the longest to the shortest suffix so that ``_mwh`` is
    not mistakenly recognised as ``_mw``.

    >>> unit_of("p_slack_mw")
    '_mw'
    >>> unit_of("energy_mwh")
    '_mwh'
    >>> unit_of("something_else")
    """
    for suffix in _SUFFIXES_BY_LENGTH:
        if field_name.endswith(suffix):
            return suffix
    return None


def describe_unit(field_name: str) -> str:
    """Describe the unit of a field name, for error messages."""
    suffix = unit_of(field_name)
    if suffix is None:
        return f"{field_name}: no known unit"
    return f"{field_name}: {UNIT_SUFFIXES[suffix]}"
