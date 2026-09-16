"""Loading and preparing pandapower grids from SimBench.

Contains one non-obvious repair that is required for the combination of
simbench 1.6.x with pandapower 3.x, see :func:`fix_zip_load_model`.

The loader also derives the set of grid connection points (decision D7):
a bus counts as a connection point if at least one customer-side element is
attached to it -- load, generator **or** storage. Generators are not optional
here: a PV system or a battery on a bus without load is just as much a point of
common coupling as a house connection, and such buses tend to sit at the feeder
end, exactly where the voltage criterion becomes binding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from lvgrid_rl.grid.scenario import (
    SCENARIO_LIBRARY,
    ScenarioSpec,
    apply_scenario,
)

if TYPE_CHECKING:  # pragma: no cover
    from pandapower.auxiliary import pandapowerNet

__all__ = [
    "ZIP_COLUMNS",
    "CONNECTION_POINT_TABLES",
    "fix_zip_load_model",
    "GridModel",
    "load_grid",
]

ZIP_COLUMNS: Final[tuple[str, ...]] = (
    "const_z_p_percent",
    "const_i_p_percent",
    "const_z_q_percent",
    "const_i_q_percent",
)
"""Columns of the voltage-dependent (ZIP) load model in pandapower."""

CONNECTION_POINT_TABLES: Final[tuple[str, ...]] = (
    "load",
    "sgen",
    "gen",
    "storage",
    "asymmetric_load",
    "asymmetric_sgen",
)
"""Element tables whose buses count as grid connection points (decision D7)."""


def fix_zip_load_model(net: pandapowerNet) -> int:
    """Fill the ZIP load model columns, returning the number of repaired rows.

    **Why this exists.** simbench 1.6.1 leaves ``const_z_p_percent`` and its
    three siblings as ``NaN``. pandapower 3.x propagates those NaNs into the
    Jacobian, which becomes exactly singular, and the Newton-Raphson power flow
    fails to converge -- for *every* SimBench low-voltage grid, including the
    trivial base scenario. The failure mode is misleading: pandapower reports
    "Power Flow nr did not converge", which suggests an infeasible operating
    point rather than a data defect. The built-in diagnostic blames implausible
    impedance values, which is a red herring.

    The discriminating observation is that ``fdbx`` and ``fdxb`` converge while
    ``nr`` does not -- and those two algorithms ignore the voltage-dependent
    load model.

    Setting the columns to zero means a constant-power load, which is
    pandapower's own default for a newly created load and matches what SimBench
    intends. Should a future simbench release fill these columns, this function
    becomes a no-op because it only touches NaN entries.
    """
    repaired = 0
    for table in ("load", "asymmetric_load"):
        if table not in net or len(net[table]) == 0:
            continue
        for column in ZIP_COLUMNS:
            if column not in net[table].columns:
                net[table][column] = 0.0
                continue
            missing = net[table][column].isna()
            if missing.any():
                net[table].loc[missing, column] = 0.0
                repaired += int(missing.sum())
    return repaired


@dataclass(frozen=True, slots=True)
class GridModel:
    """A prepared grid together with the index caches needed on the hot path.

    Resolving pandas indices once at load time rather than per simulation step
    matters: a lookup per element per step would dominate the runtime of a
    training run with millions of steps.

    Args:
        code: SimBench code the grid was loaded from.
        net: The prepared pandapower grid.
        connection_point_buses: Buses carrying customer-side elements (D7).
        evaluated_bus_positions: Positional indices of those buses in
            ``net.res_bus``, for fast slicing of voltage results.
        zip_rows_repaired: Number of load rows repaired by
            :func:`fix_zip_load_model`; reported so the repair cannot happen
            unnoticed.
        scenario: The applied scenario specification.
        scenario_modifications: Counts of elements the scenario touched. A
            scenario that silently modifies nothing is a configuration error,
            so the counts are carried rather than discarded.
    """

    code: str
    net: pandapowerNet
    connection_point_buses: tuple[int, ...]
    evaluated_bus_positions: np.ndarray
    zip_rows_repaired: int
    scenario: ScenarioSpec
    scenario_modifications: Mapping[str, int]

    @property
    def n_evaluated_buses(self) -> int:
        """Number of buses assessed against the voltage criterion."""
        return len(self.connection_point_buses)


def derive_connection_points(
    net: pandapowerNet,
    exclude_buses: frozenset[int] = frozenset(),
) -> tuple[int, ...]:
    """Derive the set of grid connection points (decision D7).

    Args:
        net: The grid.
        exclude_buses: Buses to exclude, for instance equivalent infeeds that
            model an adjacent grid rather than a customer. Maintained per grid
            in ``configs/grid/*`` and never guessed from element names.
    """
    buses: set[int] = set()
    for table in CONNECTION_POINT_TABLES:
        if table not in net or len(net[table]) == 0:
            continue
        df = net[table]
        in_service = df["in_service"] if "in_service" in df.columns else True
        buses.update(int(b) for b in df.loc[in_service, "bus"])
    return tuple(sorted(buses - set(exclude_buses)))


def load_grid(
    code: str,
    scenario: ScenarioSpec | None = None,
    exclude_buses: frozenset[int] = frozenset(),
) -> GridModel:
    """Load a SimBench grid, apply a scenario and prepare it for simulation.

    Args:
        code: SimBench code, for example ``"1-LV-rural1--2-sw"``.
        scenario: Scenario modification. ``None`` means the published SimBench
            scenario unchanged.
        exclude_buses: Buses excluded from the connection point set.

    Returns:
        The prepared grid with its index caches.
    """
    import simbench

    net = simbench.get_simbench_net(code)
    repaired = fix_zip_load_model(net)
    spec = scenario or SCENARIO_LIBRARY["reference"]
    modified = apply_scenario(net, spec)
    buses = derive_connection_points(net, exclude_buses)
    positions = np.array([net.bus.index.get_loc(b) for b in buses], dtype=np.intp)
    return GridModel(
        code=code,
        net=net,
        connection_point_buses=buses,
        evaluated_bus_positions=positions,
        zip_rows_repaired=repaired,
        scenario=spec,
        scenario_modifications=modified,
    )
