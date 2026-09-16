"""Configurable scenario modifications on top of a SimBench grid.

Why this module exists: a survey of all six SimBench low-voltage grids at
development stage 2 produced **no voltage band violation at all** -- the worst
value across the year is about 1.083 pu against a limit of 1.10, and EN 50160
averages over ten minutes on top of that. What does bind is thermal: 245 %
transformer loading in ``1-LV-rural1``, 123 % line loading in
``1-LV-semiurb4``. Reproduce it with ``scripts/survey_grids.py``.

The published SimBench scenarios are therefore a sound starting point but not a
sufficient test bed for a controller whose stated purpose includes voltage band
violations. This module provides the two levers that change that, both
declarative and both logged, so that a scenario remains reproducible and its
distance from the published reference stays visible:

* **the point of common coupling** -- slack voltage and transformer tap. This is
  the physically more honest lever: a weaker upstream connection or an
  unfavourable tap position produces voltage problems without inventing
  installed capacity that nobody has.
* **penetration and sizing** -- scaling factors per asset category, plus
  optional replication of existing assets.

Deliberately not a random scenario generator. Every modification is an explicit
number in the configuration, because a scenario that cannot be written down
cannot be reported in a paper either.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lvgrid_rl.data.sources.simbench import AssetCategory, categorize

if TYPE_CHECKING:  # pragma: no cover
    from pandapower.auxiliary import pandapowerNet

__all__ = ["ScenarioSpec", "apply_scenario", "SCENARIO_LIBRARY"]


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Declarative description of a scenario modification.

    Args:
        name: Identifier used in run manifests and result tables.
        slack_vm_pu: Voltage setpoint at the external grid. SimBench ships
            1.025 pu for the low-voltage grids, which sits high and leaves
            little headroom towards the upper limit while making undervoltage
            practically unreachable. ``None`` keeps the dataset value.
        trafo_tap_pos: Tap position of the transformer. ``None`` keeps the
            dataset value. Raising the low-voltage side lifts the whole feeder.
        scale: Multiplicative factors on nominal power per asset category.
            A factor of 2.0 for :attr:`AssetCategory.PV` doubles every PV
            system's rating without adding assets, which keeps the spatial
            distribution of the published scenario intact.
        trafo_sn_scale: Factor on transformer rating. Below 1.0 this models a
            weaker station; it is the cleanest way to produce thermal stress
            without touching customer installations.
        notes: Free text, carried into the manifest. Use it to record why a
            scenario deviates from the published reference.

    Example:
        >>> spec = ScenarioSpec(name="pv_stress", scale={AssetCategory.PV: 2.0})
        >>> spec.scale[AssetCategory.PV]
        2.0
    """

    name: str
    slack_vm_pu: float | None = None
    trafo_tap_pos: float | None = None
    scale: Mapping[AssetCategory, float] = field(default_factory=dict)
    trafo_sn_scale: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.slack_vm_pu is not None and not 0.9 <= self.slack_vm_pu <= 1.1:
            raise ValueError(
                f"slack_vm_pu={self.slack_vm_pu} is outside a plausible range. "
                "Values beyond +/-10 % describe a grid that is already in "
                "violation before any control acts."
            )
        if self.trafo_sn_scale <= 0.0:
            raise ValueError("trafo_sn_scale must be positive")
        for category, factor in self.scale.items():
            if factor < 0.0:
                raise ValueError(f"Negative scaling factor for {category}: {factor}")

    def describe(self) -> dict[str, object]:
        """Flat description for the run manifest."""
        return {
            "name": self.name,
            "slack_vm_pu": self.slack_vm_pu,
            "trafo_tap_pos": self.trafo_tap_pos,
            "trafo_sn_scale": self.trafo_sn_scale,
            "scale": {k.value: v for k, v in sorted(self.scale.items())},
            "notes": self.notes,
        }


def apply_scenario(net: pandapowerNet, spec: ScenarioSpec) -> dict[str, int]:
    """Apply a scenario to a loaded grid, in place.

    Returns:
        Counts of modified elements per category, for logging. A scenario that
        silently modifies nothing -- because a category is absent from the grid
        -- is a configuration error that should be visible, not guessed at.
    """
    touched: dict[str, int] = {}

    if spec.slack_vm_pu is not None and len(net.ext_grid):
        net.ext_grid.loc[:, "vm_pu"] = spec.slack_vm_pu
        touched["ext_grid"] = len(net.ext_grid)

    if spec.trafo_tap_pos is not None and len(net.trafo):
        net.trafo.loc[:, "tap_pos"] = spec.trafo_tap_pos
        touched["trafo_tap"] = len(net.trafo)

    if spec.trafo_sn_scale != 1.0 and len(net.trafo):
        net.trafo.loc[:, "sn_mva"] = net.trafo["sn_mva"] * spec.trafo_sn_scale
        touched["trafo_sn"] = len(net.trafo)

    for table in ("load", "sgen", "storage"):
        if table not in net or len(net[table]) == 0:
            continue
        df = net[table]
        if "profile" not in df.columns:
            continue
        categories = df["profile"].fillna("").map(categorize)
        for category, factor in spec.scale.items():
            if factor == 1.0:
                continue
            mask = categories == category
            if not mask.any():
                continue
            for column in ("p_mw", "q_mvar", "sn_mva", "max_e_mwh"):
                if column in df.columns:
                    df.loc[mask, column] = df.loc[mask, column] * factor
            touched[f"{table}:{category.value}"] = int(mask.sum())

    return touched


SCENARIO_LIBRARY: Mapping[str, ScenarioSpec] = {
    "reference": ScenarioSpec(
        name="reference",
        notes="Published SimBench scenario, unmodified. Comparability baseline.",
    ),
    "weak_connection": ScenarioSpec(
        name="weak_connection",
        slack_vm_pu=1.04,
        trafo_sn_scale=0.8,
        notes=(
            "Point of common coupling sits higher and the station is weaker. "
            "Produces voltage headroom problems without inventing installed "
            "capacity."
        ),
    ),
    "moderate_growth": ScenarioSpec(
        name="moderate_growth",
        slack_vm_pu=1.04,
        scale={
            AssetCategory.PV: 1.5,
            AssetCategory.HEAT_PUMP: 1.3,
            AssetCategory.EV_CHARGER: 1.3,
        },
        notes=(
            "Working scenario: raised point of common coupling plus moderate "
            "growth in PV, heat pumps and charge points."
        ),
    ),
    "voltage_stress": ScenarioSpec(
        name="voltage_stress",
        slack_vm_pu=1.05,
        trafo_sn_scale=0.7,
        scale={AssetCategory.PV: 2.5},
        notes=(
            "Deliberately beyond the SimBench data: extreme case for testing "
            "the controller and the evaluation chain, not a realistic forecast."
        ),
    ),
    "undervoltage_stress": ScenarioSpec(
        name="undervoltage_stress",
        slack_vm_pu=0.96,
        trafo_sn_scale=0.7,
        scale={
            AssetCategory.HEAT_PUMP: 2.5,
            AssetCategory.EV_CHARGER: 2.5,
            AssetCategory.PV: 0.5,
        },
        notes=(
            "Counterpart to voltage_stress. K100 is asymmetric (-15 % against "
            "+10 %), so at least one undervoltage-dominated scenario is needed "
            "to exercise the lower criterion at all."
        ),
    ),
}
"""Named scenarios. Extreme cases are marked as such in their notes."""
