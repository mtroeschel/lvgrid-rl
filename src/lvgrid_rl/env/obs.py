"""Observation construction: a pure projection of the system state.

**Invariant I1.** The observation is derived from :class:`SystemState` and never
carries state of its own. The builder holds only configuration -- which feature
groups, which sensors, which scales -- and nothing that could not be
reconstructed from the state it is given. That is what allows a later certifier
to receive more information than the agent: it is a separate, better
instrumented component reading the same source of truth.

Normalisation uses **fixed physical scales**, not learned statistics. Two reasons:
running statistics are state, which would break I1 and the multi-agent path
(§6.3, rule 3), and they make a run irreproducible in a way that is hard to see.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from lvgrid_rl.core.information import InformationSet
from lvgrid_rl.core.schemas import SystemState

__all__ = [
    "SensorConfig",
    "FeatureGroup",
    "ObservationLayoutMode",
    "AssetFeatureSpec",
    "AssetBlock",
    "ObservationLayout",
    "ObservationSpec",
    "ObservationBuilder",
    "ASSET_FEATURES",
    "observation_layout",
]


class SensorConfig(StrEnum):
    """How much of the grid the agent may measure.

    ``FULL_STATE`` is an upper bound for research, not a realistic assumption:
    it presumes voltage measurement at every connection point. ``REALISTIC``
    restricts the agent to the substation plus a configured set of feeder-end
    measurements plus its own local quantities.
    """

    FULL_STATE = "full_state"
    REALISTIC = "realistic"


class FeatureGroup(StrEnum):
    """Feature groups of §6.2."""

    TIME = "time"
    MEASUREMENTS = "measurements"
    LOCAL_POWER = "local_power"
    ASSET_STATE = "asset_state"
    PQ_BUDGET = "pq_budget"
    FORECAST = "forecast"


class ObservationLayoutMode(StrEnum):
    """How the vector is organised.

    ``FLAT`` is the M3 layout: feature groups one after another, forecasts and
    states of charge in their own groups. ``PER_ASSET`` splits it into a global
    block and one block per asset of a fixed width per asset kind, in action
    order -- what a policy with weights shared across the assets of a kind needs
    (action mode 2, §6.3). The flat layout stays the default and is unchanged.
    """

    FLAT = "flat"
    PER_ASSET = "per_asset"


@dataclass(frozen=True, slots=True)
class AssetFeatureSpec:
    """What the per-asset block of one asset is built from.

    Args:
        asset_id: The asset.
        kind: Its kind; decides which features the block holds.
        bus_position: Position of its bus among the assessed connection points,
            for the local voltage.
        rated_p_mw: Rated power. Every power feature of the block is divided by
            it, so assets of different size look alike to a shared network; the
            rating itself is one feature, so size is not lost.
        series_id: Profile of the asset itself, for PV the available power.
        capacity_mwh: Battery capacity.
        colocated_series: PV profiles at the same bus. What a battery beside a
            PV system most needs to know is how much that system will produce.
        cop_key: Forecast key of a heat pump's COP.
    """

    asset_id: str
    kind: str
    bus_position: int
    rated_p_mw: float
    series_id: str | None = None
    capacity_mwh: float | None = None
    colocated_series: tuple[str, ...] = ()
    cop_key: str | None = None

    def __post_init__(self) -> None:
        if self.rated_p_mw <= 0.0:
            raise ValueError(f"{self.asset_id}: rated_p_mw must be positive")
        if self.kind not in ASSET_FEATURES:
            raise ValueError(
                f"{self.asset_id}: no per-asset features defined for kind "
                f"{self.kind!r}; known: {sorted(ASSET_FEATURES)}"
            )


ASSET_FEATURES: dict[str, tuple[str, ...]] = {
    # Local voltage, size and last setpoint first, identical for every kind, so
    # that the leading features mean the same thing in every block.
    "pv": ("vm_local", "rated_p", "last_p", "forecast"),
    "bess": (
        "vm_local",
        "rated_p",
        "last_p",
        "soc_frac",
        "energy_to_power",
        "colocated_pv_forecast",
    ),
    "hp": (
        "vm_local",
        "rated_p",
        "last_p",
        "buffer_frac",
        "running",
        "heat_forecast",
        "cop_forecast",
    ),
}
"""Feature names per asset kind. ``forecast`` and ``colocated_pv_forecast``
expand to one entry per forecast step; every other name is one entry."""

_EXPANDING = frozenset(
    {"forecast", "colocated_pv_forecast", "heat_forecast", "cop_forecast"}
)

COP_SCALE = 5.0
"""COPs are divided by this, so a typical value reads around 0.7."""


@dataclass(frozen=True, slots=True)
class AssetBlock:
    """Where one asset's features sit, and where its action components sit."""

    asset_id: str
    kind: str
    obs_start: int
    obs_stop: int
    action_start: int = -1
    action_stop: int = -1


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    """The per-asset layout: a global block, then one block per asset.

    Plain integers rather than slices or arrays, so that it pickles with a saved
    policy and reads the same in a year.
    """

    global_dim: int
    blocks: tuple[AssetBlock, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        """Asset kinds in order of first appearance."""
        return tuple(dict.fromkeys(b.kind for b in self.blocks))

    def blocks_of(self, kind: str) -> tuple[AssetBlock, ...]:
        """The blocks of one kind, in action order."""
        return tuple(b for b in self.blocks if b.kind == kind)


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Configuration of the observation.

    Args:
        groups: Feature groups to include, in this order.
        sensor_config: Which voltage measurements are available.
        measured_buses: Positions within the assessed connection points that are
            measured under ``REALISTIC``. Ignored under ``FULL_STATE``.
        forecast_horizon: Number of forecast steps per forecast series.
        forecast_series: Series to include in the forecast group.
        voltage_scale: Voltages are centred on 1.0 pu and divided by this, so a
            10 % deviation becomes 1.0. Chosen deliberately rather than
            empirically: it makes the normalised value read directly as "share of
            the permitted band".
        power_scale_mw: Scale for power features.
        storage_assets: ``(asset_id, capacity_mwh)`` per battery, whose state
            of charge forms the ``asset_state`` group. Empty without batteries,
            which leaves the M3 observation unchanged.
        thermal_assets: ``(asset_id, capacity_mwh, cop_key)`` per heat pump:
            buffer fill level, whether it runs, and the current COP join the
            ``asset_state`` group. Empty without heat pumps.
        layout: ``flat`` (default, the M3 layout) or ``per_asset``.
        assets: Per-asset feature specifications, in action order. Filled in
            by the environment when left empty; a builder needs them for
            ``per_asset``.
    """

    groups: tuple[FeatureGroup, ...] = (
        FeatureGroup.TIME,
        FeatureGroup.MEASUREMENTS,
        FeatureGroup.LOCAL_POWER,
        FeatureGroup.ASSET_STATE,
        FeatureGroup.PQ_BUDGET,
        FeatureGroup.FORECAST,
    )
    sensor_config: SensorConfig = SensorConfig.FULL_STATE
    measured_buses: tuple[int, ...] = ()
    forecast_horizon: int = 4
    forecast_series: tuple[str, ...] = ()
    voltage_scale: float = 0.10
    power_scale_mw: float = 0.1
    storage_assets: tuple[tuple[str, float], ...] = ()
    thermal_assets: tuple[tuple[str, float, str], ...] = ()
    layout: ObservationLayoutMode = ObservationLayoutMode.FLAT
    assets: tuple[AssetFeatureSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1")
        if (
            self.sensor_config is SensorConfig.REALISTIC
            and FeatureGroup.MEASUREMENTS in self.groups
            and not self.measured_buses
        ):
            raise ValueError(
                "sensor_config 'realistic' requires measured_buses; an empty "
                "sensor set would silently hide the voltage from the agent."
            )


@dataclass(frozen=True, slots=True)
class ObservationBuilder:
    """Projects a :class:`SystemState` onto a flat observation vector.

    Args:
        spec: Observation configuration.
        n_evaluated_buses: Number of assessed connection points.
        feature_names: Filled on construction; the layout of the vector, for
            logging and for reading a trained policy's attributions.
    """

    spec: ObservationSpec
    n_evaluated_buses: int
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.spec.layout is ObservationLayoutMode.PER_ASSET and not self.spec.assets:
            raise ValueError(
                "layout 'per_asset' needs the asset feature specs; the environment "
                "fills them in from its assets"
            )
        names: list[str] = []
        for group in self._global_groups():
            names.extend(self._names_for(group))
        if self.spec.layout is ObservationLayoutMode.PER_ASSET:
            for asset in self.spec.assets:
                names.extend(
                    f"asset/{asset.asset_id}/{name}"
                    for name in self._asset_feature_names(asset)
                )
        object.__setattr__(self, "feature_names", tuple(names))

    def _global_groups(self) -> tuple[FeatureGroup, ...]:
        """Groups of the global part.

        Per asset, forecasts and states of charge move into the asset blocks
        instead of standing on their own.
        """
        if self.spec.layout is ObservationLayoutMode.FLAT:
            return self.spec.groups
        return tuple(
            g
            for g in self.spec.groups
            if g not in (FeatureGroup.FORECAST, FeatureGroup.ASSET_STATE)
        )

    def _asset_feature_names(self, asset: AssetFeatureSpec) -> list[str]:
        out: list[str] = []
        for name in ASSET_FEATURES[asset.kind]:
            if name in _EXPANDING:
                out.extend(f"{name}/{h}" for h in range(self.spec.forecast_horizon))
            else:
                out.append(name)
        return out

    def layout(self) -> ObservationLayout:
        """Where the global block and each asset block sit in the vector.

        A method rather than a field: the builder holds configuration only
        (invariant I1), and the layout is derived from it. Action ranges are
        filled in by :func:`observation_layout`, which also knows the mapper.
        """
        if self.spec.layout is not ObservationLayoutMode.PER_ASSET:
            raise ValueError("the flat layout has no asset blocks")
        cursor = sum(len(self._names_for(g)) for g in self._global_groups())
        global_dim = cursor
        blocks = []
        for asset in self.spec.assets:
            width = len(self._asset_feature_names(asset))
            blocks.append(AssetBlock(asset.asset_id, asset.kind, cursor, cursor + width))
            cursor += width
        return ObservationLayout(global_dim=global_dim, blocks=tuple(blocks))

    def _voltage_positions(self) -> tuple[int, ...]:
        if self.spec.sensor_config is SensorConfig.FULL_STATE:
            return tuple(range(self.n_evaluated_buses))
        return self.spec.measured_buses

    def _names_for(self, group: FeatureGroup) -> list[str]:
        if group is FeatureGroup.TIME:
            return [
                "time/sin_day",
                "time/cos_day",
                "time/sin_year",
                "time/cos_year",
                "time/is_weekend",
            ]
        if group is FeatureGroup.MEASUREMENTS:
            return [f"vm_pu/bus_{i}" for i in self._voltage_positions()] + [
                "trafo_loading_percent/max",
                "line_loading_percent/max",
            ]
        if group is FeatureGroup.LOCAL_POWER:
            return ["local/p_slack_mw", "local/losses_mw"]
        if group is FeatureGroup.ASSET_STATE:
            return [
                f"asset/{asset_id}/soc_frac" for asset_id, _ in self.spec.storage_assets
            ] + [
                f"asset/{asset_id}/{name}"
                for asset_id, _, _ in self.spec.thermal_assets
                for name in ("buffer_frac", "running", "cop")
            ]
        if group is FeatureGroup.PQ_BUDGET:
            return [
                "pq/budget_used_max",
                "pq/budget_used_mean",
                "pq/n_buses_above_80pct",
                "pq/n_buses_over_budget",
                "pq/week_progress",
                "pq/open_window_progress",
            ]
        if group is FeatureGroup.FORECAST:
            return [
                f"forecast/{series}/{h}"
                for series in self.spec.forecast_series
                for h in range(self.spec.forecast_horizon)
            ]
        raise AssertionError(f"Unhandled feature group {group}")

    @property
    def dim(self) -> int:
        """Length of the observation vector."""
        return len(self.feature_names)

    def low(self) -> np.ndarray:
        """Lower bounds of the observation space.

        Deliberately generous rather than tight: a hard bound that a rare
        operating point exceeds would silently clip information, and the
        resulting failure looks like a learning problem.
        """
        return np.full(self.dim, -10.0, dtype=np.float32)

    def high(self) -> np.ndarray:
        """Upper bounds of the observation space."""
        return np.full(self.dim, 10.0, dtype=np.float32)

    def build(self, state: SystemState, info: InformationSet) -> np.ndarray:
        """Project state and information set onto the observation vector.

        Args:
            state: The full system state.
            info: What the controller may know at this decision point. Forecast
                features are taken from here, never from the state, because the
                state holds realisations.
        """
        parts: list[np.ndarray] = []
        for group in self._global_groups():
            parts.append(self._build_group(group, state, info))
        if self.spec.layout is ObservationLayoutMode.PER_ASSET:
            for asset in self.spec.assets:
                parts.append(self._build_asset(asset, state, info))
        return np.concatenate(parts).astype(np.float32)

    def _build_asset(
        self, asset: AssetFeatureSpec, state: SystemState, info: InformationSet
    ) -> np.ndarray:
        """One asset's block. Powers relative to its rating, consumer sign.

        The local voltage is the measurement at the asset's own connection
        point. That is available under the realistic sensor configuration too
        (§6.2: the local house connection values), so the block is the same in
        both configurations; only the global part differs.
        """
        horizon = self.spec.forecast_horizon
        vm = float(np.asarray(info.measurements["vm_pu"])[asset.bus_position])
        own = state.assets[asset.asset_id]
        values: list[float] = []
        for name in ASSET_FEATURES[asset.kind]:
            if name == "vm_local":
                values.append((vm - 1.0) / self.spec.voltage_scale)
            elif name == "rated_p":
                values.append(asset.rated_p_mw / self.spec.power_scale_mw)
            elif name == "last_p":
                values.append(own.last_p_mw / asset.rated_p_mw)
            elif name == "forecast":
                series = np.asarray(info.forecast[asset.series_id][:horizon])
                values.extend((series / asset.rated_p_mw).tolist())
            elif name == "soc_frac":
                values.append(own.energy_mwh / asset.capacity_mwh)
            elif name == "energy_to_power":
                # Hours of full power, as a share of a day: 2 h reads 0.083.
                values.append(asset.capacity_mwh / asset.rated_p_mw / 24.0)
            elif name == "buffer_frac":
                values.append(own.energy_mwh / asset.capacity_mwh)
            elif name == "running":
                values.append(1.0 if own.running else 0.0)
            elif name == "heat_forecast":
                series = np.asarray(info.forecast[asset.series_id][:horizon])
                values.extend((series / asset.rated_p_mw).tolist())
            elif name == "cop_forecast":
                series = np.asarray(info.forecast[asset.cop_key][:horizon])
                values.extend((series / COP_SCALE).tolist())
            elif name == "colocated_pv_forecast":
                total = np.zeros(horizon)
                for series in asset.colocated_series:
                    total += np.asarray(info.forecast[series][:horizon])
                values.extend((total / asset.rated_p_mw).tolist())
            else:  # pragma: no cover - ASSET_FEATURES and this branch go together
                raise AssertionError(f"Unhandled asset feature {name}")
        return np.asarray(values, dtype=np.float64)

    def _build_group(
        self, group: FeatureGroup, state: SystemState, info: InformationSet
    ) -> np.ndarray:
        if group is FeatureGroup.TIME:
            ts = state.timestamp
            day = (ts.hour * 60 + ts.minute) / 1440.0
            year = (ts.timetuple().tm_yday - 1) / 366.0
            return np.array(
                [
                    np.sin(2 * np.pi * day),
                    np.cos(2 * np.pi * day),
                    np.sin(2 * np.pi * year),
                    np.cos(2 * np.pi * year),
                    1.0 if ts.weekday() >= 5 else 0.0,
                ]
            )

        if group is FeatureGroup.MEASUREMENTS:
            positions = np.array(self._voltage_positions(), dtype=np.intp)
            vm = np.asarray(info.measurements["vm_pu"])[positions]
            return np.concatenate(
                [
                    (vm - 1.0) / self.spec.voltage_scale,
                    [
                        info.measurements["trafo_loading_percent_max"] / 100.0,
                        info.measurements["line_loading_percent_max"] / 100.0,
                    ],
                ]
            )

        if group is FeatureGroup.LOCAL_POWER:
            return np.array(
                [
                    info.measurements["p_slack_mw"] / self.spec.power_scale_mw,
                    info.measurements["losses_mw"] / self.spec.power_scale_mw,
                ]
            )

        if group is FeatureGroup.ASSET_STATE:
            # Without the state of charge a battery policy is not Markovian: the
            # same grid state asks for charging or discharging depending on how
            # full the battery already is. A fraction needs no further scale.
            # The same holds for a heat pump's buffer, and its minimum run and
            # idle times make whether it is running part of the state too.
            values = [
                state.assets[asset_id].energy_mwh / capacity_mwh
                for asset_id, capacity_mwh in self.spec.storage_assets
            ]
            for asset_id, capacity_mwh, cop_key in self.spec.thermal_assets:
                own = state.assets[asset_id]
                values += [
                    own.energy_mwh / capacity_mwh,
                    1.0 if own.running else 0.0,
                    float(info.forecast[cop_key][0]) / COP_SCALE,
                ]
            return np.array(values, dtype=np.float64)

        if group is FeatureGroup.PQ_BUDGET:
            # Without this group the control problem is not Markovian: the agent
            # cannot tell whether an excursion still fits the weekly budget or
            # breaks it (§6.6). Condensed rather than per-bus, so the dimension
            # does not grow with the grid.
            pq = info.pq
            used = pq.budget_used_frac()
            if used.size == 0:
                used = np.zeros(1)
            window = pq.windows[0] if pq.windows else None
            open_progress = (
                window.samples_count / max(window.samples_count + 1, 1)
                if window is not None
                else 0.0
            )
            return np.array(
                [
                    float(used.max()),
                    float(used.mean()),
                    float((used >= 0.8).sum()) / max(used.size, 1),
                    float((used >= 1.0).sum()) / max(used.size, 1),
                    pq.windows_elapsed_count / max(pq.windows_total_count, 1),
                    float(open_progress),
                ]
            )

        if group is FeatureGroup.FORECAST:
            rows = [
                np.asarray(info.forecast[series][: self.spec.forecast_horizon])
                / self.spec.power_scale_mw
                for series in self.spec.forecast_series
            ]
            return np.concatenate(rows) if rows else np.zeros(0)

        raise AssertionError(f"Unhandled feature group {group}")


def default_forecast_series(asset_ids: Sequence[str]) -> tuple[str, ...]:
    """Forecast series for a PV-only environment: one per controllable asset."""
    return tuple(asset_ids)


def observation_layout(builder: ObservationBuilder, mapper) -> ObservationLayout:
    """The builder's per-asset layout with the action range of every asset.

    Requires the blocks to be in action order -- the whole point of a shared
    network is that block ``i`` produces the action of asset ``i``, and an order
    that silently differed would pair each asset with another's features.
    """
    layout = builder.layout()
    if tuple(b.asset_id for b in layout.blocks) != tuple(mapper.asset_ids):
        raise ValueError("per-asset blocks are not in action order")
    blocks = []
    cursor = 0
    for block, spec in zip(layout.blocks, mapper.specs, strict=True):
        blocks.append(
            AssetBlock(
                block.asset_id,
                block.kind,
                block.obs_start,
                block.obs_stop,
                cursor,
                cursor + spec.dim,
            )
        )
        cursor += spec.dim
    return ObservationLayout(global_dim=layout.global_dim, blocks=tuple(blocks))
