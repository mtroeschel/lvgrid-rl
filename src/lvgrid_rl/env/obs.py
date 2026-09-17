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

__all__ = ["SensorConfig", "FeatureGroup", "ObservationSpec", "ObservationBuilder"]


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
    PQ_BUDGET = "pq_budget"
    FORECAST = "forecast"


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
    """

    groups: tuple[FeatureGroup, ...] = (
        FeatureGroup.TIME,
        FeatureGroup.MEASUREMENTS,
        FeatureGroup.LOCAL_POWER,
        FeatureGroup.PQ_BUDGET,
        FeatureGroup.FORECAST,
    )
    sensor_config: SensorConfig = SensorConfig.FULL_STATE
    measured_buses: tuple[int, ...] = ()
    forecast_horizon: int = 4
    forecast_series: tuple[str, ...] = ()
    voltage_scale: float = 0.10
    power_scale_mw: float = 0.1

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
        names: list[str] = []
        for group in self.spec.groups:
            names.extend(self._names_for(group))
        object.__setattr__(self, "feature_names", tuple(names))

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
        for group in self.spec.groups:
            parts.append(self._build_group(group, state, info))
        return np.concatenate(parts).astype(np.float32)

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
