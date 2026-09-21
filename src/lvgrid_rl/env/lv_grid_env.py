"""The Gymnasium environment: PV curtailment on a low-voltage grid.

M3 restricts the actuators to PV curtailment. That is not a toy: under
``moderate_growth`` the fallback action brings transformer loading from 280 %
down to 55.6 % and voltage to 1.051 pu, so curtailing PV alone resolves both the
voltage and the thermal problem. What makes it a control problem is the cost --
every curtailed kilowatt-hour is lost yield, and the EN 50160 budget means the
agent may spend a limited amount of voltage excursion rather than avoiding all of
it.

Three structural commitments are visible in :meth:`LVGridEnv.step` and matter
beyond M3:

* **Information ordering** (invariant I3): action construction runs inside a
  :class:`DecisionScope`, so an accidental read of the coming interval's
  realisation raises instead of silently producing a clairvoyant controller.
* **Two safety intervention points** (invariant I7): ``env.safety`` sits inside
  the step before setpoints are written, and ``info["action_mask"]`` is always
  present. Both have null implementations in M3.
* **Assets are held per actuator** (§6.3, rule 1), and the flat single-agent
  action vector is a view over one partition containing all of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lvgrid_rl.components.pv import PvSystem
from lvgrid_rl.core.information import DecisionScope, InformationSet
from lvgrid_rl.core.protocols import NullSafetyComponent, SafetyComponent, Setpoint
from lvgrid_rl.core.schemas import (
    AssetState,
    ExogenousInput,
    GridState,
    SystemState,
)
from lvgrid_rl.data.timebase import TimeBase
from lvgrid_rl.env.actions import ActionMapper
from lvgrid_rl.env.episodes import Episode, EpisodeSampler
from lvgrid_rl.env.obs import ObservationBuilder, ObservationSpec
from lvgrid_rl.env.reward import RewardComposer, RewardConfig
from lvgrid_rl.grid.loader import GridModel
from lvgrid_rl.grid.metrics import ViolationMetrics, violation_metrics
from lvgrid_rl.grid.powerflow import PandapowerEngine
from lvgrid_rl.grid.pq import PQAggregator

try:  # pragma: no cover - import guard
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The environment needs the 'env' extra: uv sync --extra sim --extra env"
    ) from exc

__all__ = ["EnvConfig", "LVGridEnv"]


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Environment configuration.

    Args:
        sim_dt_min: Simulation step size. Two constraints apply at once and
            their intersection is narrower than either alone: EN 50160 requires
            the step to divide ten minutes, giving ``{1, 2, 5, 10}``, and the
            source data must be resamplable onto it without a grid offset. With
            the 15-minute SimBench series that leaves **only 1 and 5 minutes**;
            2 and 10 are not representable (15/10 = 1.5). The default is 5 --
            two power flows per assessment window -- with 1 minute for the final
            evaluation.
        control_dt_min: Control step size, a multiple of ``sim_dt_min``.
        forecast_horizon: Forecast steps handed to the agent.
        perfect_forecast: Whether the forecast is the realisation. A declared
            special case and upper bound; the error model arrives in M5.
        observation: Observation configuration.
        reward: Reward configuration.
    """

    sim_dt_min: int = 5
    control_dt_min: int = 15
    forecast_horizon: int = 4
    perfect_forecast: bool = True
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        from lvgrid_rl.data.timebase import TimeBase

        TimeBase(self.sim_dt_min, self.control_dt_min)  # validates both


class LVGridEnv(gym.Env):
    """Low-voltage grid control with PV curtailment.

    Args:
        model: Prepared grid with its scenario applied.
        assets: Controllable assets, in a stable order.
        profiles: Absolute power per series and simulation step, consumer sign
            convention, already resampled to ``sim_dt``.
        sampler: Episode sampler bound to one set of the committed split.
        config: Environment configuration.
        safety: Safety component; the null implementation by default.
        seed: Base seed for the environment's own generator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: GridModel,
        assets: Sequence[PvSystem],
        profiles: np.ndarray,
        series_ids: Sequence[str],
        timestamps: Sequence[Any],
        profiles_index: Any,
        uncontrolled: Mapping[str, np.ndarray],
        sampler: EpisodeSampler,
        config: EnvConfig | None = None,
        safety: SafetyComponent | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.assets = tuple(assets)
        self.config = config or EnvConfig()
        self.safety = safety or NullSafetyComponent()
        self.sampler = sampler
        self.series_ids = tuple(series_ids)
        self._profiles = profiles
        self._timestamps = timestamps
        # Kept for tests and for tooling that needs to resolve weeks to steps
        # without rebuilding the profile frame.
        self._profiles_index = profiles_index
        self._uncontrolled = dict(uncontrolled)

        self.engine = PandapowerEngine(model)
        self.mapper = ActionMapper.from_assets(self.assets)
        self.steps_per_control = self.config.control_dt_min // self.config.sim_dt_min
        self.samples_per_window = 10 // self.config.sim_dt_min
        self.aggregator = PQAggregator(model.n_evaluated_buses, self.samples_per_window)
        self.reward_composer = RewardComposer(
            self.config.reward, self.aggregator.budget_windows
        )

        observation_spec = self.config.observation
        if not observation_spec.forecast_series:
            observation_spec = ObservationSpec(
                groups=observation_spec.groups,
                sensor_config=observation_spec.sensor_config,
                measured_buses=observation_spec.measured_buses,
                forecast_horizon=self.config.forecast_horizon,
                forecast_series=tuple(a.series_id for a in self.assets),
                voltage_scale=observation_spec.voltage_scale,
                power_scale_mw=observation_spec.power_scale_mw,
            )
        self.obs_builder = ObservationBuilder(observation_spec, model.n_evaluated_buses)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.mapper.dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.obs_builder.low(),
            high=self.obs_builder.high(),
            dtype=np.float32,
        )

        timebase_gamma = TimeBase(
            self.config.sim_dt_min, self.config.control_dt_min
        ).suggested_gamma()
        self._gamma = timebase_gamma
        self._phi = 0.0
        self._rng = np.random.default_rng(seed)
        self._episode: Episode | None = None
        self._t = 0
        self._steps_done = 0
        self._asset_states: dict[str, AssetState] = {}
        self._last_grid: GridState | None = None

    # -- helpers ------------------------------------------------------------

    @property
    def dt_hours(self) -> float:
        """Duration of one simulation step in hours."""
        return self.config.sim_dt_min / 60.0

    def _budget_potential(self) -> float:
        """Current value of the shaping potential."""
        pq = self.aggregator.state()
        used = pq.budget_used_frac()
        progress = pq.windows_elapsed_count / max(pq.windows_total_count, 1)
        return self.reward_composer.potential(
            float(used.max()) if used.size else 0.0, progress
        )

    def _exogenous(self, t: int) -> ExogenousInput:
        values = self._profiles[t]
        bounds = np.vstack([np.minimum(values, 0.0), np.maximum(values, 0.0)])
        return ExogenousInput(
            t_index=t,
            series_ids=self.series_ids,
            realized_mw=values.copy(),
            bounds_mw=bounds,
            ambient_temp_degc=0.0,
            ghi_wm2=0.0,
        )

    def _forecast(self, t: int) -> dict[str, np.ndarray]:
        """Forecast per controllable series.

        With ``perfect_forecast`` this is the realisation, which is a declared
        special case (§6.7) rather than a leak: the error model arrives in M5 and
        plugs in here.
        """
        horizon = self.config.forecast_horizon
        end = min(t + horizon, len(self._profiles))
        out: dict[str, np.ndarray] = {}
        for i, series in enumerate(self.series_ids):
            window = self._profiles[t:end, i]
            if len(window) < horizon:
                window = np.pad(window, (0, horizon - len(window)), mode="edge")
            out[series] = window
        return out

    def _information_set(self, t: int, grid: GridState) -> InformationSet:
        measurements = {
            "vm_pu": grid.vm_pu[self.model.evaluated_bus_positions],
            "trafo_loading_percent_max": float(np.nanmax(grid.trafo_loading_percent)),
            "line_loading_percent_max": float(np.nanmax(grid.line_loading_percent)),
            "p_slack_mw": grid.p_slack_mw,
            "losses_mw": grid.losses_mw,
        }
        exogenous = self._exogenous(t)
        return InformationSet(
            t_index=t,
            timestamp=self._timestamps[t],
            measurements=measurements,
            asset_states=dict(self._asset_states),
            series_ids=self.series_ids,
            exogenous_bounds_mw=exogenous.bounds_mw.copy(),
            forecast=self._forecast(t),
            pq=self.aggregator.state(),
        )

    def _system_state(self, t: int, grid: GridState) -> SystemState:
        return SystemState(
            t_index=t,
            timestamp=self._timestamps[t],
            topology_id=self.model.code,
            grid=grid,
            assets=dict(self._asset_states),
            exogenous=self._exogenous(t),
            pq=self.aggregator.state(),
        )

    def _setpoints_for(self, t: int) -> dict[str, Setpoint]:
        """Uncontrolled elements at step ``t``."""
        out: dict[str, Setpoint] = {}
        for asset_id, series in self._uncontrolled.items():
            out[asset_id] = Setpoint(asset_id=asset_id, p_mw=float(series[t]))
        return out

    # -- Gymnasium API ------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Cold start for the first solve of the episode: the warm start would
        # otherwise carry the previous episode's operating point into this one
        # and make replays with the same seed differ.
        self.engine.reset_warm_start()

        episode = self.sampler.sample(self._rng)
        self._episode = episode
        self._t = episode.start_step
        self._steps_done = 0
        self.aggregator.reset_week(k95=episode.initial_k95_violations)
        self._asset_states = {
            asset.asset_id: asset.initial_state(self._rng) for asset in self.assets
        }

        setpoints = self._setpoints_for(self._t)
        for asset in self.assets:
            setpoints[asset.asset_id] = Setpoint(asset.asset_id, p_mw=0.0)
        grid = self.engine.run(setpoints, self._t)
        self._last_grid = grid
        # The potential is anchored at the episode start, so the first shaped
        # step compares against the initial budget rather than against zero.
        self._phi = self._budget_potential()

        info_set = self._information_set(self._t, grid)
        state = self._system_state(self._t, grid)
        observation = self.obs_builder.build(state, info_set)
        return observation, {
            "week": episode.week,
            "action_mask": self.safety.action_mask(state, info_set),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one control step."""
        assert self._episode is not None, "call reset() first"
        grid = self._last_grid
        assert grid is not None

        info_set = self._information_set(self._t, grid)
        state = self._system_state(self._t, grid)

        # Steps 1 to 3 of §6.1 may only use what is knowable now. The scope makes
        # an accidental read of the coming interval raise rather than silently
        # produce a clairvoyant controller.
        with DecisionScope(t_decision=self._t):
            physical = self.mapper.to_physical(np.asarray(action))
            physical, intervention = self.safety.transform(physical, state, info_set)
            per_asset = self.mapper.split(physical)
            controlled = {
                asset.asset_id: asset.to_setpoint(
                    self._asset_states[asset.asset_id],
                    per_asset[asset.asset_id],
                    info_set,
                )
                for asset in self.assets
            }

        # The reward is evaluated **once per control step**, on quantities
        # aggregated over the inner simulation steps. Evaluating it per inner
        # step and passing running totals would count the same cost repeatedly
        # and grow it superlinearly with the number of inner steps -- a mistake
        # that is invisible in the reward curve because the shape still looks
        # plausible.
        previous = {
            asset.asset_id: self._asset_states[asset.asset_id].last_p_mw
            for asset in self.assets
        }
        setpoint_change_mw = sum(
            abs(controlled[asset.asset_id].p_mw - previous[asset.asset_id])
            for asset in self.assets
        )

        curtailed_mwh = 0.0
        k95_total = 0
        k100_total = 0
        overload_excess = 0.0
        losses_mwh = 0.0
        worst_vm = (float("inf"), -float("inf"))
        excursion = 0.0
        outside_k95 = 0
        outside_k100 = 0
        converged = True
        clipped = any(sp.was_clipped for sp in controlled.values())
        inner_steps = 0

        for _ in range(self.steps_per_control):
            if self._t >= len(self._profiles) - 1:
                break
            exogenous = self._exogenous(self._t)
            # The decision is held for the whole control step, but the physics is
            # not: a PV system cannot feed in more than the current irradiance
            # allows. Limiting here rather than in the controller keeps the
            # decision/physics split clean.
            applied = {
                asset.asset_id: asset.apply_availability(
                    controlled[asset.asset_id], exogenous
                )
                for asset in self.assets
            }
            setpoints = self._setpoints_for(self._t)
            setpoints.update(applied)
            grid = self.engine.run(setpoints, self._t)
            inner_steps += 1

            before = self.aggregator.state()
            if grid.converged:
                self.aggregator.add_sample(grid.vm_pu[self.model.evaluated_bus_positions])
            after = self.aggregator.state()
            k95_total += int(
                np.asarray(after.violations_k95_count).sum()
                - np.asarray(before.violations_k95_count).sum()
            )
            k100_total += int(
                np.asarray(after.violations_k100_count).sum()
                - np.asarray(before.violations_k100_count).sum()
            )

            for asset in self.assets:
                new_state, outcome = asset.dynamics(
                    self._asset_states[asset.asset_id],
                    applied[asset.asset_id],
                    exogenous,
                    grid,
                )
                self._asset_states[asset.asset_id] = new_state
                curtailed_mwh += outcome.curtailed_energy_mwh * self.dt_hours

            metrics = violation_metrics(grid, self.model.evaluated_bus_positions)
            if metrics.converged:
                overload_excess += metrics.overload_excess_percent
                losses_mwh += metrics.losses_mw * self.dt_hours
                worst_vm = (
                    min(worst_vm[0], metrics.min_vm_pu),
                    max(worst_vm[1], metrics.max_vm_pu),
                )
                excursion += metrics.voltage_excursion_pu
                outside_k95 = max(outside_k95, metrics.n_buses_outside_k95)
                outside_k100 = max(outside_k100, metrics.n_buses_outside_k100)
            else:
                converged = False

            self._t += 1
            self._steps_done += 1

        aggregated = ViolationMetrics(
            max_vm_pu=worst_vm[1] if converged else float("nan"),
            min_vm_pu=worst_vm[0] if converged else float("nan"),
            voltage_excursion_pu=excursion,
            n_buses_outside_k95=outside_k95,
            n_buses_outside_k100=outside_k100,
            max_line_loading_percent=float("nan"),
            max_trafo_loading_percent=float("nan"),
            overload_excess_percent=overload_excess,
            losses_mw=losses_mwh / max(self.dt_hours * max(inner_steps, 1), 1e-12),
            converged=converged,
        )
        step_reward = self.reward_composer.compute(
            metrics=aggregated,
            curtailed_energy_mwh=curtailed_mwh,
            setpoint_change_mw=setpoint_change_mw,
            k95_violations_this_step=k95_total,
            k100_violations_this_step=k100_total,
            n_buses=self.model.n_evaluated_buses,
            dt_hours=self.dt_hours * max(inner_steps, 1),
        )
        reward_total = step_reward.total
        breakdown_terms = dict(step_reward.terms)
        breakdown_costs = dict(step_reward.costs)
        totals = {
            "curtailed_energy_mwh": curtailed_mwh,
            "k95": k95_total,
            "k100": k100_total,
        }

        # Potential-based shaping on the budget: the true cost is terminal, so a
        # dense signal is needed for credit assignment over hundreds of steps.
        #
        # It must be the *difference* gamma * phi(s') - phi(s), not phi(s')
        # alone. Only the difference telescopes over a trajectory and leaves the
        # optimal policy unchanged (Ng et al.); adding the potential directly
        # would be an ordinary reward term that biases the policy -- and it would
        # do so invisibly, because the shaped return still looks plausible.
        phi_next = self._budget_potential()
        shaping = self._gamma * phi_next - self._phi
        self._phi = phi_next
        reward_total += shaping
        breakdown_terms["budget_potential"] = shaping

        self._last_grid = grid
        info_set = self._information_set(min(self._t, len(self._profiles) - 1), grid)
        state = self._system_state(min(self._t, len(self._profiles) - 1), grid)
        observation = self.obs_builder.build(state, info_set)

        truncated = (
            self._steps_done >= self._episode.n_steps
            or self._t >= len(self._profiles) - 1
        )
        terminated = False

        info: dict[str, Any] = {
            "action_mask": self.safety.action_mask(state, info_set),
            "intervened": intervention.intervened,
            "action_clipped": clipped,
            "pf_diverged": not grid.converged,
            "k95_violations": totals["k95"],
            "k100_violations": totals["k100"],
            "curtailed_energy_mwh": totals["curtailed_energy_mwh"],
        }
        info.update({f"reward/{k}": v for k, v in breakdown_terms.items()})
        info.update({f"cost/{k}": v for k, v in breakdown_costs.items()})
        info["reward/total"] = reward_total

        return observation, float(reward_total), terminated, truncated, info
