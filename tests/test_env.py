"""Tests of the environment core (M3)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.core.information import ClairvoyanceError, DecisionScope, assert_readable
from lvgrid_rl.core.protocols import ActionSpec
from lvgrid_rl.core.schemas import AssetRatings, Interval
from lvgrid_rl.env.actions import ActionMapper
from lvgrid_rl.env.obs import (
    FeatureGroup,
    ObservationBuilder,
    ObservationSpec,
    SensorConfig,
)
from lvgrid_rl.env.reward import RewardComposer, RewardConfig, RewardMode, TermSpec
from lvgrid_rl.grid.metrics import ViolationMetrics

pytest.importorskip("gymnasium", reason="extra 'env' not installed")


def _spec(lo: float = -0.02, hi: float = 0.0) -> ActionSpec:
    return ActionSpec(names=("p_mw",), bounds=(Interval(lo, hi),))


def _metrics(**kwargs) -> ViolationMetrics:
    defaults = dict(
        max_vm_pu=1.05,
        min_vm_pu=0.99,
        voltage_excursion_pu=0.0,
        n_buses_outside_k95=0,
        n_buses_outside_k100=0,
        max_line_loading_percent=50.0,
        max_trafo_loading_percent=60.0,
        overload_excess_percent=0.0,
        losses_mw=0.01,
        converged=True,
    )
    return ViolationMetrics(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# ActionMapper -- invariant I4
# ---------------------------------------------------------------------------


def test_normalisation_is_invertible() -> None:
    """I4: the normalisation must be invertible.

    The certified feasible set is formulated in injections, so the map between
    action and injection has to be reversible.
    """
    mapper = ActionMapper(("a", "b"), (_spec(-0.02, 0.0), _spec(-0.05, 0.0)))
    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.uniform(-1.0, 1.0, size=mapper.dim)
        assert np.allclose(mapper.to_normalised(mapper.to_physical(action)), action)


def test_normalisation_is_affine() -> None:
    """I4: the mapping must preserve convex combinations.

    A non-affine mapping would make a box in action coordinates a non-box in
    injection coordinates, which is exactly what a projection onto the certified
    set cannot handle.
    """
    mapper = ActionMapper(("a",), (_spec(-0.02, 0.0),))
    x, y = np.array([-0.7]), np.array([0.4])
    for weight in (0.0, 0.25, 0.5, 1.0):
        mixed = weight * x + (1 - weight) * y
        expected = weight * mapper.to_physical(x) + (1 - weight) * mapper.to_physical(y)
        assert np.allclose(mapper.to_physical(mixed), expected)


def test_bounds_map_to_the_physical_limits() -> None:
    mapper = ActionMapper(("a",), (_spec(-0.02, 0.0),))
    assert mapper.to_physical(np.array([-1.0]))[0] == pytest.approx(-0.02)
    assert mapper.to_physical(np.array([1.0]))[0] == pytest.approx(0.0)


def test_degenerate_component_does_not_divide_by_zero() -> None:
    """A PV system with zero rated power has a one-point interval."""
    mapper = ActionMapper(("a",), (_spec(0.0, 0.0),))
    assert mapper.to_physical(np.array([0.5]))[0] == 0.0
    assert mapper.to_normalised(np.array([0.0]))[0] == 0.0


def test_neutral_action_means_no_curtailment() -> None:
    """A helper against the sign trap.

    For PV the upper bound is zero infeed, so +1 is full curtailment and -1 is
    full infeed.
    """
    mapper = ActionMapper(("a",), (_spec(-0.02, 0.0),))
    assert mapper.to_physical(mapper.neutral_action())[0] == pytest.approx(-0.02)


def test_component_order_is_stable_and_named() -> None:
    mapper = ActionMapper(("sgen:1", "sgen:0"), (_spec(), _spec()))
    assert mapper.component_names == ("sgen:1/p_mw", "sgen:0/p_mw")
    assert list(mapper.split(np.array([1.0, 2.0]))["sgen:1"]) == [1.0]


def test_duplicate_asset_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        ActionMapper(("a", "a"), (_spec(), _spec()))


# ---------------------------------------------------------------------------
# ObservationBuilder -- invariant I1
# ---------------------------------------------------------------------------


def test_observation_layout_is_introspectable() -> None:
    builder = ObservationBuilder(ObservationSpec(forecast_series=("s",)), 3)
    assert builder.dim == len(builder.feature_names)
    assert "pq/budget_used_max" in builder.feature_names


def test_sensor_config_changes_what_is_visible() -> None:
    """I1: sensor configuration changes what the agent sees.

    Two builders with different configurations must produce different
    observations from the same state.
    """
    full = ObservationBuilder(
        ObservationSpec(
            groups=(FeatureGroup.MEASUREMENTS,), sensor_config=SensorConfig.FULL_STATE
        ),
        5,
    )
    realistic = ObservationBuilder(
        ObservationSpec(
            groups=(FeatureGroup.MEASUREMENTS,),
            sensor_config=SensorConfig.REALISTIC,
            measured_buses=(0, 4),
        ),
        5,
    )
    assert full.dim > realistic.dim


def test_realistic_sensing_without_sensors_is_rejected() -> None:
    """An empty sensor set would silently hide the voltage from the agent."""
    with pytest.raises(ValueError, match="measured_buses"):
        ObservationSpec(sensor_config=SensorConfig.REALISTIC, measured_buses=())


def test_builder_holds_no_state() -> None:
    """I1: the builder carries no state.

    It is configuration only; anything else would break the claim that the
    observation is a pure projection.
    """
    builder = ObservationBuilder(ObservationSpec(forecast_series=("s",)), 2)
    assert builder.__slots__  # frozen dataclass with slots
    assert set(builder.__slots__) == {"spec", "n_evaluated_buses", "feature_names"}


# ---------------------------------------------------------------------------
# RewardComposer
# ---------------------------------------------------------------------------


def test_terms_are_reported_individually() -> None:
    """Every term is reported individually.

    Without the decomposition it is impossible to reconstruct which term
    dominated learning.
    """
    composer = RewardComposer(RewardConfig(), budget_windows=50)
    result = composer.compute(
        metrics=_metrics(overload_excess_percent=20.0),
        curtailed_energy_mwh=0.5,
        setpoint_change_mw=0.01,
        k95_violations_this_step=2,
        k100_violations_this_step=0,
        n_buses=13,
        dt_hours=0.25,
    )
    assert set(result.terms) == {
        "pv_curtailment",
        "action_smoothness",
        "grid_losses",
        "en50160_k95",
        "en50160_k100",
        "thermal_overload",
    }
    assert result.total == pytest.approx(sum(result.terms.values()))
    assert "reward/pv_curtailment" in result.as_info()


def test_costs_are_reported_unweighted() -> None:
    """Costs are reported unweighted.

    The Lagrangian callback and the KPI engine consume the physical values, not
    the weighted ones.
    """
    composer = RewardComposer(RewardConfig(), budget_windows=50)
    result = composer.compute(
        metrics=_metrics(),
        curtailed_energy_mwh=0.0,
        setpoint_change_mw=0.0,
        k95_violations_this_step=13,
        k100_violations_this_step=0,
        n_buses=13,
        dt_hours=0.25,
    )
    assert result.costs["en50160_k95"] == pytest.approx(13 / (13 * 50))


def test_lagrangian_mode_ignores_constraints_until_multipliers_are_set() -> None:
    config = RewardConfig(mode=RewardMode.LAGRANGIAN)
    composer = RewardComposer(config, budget_windows=50)
    kwargs = dict(
        metrics=_metrics(overload_excess_percent=50.0),
        curtailed_energy_mwh=0.0,
        setpoint_change_mw=0.0,
        k95_violations_this_step=5,
        k100_violations_this_step=0,
        n_buses=13,
        dt_hours=0.25,
    )
    before = composer.compute(**kwargs)
    assert before.terms["en50160_k95"] == 0.0
    composer.set_multipliers({"en50160_k95": 4.0})
    after = composer.compute(**kwargs)
    assert after.terms["en50160_k95"] < 0.0


def test_unknown_multiplier_is_rejected() -> None:
    composer = RewardComposer(RewardConfig(), budget_windows=50)
    with pytest.raises(KeyError, match="Unknown constraint terms"):
        composer.set_multipliers({"not_a_term": 1.0})


def test_k95_limit_is_the_standard_criterion() -> None:
    """The K95 limit is the standard's own criterion.

    In the Lagrangian formulation the limit is the norm itself, so no penalty
    weight has to be guessed.
    """
    assert RewardConfig().constraint["en50160_k95"].limit == pytest.approx(0.05)


def test_a_term_cannot_be_both_objective_and_constraint() -> None:
    with pytest.raises(ValueError, match="both categories"):
        RewardConfig(objective={"x": TermSpec(-1.0)}, constraint={"x": TermSpec(-1.0)})


def test_divergence_is_penalised_not_raised() -> None:
    composer = RewardComposer(RewardConfig(), budget_windows=50)
    result = composer.compute(
        metrics=_metrics(converged=False),
        curtailed_energy_mwh=0.0,
        setpoint_change_mw=0.0,
        k95_violations_this_step=0,
        k100_violations_this_step=0,
        n_buses=13,
        dt_hours=0.25,
    )
    assert result.total == RewardConfig().divergence_penalty


def test_potential_penalises_spending_budget_early() -> None:
    """Spending budget early costs more than spending it late.

    Early consumption removes options.
    """
    composer = RewardComposer(RewardConfig(), budget_windows=50)
    early = composer.potential(budget_used_max=0.5, week_progress=0.1)
    late = composer.potential(budget_used_max=0.5, week_progress=0.45)
    assert early < late <= 0.0


# ---------------------------------------------------------------------------
# Information ordering -- invariant I3
# ---------------------------------------------------------------------------


def test_decision_scope_blocks_reading_the_coming_interval() -> None:
    with DecisionScope(t_decision=100):
        assert_readable(100)
        with pytest.raises(ClairvoyanceError):
            assert_readable(101)


def test_timestamps_must_be_timezone_aware() -> None:
    """Guards the calendar features against a silent daylight saving error."""
    from lvgrid_rl.core.schemas import SystemState

    with pytest.raises(ValueError, match="UTC"):
        SystemState(
            t_index=0,
            timestamp=datetime(2016, 6, 1, 12, 0),
            topology_id="t",
            grid=None,  # type: ignore[arg-type]
            assets={},
            exogenous=None,  # type: ignore[arg-type]
            pq=None,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# PV asset
# ---------------------------------------------------------------------------


def test_pv_clipping_is_reported_not_silent() -> None:
    """A policy systematically proposing infeasible actions must stay visible."""
    from lvgrid_rl.components.pv import PvSystem
    from lvgrid_rl.core.information import InformationSet
    from lvgrid_rl.core.schemas import PQBudgetState

    asset = PvSystem(
        asset_id="sgen:0",
        bus=1,
        ratings=AssetRatings(p_min_mw=-0.02, p_max_mw=0.0),
        series_id="sgen:0",
    )
    info = InformationSet(
        t_index=0,
        timestamp=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
        measurements={},
        asset_states={},
        series_ids=("sgen:0",),
        exogenous_bounds_mw=np.zeros((2, 1)),
        forecast={"sgen:0": np.array([-0.005])},
        pq=PQBudgetState(windows_elapsed_count=0),
    )
    state = asset.initial_state(np.random.default_rng(0))
    # Asks for full infeed although only a quarter is available.
    setpoint = asset.to_setpoint(state, np.array([-0.02]), info)
    assert setpoint.p_mw == pytest.approx(-0.005)
    assert setpoint.was_clipped


def test_pv_dynamics_is_a_pure_function() -> None:
    """Prepares invariant I2, which falls due with the stateful assets in M4."""
    from lvgrid_rl.components.pv import PvState, PvSystem
    from lvgrid_rl.core.protocols import Setpoint
    from lvgrid_rl.core.schemas import ExogenousInput

    asset = PvSystem(
        asset_id="sgen:0",
        bus=1,
        ratings=AssetRatings(p_min_mw=-0.02, p_max_mw=0.0),
        series_id="sgen:0",
    )
    state = PvState(asset_id="sgen:0", last_p_mw=-0.01)
    setpoint = Setpoint(asset_id="sgen:0", p_mw=-0.008)
    exogenous = ExogenousInput(
        t_index=0,
        series_ids=("sgen:0",),
        realized_mw=np.array([-0.015]),
        bounds_mw=np.array([[-0.02], [0.0]]),
        ambient_temp_degc=10.0,
        ghi_wm2=500.0,
    )
    first = asset.dynamics(state, setpoint, exogenous, None)  # type: ignore[arg-type]
    second = asset.dynamics(state, setpoint, exogenous, None)  # type: ignore[arg-type]
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert state.last_p_mw == -0.01, "input state was mutated"
    assert first[1].curtailed_energy_mwh == pytest.approx(0.007)


# ---------------------------------------------------------------------------
# Environment as a whole
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """A training environment on the working scenario."""
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.env.factory import make_env

    return make_env(seed=1)


def test_env_passes_the_gymnasium_checker(env) -> None:
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


def test_replays_with_the_same_seed_are_identical(env) -> None:
    """The reproducibility claim of section 9.2, checked at the environment.

    The power flow warm start makes a solution depend on the operating point
    that preceded it, so the engine has to forget it on reset. Without that,
    two identical replays drift apart invisibly.
    """
    from lvgrid_rl.env.factory import make_env

    def rollout() -> list[float]:
        e = make_env(seed=7)
        e.reset(seed=7)
        action = e.mapper.neutral_action()
        return [float(e.step(action)[1]) for _ in range(6)]

    assert rollout() == rollout()


def test_only_representable_step_sizes_are_accepted() -> None:
    """Two constraints apply at once and their intersection is narrower.

    EN 50160 permits {1, 2, 5, 10}; the 15-minute SimBench series restricts that
    to {1, 5}. A 10-minute step is representable under the standard but not from
    this source.
    """
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.env.factory import make_env
    from lvgrid_rl.env.lv_grid_env import EnvConfig

    with pytest.raises(ValueError, match="cannot be formed"):
        make_env(config=EnvConfig(sim_dt_min=10, control_dt_min=30))


def test_curtailment_costs_yield_and_removes_overload(env) -> None:
    """The trade-off that makes M3 a control problem rather than a toy.

    On a high-PV week, feeding in everything overloads the transformer while
    curtailing everything removes the overload at the cost of lost yield. The
    optimum lies between, which is what the agent has to find.
    """
    import numpy as np

    from lvgrid_rl.env.episodes import EpisodeMode, EpisodeSpec
    from lvgrid_rl.env.factory import make_env

    spec = EpisodeSpec(mode=EpisodeMode.EVALUATE, randomise_budget=False)
    results = {}
    for label, action_of in (
        ("none", lambda e: e.mapper.neutral_action()),
        ("full", lambda e: np.ones(e.mapper.dim)),
    ):
        e = make_env(seed=1, episode_spec=spec, set_name="stress")
        for _ in range(3):  # advance to the PV-strongest stress week
            e.reset(seed=1)
        action = action_of(e)
        curtailed = 0.0
        overload = 0.0
        for _ in range(96):
            _, _, _, _, info = e.step(action)
            curtailed += info["curtailed_energy_mwh"]
            overload += info.get("cost/thermal_overload", 0.0)
        results[label] = (curtailed, overload)

    assert results["none"][0] == pytest.approx(0.0)
    assert results["none"][1] > 0.0, "unconstrained infeed must overload something"
    assert results["full"][0] > 1.0, "full curtailment must cost yield"
    assert results["full"][1] == pytest.approx(0.0)


def test_pq_budget_features_are_present(env) -> None:
    """Without them the control problem is not Markovian (section 6.6)."""
    names = env.obs_builder.feature_names
    assert any(n.startswith("pq/") for n in names)
    assert "pq/budget_used_max" in names
    assert "pq/week_progress" in names
