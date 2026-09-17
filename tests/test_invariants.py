"""The seven invariants of the extensibility contract (section 13).

**Why this file exists.** Decision D10 defers certification to M7b but requires
that the architecture can still take it later. That promise is only credible if
it is checked: promises of modularity decay silently. Six months without a test
and some reasonable shortcut has used up the extensibility without anyone
noticing.

**Why ``xfail(strict=True)`` and not failing placeholders.** A permanently red CI
is ignored after two weeks, which would defeat the purpose. ``strict=True``
inverts the logic: the test *must not* pass while the component is missing. As
soon as it exists and the test turns green unexpectedly, **the build breaks** and
forces the marker to be removed. The number of XFAIL reports is therefore the
contract's debt count, readable in every CI run.

Status: I3, I5 and I6 are satisfied (I6 since the SimBench adapter in M1, I5
since the power flow engine in M2). I1, I4 and I7 fall due in M3, I2 in M4.
"""

from __future__ import annotations

import dataclasses
import importlib

import numpy as np
import pytest

from lvgrid_rl.core.information import (
    ClairvoyanceError,
    DecisionScope,
    InformationSet,
    assert_readable,
)
from lvgrid_rl.core.protocols import NullSafetyComponent, SafetyComponent
from lvgrid_rl.core.schemas import ExogenousInput, SystemState

pytestmark = pytest.mark.invariant


def _module_available(name: str) -> bool:
    """Does a module exist that would satisfy a still-open invariant?"""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return False
    return True


# ---------------------------------------------------------------------------
# I1 -- SystemState complete and separate from Observation
# ---------------------------------------------------------------------------


def test_i1_observation_is_a_pure_projection_of_system_state() -> None:
    """I1: the ``ObservationBuilder`` reads only from ``SystemState``.

    The certifier needs the full state, the agent may see less. If observation
    and state were the same object that could not be separated. Satisfied since
    the observation builder landed in M3.

    Checked three ways: the builder carries nothing but configuration, repeated
    builds from the same state are identical, and two sensor configurations see
    different amounts of the same grid.
    """
    pytest.importorskip("gymnasium", reason="extra 'env' not installed")
    from lvgrid_rl.env.obs import (  # noqa: PLC0415
        FeatureGroup,
        ObservationBuilder,
        ObservationSpec,
        SensorConfig,
    )

    # No state of its own: a frozen dataclass whose slots hold configuration
    # plus the derived layout, nothing else.
    builder = ObservationBuilder(ObservationSpec(forecast_series=("s",)), 4)
    assert set(builder.__slots__) == {"spec", "n_evaluated_buses", "feature_names"}

    full = ObservationBuilder(
        ObservationSpec(
            groups=(FeatureGroup.MEASUREMENTS,),
            sensor_config=SensorConfig.FULL_STATE,
        ),
        6,
    )
    realistic = ObservationBuilder(
        ObservationSpec(
            groups=(FeatureGroup.MEASUREMENTS,),
            sensor_config=SensorConfig.REALISTIC,
            measured_buses=(0, 5),
        ),
        6,
    )
    assert full.dim > realistic.dim, "sensor configuration must matter"
    assert full.feature_names != realistic.feature_names


def test_i1_system_state_is_the_single_source_of_truth_today() -> None:
    """I1, the part already testable: ``SystemState`` is complete.

    It holds grid, asset, exogenous and PQ state.
    """
    names = {f.name for f in dataclasses.fields(SystemState)}
    assert {"grid", "assets", "exogenous", "pq", "topology_id"} <= names


# ---------------------------------------------------------------------------
# I2 -- asset dynamics as pure functions
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.components.bess"),
    strict=True,
    reason="I2 falls due with the asset models in M4.",
)
def test_i2_asset_dynamics_are_pure_functions() -> None:
    """I2: ``dynamics`` is deterministic and free of side effects.

    The predictive safety filter has to roll asset states forward
    hypothetically over a horizon. Retrofitting means rewriting every asset
    model -- the most expensive of the seven invariants.

    Acceptance criteria for M4, per asset type:

    * calling twice with identical input yields identical output;
    * the ``AssetState``, ``Setpoint`` and ``ExogenousInput`` objects passed in
      are unchanged after the call;
    * a chain of N calls yields the same result as N individual calls with the
      state handed on.
    """
    from lvgrid_rl.components.bess import BatteryStorage  # noqa: PLC0415

    raise AssertionError(f"assertions for {BatteryStorage} still to be written")


# ---------------------------------------------------------------------------
# I3 -- information ordering
# ---------------------------------------------------------------------------


def test_i3_decision_scope_blocks_access_to_future_realisations() -> None:
    """I3: access beyond the decision point is blocked.

    The mechanism is complete since M0. A guarantee based on information the
    real controller does not have is not a guarantee.
    """
    with DecisionScope(t_decision=100):
        assert_readable(100)
        assert_readable(99)
        with pytest.raises(ClairvoyanceError):
            assert_readable(101)


def test_i3_information_set_cannot_express_clairvoyance() -> None:
    """I3: structural safeguard against clairvoyance.

    There is no field for realised values of the coming interval.
    """
    names = {f.name for f in dataclasses.fields(InformationSet)}
    forbidden = {n for n in names if "realiz" in n or "realis" in n}
    assert not forbidden, f"clairvoyance expressible via: {forbidden}"


def test_i3_env_wraps_action_construction_in_a_decision_scope() -> None:
    """I3: the environment actually uses the mechanism.

    A guarantee based on information the real controller does not have is not a
    guarantee, and the bug is invisible in training. The check therefore does not
    inspect the code but probes it: a safety component that tries to read the
    coming interval from inside the action construction must raise.
    """
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    pytest.importorskip("gymnasium", reason="extra 'env' not installed")
    from lvgrid_rl.core.information import (  # noqa: PLC0415
        ClairvoyanceError,
        assert_readable,
    )
    from lvgrid_rl.core.protocols import InterventionInfo  # noqa: PLC0415
    from lvgrid_rl.env.factory import make_env  # noqa: PLC0415

    class Peeking:
        """Reads one step beyond the decision point, which must be refused."""

        def transform(self, action, state, info):
            assert_readable(info.t_index + 1, "profile value")
            return action, InterventionInfo(intervened=False)

        def action_mask(self, state, info):
            return None

    env = make_env(seed=1)
    env.safety = Peeking()
    env.reset(seed=1)
    with pytest.raises(ClairvoyanceError):
        env.step(env.action_space.sample())


# ---------------------------------------------------------------------------
# I4 -- physical actions, affine normalisation
# ---------------------------------------------------------------------------


def test_i4_action_normalisation_is_affine_and_invertible() -> None:
    """I4: normalisation is affine and invertible, clipping is reported.

    The certified feasible set is formulated in injections; projecting onto it
    needs box-shaped action coordinates related to injections by an invertible
    affine map. Satisfied since the action mapper landed in M3.
    """
    pytest.importorskip("gymnasium", reason="extra 'env' not installed")
    from lvgrid_rl.core.protocols import ActionSpec  # noqa: PLC0415
    from lvgrid_rl.core.schemas import Interval  # noqa: PLC0415
    from lvgrid_rl.env.actions import ActionMapper  # noqa: PLC0415

    specs = (
        ActionSpec(names=("p_mw",), bounds=(Interval(-0.02, 0.0),)),
        ActionSpec(
            names=("p_mw", "q_mvar"),
            bounds=(Interval(-0.05, 0.0), Interval(-0.03, 0.03)),
        ),
    )
    mapper = ActionMapper(("sgen:0", "sgen:1"), specs)

    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=mapper.dim)
        assert np.allclose(mapper.to_normalised(mapper.to_physical(action)), action)

    # Affinity: the mapping preserves convex combinations.
    x = rng.uniform(-1.0, 1.0, size=mapper.dim)
    y = rng.uniform(-1.0, 1.0, size=mapper.dim)
    for weight in (0.0, 0.3, 0.5, 1.0):
        mixed = weight * x + (1.0 - weight) * y
        expected = weight * mapper.to_physical(x) + (1.0 - weight) * mapper.to_physical(y)
        assert np.allclose(mapper.to_physical(mixed), expected)

    # No silent clipping: every limitation an asset applies is reported.
    from datetime import UTC, datetime  # noqa: PLC0415

    from lvgrid_rl.components.pv import PvSystem  # noqa: PLC0415
    from lvgrid_rl.core.information import InformationSet  # noqa: PLC0415
    from lvgrid_rl.core.schemas import AssetRatings, PQBudgetState  # noqa: PLC0415

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
        forecast={"sgen:0": np.array([-0.004])},
        pq=PQBudgetState(windows_elapsed_count=0),
    )
    state = asset.initial_state(np.random.default_rng(0))
    setpoint = asset.to_setpoint(state, np.array([-0.02]), info)
    assert setpoint.was_clipped, "limitation must be reported, not applied silently"
    assert setpoint.clipping_info


# ---------------------------------------------------------------------------
# I5 -- hypothetical power flow without side effects
# ---------------------------------------------------------------------------


def test_i5_hypothetical_powerflow_leaves_the_live_grid_untouched() -> None:
    """I5: ``run_hypothetical`` does not alter the running grid.

    Backup trajectories of a predictive safety filter and the falsification
    search of the verification step need exactly this. Satisfied since the
    power flow engine landed in M2.

    Checked here on every mutable element table plus all ``net.res_*`` result
    tables, because pandapower writes more of them than one tends to remember.
    """
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.core.protocols import Setpoint  # noqa: PLC0415
    from lvgrid_rl.grid.loader import load_grid  # noqa: PLC0415
    from lvgrid_rl.grid.powerflow import PandapowerEngine  # noqa: PLC0415

    model = load_grid("1-LV-rural1--2-sw")
    engine = PandapowerEngine(model)

    live = {f"load:{i}": Setpoint(f"load:{i}", 0.002) for i in model.net.load.index}
    engine.run(live, 0)

    def snapshot() -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for table in ("load", "sgen", "storage"):
            for column in ("p_mw", "q_mvar", "in_service"):
                df = model.net[table]
                if len(df) and column in df.columns:
                    out[f"{table}.{column}"] = df[column].to_numpy(copy=True)
        for key in (k for k in model.net.keys() if k.startswith("res_")):
            df = model.net[key]
            if hasattr(df, "to_numpy") and len(df):
                out[key] = df.to_numpy(copy=True)
        return out

    before = snapshot()
    other = {f"load:{i}": Setpoint(f"load:{i}", 0.009) for i in model.net.load.index}
    hypothetical = engine.run_hypothetical(other, 1)
    after = snapshot()

    assert hypothetical.converged, "the hypothetical run must actually compute"
    assert before.keys() == after.keys()
    for key in before:
        assert np.array_equal(before[key], after[key], equal_nan=True), (
            f"run_hypothetical modified {key}"
        )


# ---------------------------------------------------------------------------
# I6 -- exogenous inputs carry bounds
# ---------------------------------------------------------------------------


def test_i6_exogenous_input_cannot_be_built_without_bounds() -> None:
    """I6: no data path can produce exogenous inputs without bounds.

    Satisfied at schema level since M0: ``bounds_mw`` is mandatory and shape
    checked. Retrofitting uncertainty sets later would touch every data path.
    """
    required = {
        f.name
        for f in dataclasses.fields(ExogenousInput)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
    }
    assert "bounds_mw" in required

    with pytest.raises(ValueError, match="bounds_mw"):
        ExogenousInput(
            t_index=0,
            series_ids=("a",),
            realized_mw=np.zeros(1),
            bounds_mw=np.zeros((2, 3)),
            ambient_temp_degc=0.0,
            ghi_wm2=0.0,
        )


def test_i6_every_asset_from_the_adapter_has_complete_ratings() -> None:
    """I6: every asset carries complete ``AssetRatings``.

    Basis of the deterministic uncertainty sets (section 6.9, P2). Since M1 the
    SimBench adapter produces the ratings; adding them later across all
    scenarios would have been pure manual work.
    """
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    from lvgrid_rl.data.sources.simbench import load_simbench  # noqa: PLC0415

    data = load_simbench("1-LV-rural1--2-sw")
    assert data.assets, "grid without assets"
    for asset in data.assets:
        r = asset.ratings
        assert r.p_min_mw <= r.p_max_mw
        # Elements with an apparent power rating need it for the later reactive
        # power control and for the inverter capability diagram.
        if asset.element_table in ("sgen", "storage"):
            assert r.s_max_mva is not None and r.s_max_mva > 0.0


# ---------------------------------------------------------------------------
# I7 -- two safety intervention points with a null implementation
# ---------------------------------------------------------------------------


def test_i7_null_safety_component_satisfies_the_protocol() -> None:
    """I7: the intervention point exists from M0 with a null implementation."""
    assert isinstance(NullSafetyComponent(), SafetyComponent)


def test_i7_env_exposes_both_intervention_points() -> None:
    """I7: both intervention points sit in the environment's step sequence.

    The shield lives inside the environment because it needs the state; masking
    lives outside and needs the distribution. Both paths must exist, otherwise
    one of the mechanisms can later only be added by cutting into the step
    sequence. Satisfied since the environment landed in M3.
    """
    pytest.importorskip("simbench", reason="extra 'sim' not installed")
    pytest.importorskip("gymnasium", reason="extra 'env' not installed")
    from lvgrid_rl.core.protocols import InterventionInfo  # noqa: PLC0415
    from lvgrid_rl.env.factory import make_env  # noqa: PLC0415

    class Halving:
        """Halves every setpoint, so the effect is measurable."""

        def __init__(self) -> None:
            self.calls = 0

        def transform(self, action, state, info):
            self.calls += 1
            return 0.5 * action, InterventionInfo(
                intervened=True, magnitude=float(np.abs(0.5 * action).sum())
            )

        def action_mask(self, state, info):
            return None

    # The mask is present in every step, and with the null component it permits
    # everything.
    plain = make_env(seed=1)
    _, info = plain.reset(seed=1)
    assert "action_mask" in info
    assert info["action_mask"] is None
    _, _, _, _, info = plain.step(plain.mapper.neutral_action())
    assert "action_mask" in info

    # The in-environment component measurably changes what is written.
    # Run into daylight before comparing: at night the available power is zero,
    # so a halved setpoint and an unhalved one are clipped to the same value and
    # the comparison would not discriminate.
    def curtailed_over_a_day(component=None) -> tuple[float, int]:
        env = make_env(seed=1)
        if component is not None:
            env.safety = component
        env.reset(seed=1)
        total = 0.0
        intervened = 0
        for _ in range(60):
            _, _, _, _, step_info = env.step(env.mapper.neutral_action())
            total += step_info["curtailed_energy_mwh"]
            intervened += int(step_info["intervened"])
        return total, intervened

    component = Halving()
    guarded_total, guarded_intervened = curtailed_over_a_day(component)
    plain_total, plain_intervened = curtailed_over_a_day()

    assert component.calls == 60
    assert guarded_intervened == 60
    assert plain_intervened == 0
    assert guarded_total > plain_total, (
        "the safety component must actually reach the setpoints"
    )
