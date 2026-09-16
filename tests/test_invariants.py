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


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.obs"),
    strict=True,
    reason="I1 falls due with the ObservationBuilder in M3.",
)
def test_i1_observation_is_a_pure_projection_of_system_state() -> None:
    """I1: the ``ObservationBuilder`` reads only from ``SystemState``.

    The certifier needs the full state, the agent may see less. If observation
    and state are the same object that cannot be separated, and retrofitting
    touches every environment component.

    Acceptance criteria for M3:

    * ``Observation`` has no setters and no state of its own;
    * two ``ObservationBuilder`` instances with different ``sensor_config``
      produce different observations from the same ``SystemState``;
    * the builder holds no state between calls that is not reconstructible from
      ``SystemState``.
    """
    from lvgrid_rl.env.obs import ObservationBuilder  # noqa: PLC0415

    raise AssertionError(f"assertions for {ObservationBuilder} still to be written")


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


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.lv_grid_env"),
    strict=True,
    reason="I3 for the environment falls due in M3.",
)
def test_i3_env_wraps_action_construction_in_a_decision_scope() -> None:
    """I3: the environment actually uses the mechanism.

    Acceptance criterion for M3: a spy scenario recording its accesses must not
    see any time step after ``t`` during steps 1 and 2 of the step sequence.
    """
    from lvgrid_rl.env.lv_grid_env import LVGridEnv  # noqa: PLC0415

    raise AssertionError(f"assertions for {LVGridEnv} still to be written")


# ---------------------------------------------------------------------------
# I4 -- physical actions, affine normalisation
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.actions"),
    strict=True,
    reason="I4 falls due with the ActionMapper in M3.",
)
def test_i4_action_normalisation_is_affine_and_invertible() -> None:
    """I4: normalisation is affine and invertible, clipping is reported.

    The certified feasible set is formulated in injections; projecting onto it
    needs box-shaped action coordinates. Semantics such as "state-of-charge
    target" break that.

    Acceptance criteria for M3:

    * ``from_normalised(to_normalised(a)) == a`` for random actions;
    * linearity: the mapping preserves convex combinations;
    * no asset model clips silently -- every limitation appears in
      ``Setpoint.clipping_info``.
    """
    from lvgrid_rl.env.actions import ActionMapper  # noqa: PLC0415

    raise AssertionError(f"assertions for {ActionMapper} still to be written")


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


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.lv_grid_env"),
    strict=True,
    reason="I7 for the environment falls due in M3.",
)
def test_i7_env_exposes_both_intervention_points() -> None:
    """I7: both intervention points sit in the environment's step sequence.

    The shield lives inside the environment because it needs the state; masking
    lives outside and needs the distribution. Both paths must exist, otherwise
    one of the mechanisms can later only be added by cutting into the step
    sequence.

    Acceptance criteria for M3:

    * a dummy ``SafetyComponent`` that halves actions measurably changes the
      setpoints written;
    * ``info["action_mask"]`` is present at every step and permits all actions
      when ``safety.mechanism: none``.
    """
    from lvgrid_rl.env.lv_grid_env import LVGridEnv  # noqa: PLC0415

    raise AssertionError(f"assertions for {LVGridEnv} still to be written")
