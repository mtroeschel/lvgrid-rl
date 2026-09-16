"""Tests for the grid layer (M2)."""

from __future__ import annotations

import numpy as np
import pytest

from lvgrid_rl.core.protocols import PowerFlowEngine, Setpoint
from lvgrid_rl.grid.loader import (
    ZIP_COLUMNS,
    derive_connection_points,
    fix_zip_load_model,
    load_grid,
)
from lvgrid_rl.grid.powerflow import PandapowerEngine

pytest.importorskip("simbench", reason="extra 'sim' not installed")
pp = pytest.importorskip("pandapower")

CODE = "1-LV-rural1--2-sw"


@pytest.fixture(scope="module")
def model():
    """Prepared grid, loaded once per test module."""
    return load_grid(CODE)


@pytest.fixture(scope="module")
def setpoint_factory(model):
    """Build setpoints for a time step from the SimBench profiles."""
    import simbench as sb

    values = sb.get_absolute_values(model.net, profiles_instead_of_study_cases=True)
    lp, lq = values[("load", "p_mw")], values[("load", "q_mvar")]
    sp = values[("sgen", "p_mw")]

    def build(t: int) -> dict[str, Setpoint]:
        out: dict[str, Setpoint] = {}
        lpv, lqv, spv = lp.loc[t].values, lq.loc[t].values, sp.loc[t].values
        for pos, idx in enumerate(model.net.load.index):
            out[f"load:{idx}"] = Setpoint(f"load:{idx}", float(lpv[pos]), float(lqv[pos]))
        for pos, idx in enumerate(model.net.sgen.index):
            # Consumer sign convention: infeed is negative.
            out[f"sgen:{idx}"] = Setpoint(f"sgen:{idx}", -float(spv[pos]))
        return out

    return build


# ---------------------------------------------------------------------------
# ZIP load model repair
# ---------------------------------------------------------------------------


def test_raw_simbench_net_has_undefined_zip_columns() -> None:
    """Documents the defect this repair exists for.

    If a future simbench release fills these columns, this test turns red and
    the repair can be removed -- which is the point of asserting on the defect
    rather than only on the fix.
    """
    import simbench as sb

    net = sb.get_simbench_net(CODE)
    assert net.load[list(ZIP_COLUMNS)].isna().all().all()


def test_raw_simbench_net_does_not_converge() -> None:
    """The failure mode is misleading.

    It looks like an infeasible operating point, but it is a data defect.
    """
    import simbench as sb

    net = sb.get_simbench_net(CODE)
    with pytest.raises(pp.LoadflowNotConverged):
        pp.runpp(net, numba=False)


def test_repaired_net_converges(model) -> None:
    assert model.zip_rows_repaired > 0
    pp.runpp(model.net, numba=False)
    assert model.net.converged


def test_repair_is_idempotent(model) -> None:
    """A second call must touch nothing.

    That keeps the repair a no-op once simbench fills the columns itself.
    """
    assert fix_zip_load_model(model.net) == 0


# ---------------------------------------------------------------------------
# Connection points (decision D7)
# ---------------------------------------------------------------------------


def test_connection_points_include_generator_and_storage_buses(model) -> None:
    net = model.net
    expected = set()
    for table in ("load", "sgen", "storage"):
        expected.update(int(b) for b in net[table]["bus"])
    assert set(model.connection_point_buses) == expected


def test_not_every_bus_is_a_connection_point(model) -> None:
    """Not every bus is a point of common coupling.

    Cable distributors and the transformer's MV bus are not, so EN 50160 is not
    assessed there.
    """
    assert model.n_evaluated_buses < len(model.net.bus)


def test_excluded_buses_are_dropped(model) -> None:
    first = model.connection_point_buses[0]
    reduced = derive_connection_points(model.net, exclude_buses=frozenset({first}))
    assert first not in reduced
    assert len(reduced) == model.n_evaluated_buses - 1


def test_evaluated_positions_match_the_bus_order(model) -> None:
    labels = model.net.bus.index.to_numpy()[model.evaluated_bus_positions]
    assert list(labels) == list(model.connection_point_buses)


# ---------------------------------------------------------------------------
# Power flow engine
# ---------------------------------------------------------------------------


def test_engine_satisfies_the_protocol(model) -> None:
    assert isinstance(PandapowerEngine(model), PowerFlowEngine)


def test_run_returns_a_converged_state(model, setpoint_factory) -> None:
    engine = PandapowerEngine(model)
    state = engine.run(setpoint_factory(16564), 16564)
    assert state.converged
    assert state.vm_pu.shape == (len(model.net.bus),)
    assert np.isfinite(state.losses_mw)
    assert state.vm_pu.flags.writeable is False


def test_sign_flip_happens_only_for_generators(model, setpoint_factory) -> None:
    """Exactly one sign flip, and it happens in the engine.

    Internally the consumer reference direction is used throughout; pandapower
    counts sgen the other way.
    """
    engine = PandapowerEngine(model)
    engine.run(setpoint_factory(16564), 16564)
    assert (model.net.sgen.p_mw >= 0).all(), "infeed must be positive in pandapower"
    assert (model.net.load.p_mw >= 0).all()


def test_hypothetical_run_leaves_the_live_grid_untouched(model, setpoint_factory) -> None:
    """Invariant I5. Backup trajectories and falsification search need this."""
    engine = PandapowerEngine(model)
    engine.run(setpoint_factory(16564), 16564)

    before_elements = {
        table: model.net[table]["p_mw"].to_numpy(copy=True) for table in ("load", "sgen")
    }
    before_results = model.net.res_bus.vm_pu.to_numpy(copy=True)

    engine.run_hypothetical(setpoint_factory(0), 0)

    for table, values in before_elements.items():
        assert np.array_equal(values, model.net[table]["p_mw"].to_numpy())
    assert np.array_equal(before_results, model.net.res_bus.vm_pu.to_numpy())


def test_hypothetical_run_actually_evaluates_the_given_setpoints(
    model, setpoint_factory
) -> None:
    """The counterpart to the previous test.

    No side effects, but also not a no-op.
    """
    engine = PandapowerEngine(model)
    noon = engine.run(setpoint_factory(16564), 16564)
    night = engine.run_hypothetical(setpoint_factory(0), 0)
    assert night.converged
    assert float(np.nanmax(night.vm_pu)) < float(np.nanmax(noon.vm_pu))


def test_non_convergence_is_reported_not_raised(model) -> None:
    """A single numerically awkward step must not abort a training run."""
    engine = PandapowerEngine(model)
    absurd = {
        f"load:{idx}": Setpoint(f"load:{idx}", 50.0) for idx in model.net.load.index
    }
    state = engine.run(absurd, 0)
    assert state.converged is False
    assert np.isnan(state.p_slack_mw)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_reference_scenario_changes_nothing(model) -> None:
    assert model.scenario.name == "reference"
    assert model.scenario_modifications == {}


def test_slack_voltage_is_applied() -> None:
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    m = load_grid(CODE, SCENARIO_LIBRARY["weak_connection"])
    assert float(m.net.ext_grid.vm_pu.iloc[0]) == pytest.approx(1.04)
    assert m.scenario_modifications["ext_grid"] == 1


def test_scaling_applies_per_asset_category() -> None:
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    # Deliberately a fresh grid rather than the module fixture: other tests run
    # the power flow on that one, which overwrites the nominal values.
    pv_before = float(load_grid(CODE).net.sgen.p_mw.sum())
    m = load_grid(CODE, SCENARIO_LIBRARY["moderate_growth"])
    assert float(m.net.sgen.p_mw.sum()) == pytest.approx(1.5 * pv_before)
    # Households must stay untouched: growth applies to PV, heat pumps and
    # charge points only.
    assert m.scenario_modifications["sgen:pv"] == len(m.net.sgen)
    assert "load:household" not in m.scenario_modifications


def test_transformer_derating_is_applied() -> None:
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    base = load_grid(CODE)
    weak = load_grid(CODE, SCENARIO_LIBRARY["weak_connection"])
    assert float(weak.net.trafo.sn_mva.sum()) == pytest.approx(
        0.8 * float(base.net.trafo.sn_mva.sum())
    )


def test_implausible_slack_voltage_is_rejected() -> None:
    """Implausible slack voltages are rejected.

    A grid already in violation before any control acts is a configuration
    error, not a scenario.
    """
    from lvgrid_rl.grid.scenario import ScenarioSpec

    with pytest.raises(ValueError, match="plausible range"):
        ScenarioSpec(name="broken", slack_vm_pu=1.3)


def test_scenario_description_is_flat_and_serialisable() -> None:
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    described = SCENARIO_LIBRARY["moderate_growth"].describe()
    assert described["slack_vm_pu"] == 1.04
    assert described["scale"]["pv"] == 1.5
    assert isinstance(described["notes"], str)


def test_extreme_scenarios_are_marked_as_such() -> None:
    """Extreme scenarios must declare themselves.

    Otherwise they end up in a results table looking like a forecast.
    """
    from lvgrid_rl.grid.scenario import SCENARIO_LIBRARY

    for name in ("voltage_stress", "undervoltage_stress"):
        assert "beyond" in SCENARIO_LIBRARY[name].notes or "Counterpart" in (
            SCENARIO_LIBRARY[name].notes
        )
