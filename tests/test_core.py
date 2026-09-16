"""Tests der Kernschemata, des Einheitenkanons und der Reproduzierbarkeit.

Diese Tests pruefen, was in M0 bereits existiert. Die Invarianten des
Erweiterbarkeits-Contracts stehen in ``test_invariants.py``.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from lvgrid_rl.core import information
from lvgrid_rl.core.information import (
    ClairvoyanceError,
    DecisionScope,
    InformationSet,
    assert_readable,
    unrestricted,
)
from lvgrid_rl.core.protocols import ActionSpec, NullSafetyComponent, Setpoint, Verdict
from lvgrid_rl.core.schemas import (
    AssetRatings,
    ExogenousInput,
    GridState,
    Interval,
    PQBudgetState,
    PQWindowState,
    SystemState,
    freeze_array,
)
from lvgrid_rl.core.units import UNIT_SUFFIXES, unit_of
from lvgrid_rl.experiment.reproducibility import (
    RunManifest,
    SeedSet,
    canonical_json,
    compute_run_id,
    config_hash,
)

# ---------------------------------------------------------------------------
# Hilfsmittel
# ---------------------------------------------------------------------------


def make_exogenous(n: int = 2, t_index: int = 0) -> ExogenousInput:
    """Minimaler, gueltiger exogener Eingang."""
    realized = np.array([0.003, -0.004][:n], dtype=float)
    bounds = np.vstack([realized - 0.001, realized + 0.001])
    return ExogenousInput(
        t_index=t_index,
        series_ids=tuple(f"series_{i}" for i in range(n)),
        realized_mw=realized,
        bounds_mw=bounds,
        ambient_temp_degc=7.5,
        ghi_wm2=180.0,
    )


def make_grid_state(n_bus: int = 3) -> GridState:
    return GridState(
        t_index=0,
        vm_pu=np.full(n_bus, 1.01),
        line_loading_percent=np.full(n_bus - 1, 42.0),
        trafo_loading_percent=np.array([55.0]),
        p_slack_mw=0.01,
        losses_mw=0.0004,
        converged=True,
    )


# ---------------------------------------------------------------------------
# Einheitenkanon
# ---------------------------------------------------------------------------

SCHEMA_TYPES = [
    Interval,
    AssetRatings,
    ExogenousInput,
    GridState,
    PQWindowState,
    PQBudgetState,
    SystemState,
    InformationSet,
    Setpoint,
    ActionSpec,
]

# Felder ohne physikalische Einheit: Indizes, Namen, Flags, verschachtelte
# Schemata. Bewusst als explizite Liste, damit ein neues Feld ohne Einheit eine
# Entscheidung erzwingt und nicht durch eine Heuristik durchrutscht.
UNITLESS_FIELDS = {
    "t_index",
    "timestamp",
    "topology_id",
    "asset_id",
    "series_ids",
    "names",
    "bounds",
    "discrete_levels",
    "grid",
    "assets",
    "asset_states",
    "exogenous",
    "pq",
    "converged",
    "measurements",
    "forecast",
    "windows",
    "clipping_info",
    "lo",
    "hi",
}


@pytest.mark.parametrize("cls", SCHEMA_TYPES, ids=lambda c: c.__name__)
def test_numeric_schema_fields_carry_a_unit_suffix(cls: type) -> None:
    """Jedes numerische Schemafeld traegt ein bekanntes Einheiten-Suffix.

    Umsetzung des Einheitenkanons (Abschnitt 13). Verhindert genau die Klasse
    von Fehlern, die spaeter in Intervallrechnungen stillschweigend falsche
    Ergebnisse liefert.
    """
    offenders = [
        f.name
        for f in dataclasses.fields(cls)
        if f.name not in UNITLESS_FIELDS and unit_of(f.name) is None
    ]
    assert not offenders, (
        f"{cls.__name__}: Felder ohne Einheiten-Suffix: {offenders}. "
        f"Erlaubte Suffixe: {sorted(UNIT_SUFFIXES)}"
    )


def test_unit_of_prefers_the_longest_suffix() -> None:
    """``_mwh`` darf nicht als ``_mw`` gelesen werden."""
    assert unit_of("energy_mwh") == "_mwh"
    assert unit_of("p_slack_mw") == "_mw"
    assert unit_of("etwas_ohne_einheit") is None


# ---------------------------------------------------------------------------
# Unveraenderlichkeit
# ---------------------------------------------------------------------------


def test_frozen_arrays_are_not_writeable() -> None:
    """``frozen=True`` schuetzt die Referenz, nicht den Arrayinhalt."""
    state = make_grid_state()
    with pytest.raises(ValueError):
        state.vm_pu[0] = 42.0


def test_freeze_array_does_not_copy() -> None:
    """Der heisse Pfad soll keine Kopie je Zeitschritt erzeugen."""
    base = np.array([1.0, 2.0])
    view = freeze_array(base)
    assert view.base is base


def test_schema_instances_reject_attribute_assignment() -> None:
    state = make_grid_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.p_slack_mw = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExogenousInput: Schranken sind Pflicht (Invariante I6, Schemaebene)
# ---------------------------------------------------------------------------


def test_exogenous_input_requires_matching_bounds() -> None:
    with pytest.raises(ValueError, match="bounds_mw"):
        ExogenousInput(
            t_index=0,
            series_ids=("a", "b"),
            realized_mw=np.zeros(2),
            bounds_mw=np.zeros((2, 1)),
            ambient_temp_degc=0.0,
            ghi_wm2=0.0,
        )


def test_exogenous_input_rejects_realisation_outside_bounds() -> None:
    """Realisierungen ausserhalb der Schranken sind ein Fehler.

    Sie wuerden jede spaetere Sicherheitsaussage auf ihrer Basis entwerten.
    """
    with pytest.raises(ValueError, match="outside the bounds"):
        ExogenousInput(
            t_index=0,
            series_ids=("a",),
            realized_mw=np.array([5.0]),
            bounds_mw=np.array([[0.0], [1.0]]),
            ambient_temp_degc=0.0,
            ghi_wm2=0.0,
        )


def test_bound_of_returns_the_series_interval() -> None:
    x = make_exogenous()
    interval = x.bound_of("series_0")
    assert interval.contains(float(x.realized_mw[0]))


def test_asset_ratings_expose_bounds_for_uncertainty_sets() -> None:
    """PV im Verbraucher-Zaehlpfeil: Einspeisung ist negativ."""
    pv = AssetRatings(p_min_mw=-0.01, p_max_mw=0.0, s_max_mva=0.011)
    assert pv.p_bounds == Interval(-0.01, 0.0)
    with pytest.raises(ValueError):
        AssetRatings(p_min_mw=1.0, p_max_mw=0.0)


# ---------------------------------------------------------------------------
# EN-50160-Budget
# ---------------------------------------------------------------------------


def test_k95_budget_is_fifty_windows_per_week() -> None:
    """1008 Zehn-Minuten-Fenster je Woche, 5 Prozent davon sind 50."""
    pq = PQBudgetState(windows_elapsed_count=0)
    assert pq.windows_total_count == 7 * 24 * 6 == 1008
    assert pq.budget_windows_count == 50


def test_budget_utilisation_can_exceed_one() -> None:
    """Ueberschreitung muss darstellbar sein, sonst ist die Kennzahl blind."""
    pq = PQBudgetState(
        windows_elapsed_count=600,
        violations_k95_count=np.array([0, 25, 75], dtype=np.int32),
    )
    used = pq.budget_used_frac()
    assert used[0] == 0.0
    assert used[1] == pytest.approx(0.5)
    assert used[2] == pytest.approx(1.5)
    assert pq.windows_remaining_count() == 408


def test_open_window_mean_is_nan_when_empty() -> None:
    assert np.isnan(PQWindowState(samples_count=0, partial_sum_pu=0.0).mean_pu)
    assert PQWindowState(samples_count=2, partial_sum_pu=2.0).mean_pu == 1.0


# ---------------------------------------------------------------------------
# SystemState
# ---------------------------------------------------------------------------


def test_system_state_requires_timezone_aware_timestamp() -> None:
    """Naive Zeitstempel erzeugen bei Sommerzeitwechseln stille Fehler."""
    with pytest.raises(ValueError, match="UTC"):
        SystemState(
            t_index=0,
            timestamp=datetime(2016, 6, 1, 12, 0),
            topology_id="base",
            grid=make_grid_state(),
            assets={},
            exogenous=make_exogenous(),
            pq=PQBudgetState(windows_elapsed_count=0),
        )


def test_system_state_accepts_utc() -> None:
    state = SystemState(
        t_index=0,
        timestamp=datetime(2016, 6, 1, 12, 0, tzinfo=UTC),
        topology_id="base",
        grid=make_grid_state(),
        assets={},
        exogenous=make_exogenous(),
        pq=PQBudgetState(windows_elapsed_count=0),
    )
    assert state.t_index == 0


# ---------------------------------------------------------------------------
# Informationsordnung (Invariante I3, Mechanismus-Ebene)
# ---------------------------------------------------------------------------


def test_access_beyond_decision_horizon_raises() -> None:
    with pytest.raises(ClairvoyanceError, match="I3"):
        with DecisionScope(t_decision=10):
            assert_readable(11, "PV-Potenzial")


def test_access_up_to_decision_horizon_is_allowed() -> None:
    with DecisionScope(t_decision=10) as scope:
        assert_readable(10)
        assert_readable(3)
    assert scope.accessed == (10, 3)


def test_no_restriction_outside_a_decision_scope() -> None:
    """Simulation, Auswertung und Zertifizierung duerfen den Vollzustand sehen."""
    assert_readable(10_000)
    assert information.current_decision_horizon() is None


def test_unrestricted_lifts_the_restriction_for_privileged_components() -> None:
    """Fuer ``mpc_oracle`` mit perfekter Vorausschau, explizit benannt."""
    with DecisionScope(t_decision=10):
        with unrestricted():
            assert_readable(50)
        with pytest.raises(ClairvoyanceError):
            assert_readable(50)


def test_decision_scopes_restore_the_previous_horizon() -> None:
    with DecisionScope(t_decision=10):
        with DecisionScope(t_decision=5):
            assert_readable(5)
        assert_readable(10)


def test_information_set_has_no_field_for_realised_future_values() -> None:
    """Strukturelle Absicherung: Hellsichtigkeit soll nicht ausdrueckbar sein."""
    names = {f.name for f in dataclasses.fields(InformationSet)}
    assert not any("realiz" in n or "realis" in n for n in names), names
    assert "exogenous_bounds_mw" in names
    assert "forecast" in names
    assert "pq" in names


# ---------------------------------------------------------------------------
# Sicherheits-Eingriffspunkte (Invariante I7, Null-Implementierung)
# ---------------------------------------------------------------------------


def test_null_safety_component_passes_actions_through() -> None:
    comp = NullSafetyComponent()
    action = np.array([0.5, -0.25])
    out, info = comp.transform(action, None, None)  # type: ignore[arg-type]
    assert np.array_equal(out, action)
    assert info.intervened is False
    assert comp.action_mask(None, None) is None  # type: ignore[arg-type]


def test_verdict_has_no_unsafe_value() -> None:
    """Es gibt kein ``UNSAFE``, und das ist Absicht.

    Ein Zertifizierer darf konservativ sein, aber nie optimistisch
    (Abschnitt 6.9).
    """
    assert {v.name for v in Verdict} == {"CERTIFIED_SAFE", "NOT_CERTIFIED"}


def test_setpoint_reports_clipping_explicitly() -> None:
    assert not Setpoint(asset_id="pv_1", p_mw=-0.005).was_clipped
    assert Setpoint(
        asset_id="pv_1", p_mw=-0.005, clipping_info={"p_mw": -0.001}
    ).was_clipped


def test_action_spec_rejects_inconsistent_definitions() -> None:
    with pytest.raises(ValueError):
        ActionSpec(names=("p_mw", "q_mvar"), bounds=(Interval(-1.0, 1.0),))


# ---------------------------------------------------------------------------
# Reproduzierbarkeit
# ---------------------------------------------------------------------------


def test_seed_derivation_is_deterministic_and_purpose_separated() -> None:
    a, b = SeedSet.from_base(42), SeedSet.from_base(42)
    assert a == b
    values = {a.scenario, a.episode, a.train, a.eval, a.forecast}
    assert len(values) == 5, "Zweck-Seeds muessen sich unterscheiden"


def test_different_base_seeds_give_different_scenario_seeds() -> None:
    assert SeedSet.from_base(1).scenario != SeedSet.from_base(2).scenario


def test_generator_rejects_unknown_purposes() -> None:
    """Tippfehler sollen auffallen, nicht stillschweigend einen Stream liefern."""
    with pytest.raises(KeyError, match="Unbekannter Seed-Zweck"):
        SeedSet.from_base(0).generator("trian")


def test_worker_generators_are_independent_but_reproducible() -> None:
    seeds = SeedSet.from_base(7)
    first = seeds.worker_generator("train", 0).random(5)
    second = seeds.worker_generator("train", 1).random(5)
    again = seeds.worker_generator("train", 0).random(5)
    assert not np.allclose(first, second)
    assert np.allclose(first, again)


def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})


def test_config_hash_detects_value_changes() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_canonical_json_is_stable() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_run_id_depends_on_all_four_components() -> None:
    base = ("commit", "cfg", "data", 1)
    reference = compute_run_id(*base)
    assert len(reference) == 12
    assert compute_run_id("other", "cfg", "data", 1) != reference
    assert compute_run_id("commit", "other", "data", 1) != reference
    assert compute_run_id("commit", "cfg", "other", 1) != reference
    assert compute_run_id("commit", "cfg", "data", 2) != reference


def test_manifest_round_trips_to_disk(tmp_path) -> None:
    manifest = RunManifest.create(
        config={"grid": "1-LV-rural1", "env": {"sim_dt_min": 5}},
        data_manifest_hash="deadbeef",
        base_seed=1,
        package_versions={"numpy": np.__version__},
    )
    path = manifest.write(tmp_path / "run")
    assert path.exists()
    import json

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == manifest.run_id
    assert loaded["seeds"]["base"] == 1
