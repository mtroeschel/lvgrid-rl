"""Die sieben Invarianten des Erweiterbarkeits-Contracts (Abschnitt 13).

**Warum diese Datei existiert.** Entscheidung D10 verschiebt die
Zertifizierung auf M7b, verlangt aber, dass die Architektur sie spaeter
aufnehmen kann. Diese Zusage ist nur belastbar, wenn sie geprueft wird:
Modularitaetsversprechen verfallen still. Sechs Monate ohne Test, und
irgendeine sinnvolle Abkuerzung hat die Erweiterbarkeit aufgebraucht, ohne dass
es jemand gemerkt haette.

**Warum ``xfail(strict=True)`` und keine rot fehlschlagenden Platzhalter.**
Eine dauerhaft rote CI wird nach zwei Wochen ignoriert; damit waere der Zweck
verfehlt. ``strict=True`` dreht die Logik um: der Test *darf* nicht bestehen,
solange die Komponente fehlt. Sobald sie implementiert ist und der Test
unerwartet gruen wird, **bricht der Build** und erzwingt, den Marker zu
entfernen. Die Anzahl der XFAIL-Meldungen ist damit der Schuldenstand des
Contracts, ablesbar in jedem CI-Lauf.

Stand: I3 und I6 sind erfuellt (I6 seit dem SimBench-Adapter in M1), I5 wird in
M2 faellig, I1, I4 und I7 in M3, I2 in M4.
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
    """Existiert ein Modul, das eine noch offene Invariante erfuellen wuerde?"""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        return False
    return True


# ---------------------------------------------------------------------------
# I1 -- SystemState vollstaendig und von Observation getrennt
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.obs"),
    strict=True,
    reason="I1 wird mit dem ObservationBuilder in M3 faellig.",
)
def test_i1_observation_is_a_pure_projection_of_system_state() -> None:
    """I1: Der ``ObservationBuilder`` liest ausschliesslich aus ``SystemState``.

    Der Zertifizierer braucht den vollen Zustand, der Agent darf weniger sehen.
    Sind Beobachtung und Zustand dasselbe Objekt, ist das nicht trennbar, und
    die Nachruestung betrifft jede Env-Komponente.

    Pruefkriterien fuer M3:

    * ``Observation`` hat keine Setter und keinen eigenen Zustand;
    * zwei ``ObservationBuilder`` mit unterschiedlicher ``sensor_config``
      liefern aus demselben ``SystemState`` unterschiedliche Beobachtungen;
    * der Builder haelt zwischen Aufrufen keinen Zustand, der nicht aus
      ``SystemState`` rekonstruierbar ist.
    """
    from lvgrid_rl.env.obs import ObservationBuilder  # noqa: PLC0415

    raise AssertionError(f"Pruefung fuer {ObservationBuilder} ausstehend")


def test_i1_system_state_is_the_single_source_of_truth_today() -> None:
    """I1, bereits pruefbarer Teil: ``SystemState`` ist vollstaendig.

    Er enthaelt Netz-, Anlagen-, exogenen und PQ-Zustand.
    """
    names = {f.name for f in dataclasses.fields(SystemState)}
    assert {"grid", "assets", "exogenous", "pq", "topology_id"} <= names


# ---------------------------------------------------------------------------
# I2 -- Anlagendynamik als reine Funktion
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.components.bess"),
    strict=True,
    reason="I2 wird mit den Anlagenmodellen in M4 faellig.",
)
def test_i2_asset_dynamics_are_pure_functions() -> None:
    """I2: ``dynamics`` ist determiniert und seiteneffektfrei.

    Der praediktive Sicherheitsfilter muss Anlagenzustaende ueber einen
    Horizont hypothetisch fortschreiben. Nachruestung bedeutet Neuschreiben
    aller Anlagenmodelle -- die teuerste der sieben Invarianten.

    Pruefkriterien fuer M4, je Anlagentyp:

    * zweimaliger Aufruf mit identischer Eingabe liefert identische Ausgabe;
    * die uebergebenen ``AssetState``-, ``Setpoint``- und
      ``ExogenousInput``-Objekte sind nach dem Aufruf unveraendert;
    * eine Kette von N Aufrufen liefert dasselbe Ergebnis wie N Einzelaufrufe
      mit weitergegebenem Zustand.
    """
    from lvgrid_rl.components.bess import BatteryStorage  # noqa: PLC0415

    raise AssertionError(f"Pruefung fuer {BatteryStorage} ausstehend")


# ---------------------------------------------------------------------------
# I3 -- Informationsordnung
# ---------------------------------------------------------------------------


def test_i3_decision_scope_blocks_access_to_future_realisations() -> None:
    """I3: Zugriffe hinter den Entscheidungszeitpunkt werden unterbunden.

    Der Mechanismus ist in M0 vollstaendig vorhanden. Eine Garantie auf Basis
    von Information, die der reale Regler nicht hat, ist keine.
    """
    with DecisionScope(t_decision=100):
        assert_readable(100)
        assert_readable(99)
        with pytest.raises(ClairvoyanceError):
            assert_readable(101)


def test_i3_information_set_cannot_express_clairvoyance() -> None:
    """I3: strukturelle Absicherung gegen Hellsichtigkeit.

    Es gibt kein Feld fuer realisierte Werte des kommenden Intervalls.
    """
    names = {f.name for f in dataclasses.fields(InformationSet)}
    forbidden = {n for n in names if "realiz" in n or "realis" in n}
    assert not forbidden, f"Hellsichtigkeit ausdrueckbar ueber: {forbidden}"


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.lv_grid_env"),
    strict=True,
    reason="I3 fuer die Env wird in M3 faellig.",
)
def test_i3_env_wraps_action_construction_in_a_decision_scope() -> None:
    """I3: die Environment nutzt den Mechanismus auch tatsaechlich.

    Pruefkriterium fuer M3: ein Spion-Szenario, dessen Zugriffe protokolliert
    werden, darf waehrend Schritt 1 und 2 des Ablaufs keinen Zeitpunkt nach
    ``t`` sehen.
    """
    from lvgrid_rl.env.lv_grid_env import LVGridEnv  # noqa: PLC0415

    raise AssertionError(f"Pruefung fuer {LVGridEnv} ausstehend")


# ---------------------------------------------------------------------------
# I4 -- Physikalische Aktionen, affine Normierung
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.actions"),
    strict=True,
    reason="I4 wird mit dem ActionMapper in M3 faellig.",
)
def test_i4_action_normalisation_is_affine_and_invertible() -> None:
    """I4: Normierung ist affin und invertierbar, Begrenzung wird gemeldet.

    Die zertifizierte Zulaessigkeitsmenge ist in Einspeisungen formuliert;
    eine Projektion darauf braucht boxfoermige Aktionskoordinaten. Semantiken
    wie "SoC-Zielwert" verletzen das.

    Pruefkriterien fuer M3:

    * ``from_normalised(to_normalised(a)) == a`` fuer Zufallsaktionen;
    * Linearitaet: die Abbildung erhaelt Konvexkombinationen;
    * kein Anlagenmodell begrenzt still -- jede Begrenzung erscheint in
      ``Setpoint.clipping_info``.
    """
    from lvgrid_rl.env.actions import ActionMapper  # noqa: PLC0415

    raise AssertionError(f"Pruefung fuer {ActionMapper} ausstehend")


# ---------------------------------------------------------------------------
# I5 -- Hypothetischer Lastfluss ohne Seiteneffekt
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
# I6 -- Exogene Eingaenge tragen Schranken
# ---------------------------------------------------------------------------


def test_i6_exogenous_input_cannot_be_built_without_bounds() -> None:
    """I6: kein Datenpfad kann exogene Eingaenge ohne Schranken erzeugen.

    In M0 auf Schemaebene erfuellt: ``bounds_mw`` ist ein Pflichtfeld mit
    Formpruefung. Unsicherheitsmengen spaeter nachzuruesten waere ein Eingriff
    in jeden Datenpfad.
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
    """I6: jede Anlage traegt vollstaendige ``AssetRatings``.

    Grundlage der deterministischen Unsicherheitsmengen (Abschnitt 6.9, P2).
    Seit M1 erzeugt der SimBench-Adapter die Ratings; sie spaeter ueber alle
    Szenarien nachzutragen waere reine Handarbeit gewesen.
    """
    pytest.importorskip("simbench", reason="Extra 'sim' nicht installiert")
    from lvgrid_rl.data.sources.simbench import load_simbench  # noqa: PLC0415

    data = load_simbench("1-LV-rural1--2-sw")
    assert data.assets, "Netz ohne Anlagen"
    for asset in data.assets:
        r = asset.ratings
        assert r.p_min_mw <= r.p_max_mw
        # Elemente mit Scheinleistungsangabe brauchen sie fuer die spaetere
        # Q-Regelung und fuer das Apparatediagramm des Wechselrichters.
        if asset.element_table in ("sgen", "storage"):
            assert r.s_max_mva is not None and r.s_max_mva > 0.0


# ---------------------------------------------------------------------------
# I7 -- Zwei Sicherheits-Eingriffspunkte mit Null-Implementierung
# ---------------------------------------------------------------------------


def test_i7_null_safety_component_satisfies_the_protocol() -> None:
    """I7: der Eingriffspunkt existiert ab M0 mit Null-Implementierung."""
    assert isinstance(NullSafetyComponent(), SafetyComponent)


@pytest.mark.xfail(
    not _module_available("lvgrid_rl.env.lv_grid_env"),
    strict=True,
    reason="I7 fuer die Env wird in M3 faellig.",
)
def test_i7_env_exposes_both_intervention_points() -> None:
    """I7: beide Eingriffspunkte liegen im Ablauf der Environment.

    Der Shield sitzt innerhalb der Env, weil er den Zustand braucht; Masking
    liegt ausserhalb und braucht die Verteilung. Beide Pfade muessen vorhanden
    sein, sonst ist einer der Mechanismen spaeter nur mit Eingriff in den
    Ablauf nachruestbar.

    Pruefkriterien fuer M3:

    * eine Dummy-``SafetyComponent``, die Aktionen halbiert, wirkt messbar auf
      die geschriebenen Setpoints;
    * ``info["action_mask"]`` ist in jedem Schritt vorhanden und erlaubt bei
      ``safety.mechanism: none`` alle Aktionen.
    """
    from lvgrid_rl.env.lv_grid_env import LVGridEnv  # noqa: PLC0415

    raise AssertionError(f"Pruefung fuer {LVGridEnv} ausstehend")
