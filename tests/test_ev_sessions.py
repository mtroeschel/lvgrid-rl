"""Home charging sessions from emobpy vehicle-years (D17)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from lvgrid_rl.data.sources.ev_sessions import (
    home_sessions,
    read_emobpy_run,
    scale_to_annual,
)

SIM = pd.date_range("2016-01-03 23:00", periods=12 * 24 * 3, freq="5min", tz="UTC")
"""Three days of simulation steps, from 2016-01-04 00:00 local."""


def _emobpy(stays, start="2016-01-04 00:00", days=3):
    """An emobpy-like frame at 15 minutes: away except for ``stays``.

    ``stays`` holds ``(first_slot, n_slots, kw_slots)``: the vehicle is home in
    ``[first, first + n)`` and draws ``kw`` for the first ``kw_slots`` slots.
    """
    n = days * 96
    times = pd.date_range(start, periods=n, freq="15min")
    state = np.array(["leisure"] * n, dtype=object)
    point = np.array(["none"] * n, dtype=object)
    grid = np.zeros(n)
    for first, length, kw, kw_slots in stays:
        state[first : first + length] = "home"
        point[first : first + length] = "home"
        grid[first : first + kw_slots] = kw
    return pd.DataFrame(
        {
            "datetime_local": times,
            "state": state,
            "charging_point": point,
            "charge_grid": grid,
        }
    )


def test_a_stay_at_home_is_one_session_with_its_grid_energy() -> None:
    frame = _emobpy([(72, 48, 11.0, 4)])  # home 18:00-06:00, 11 kW for an hour
    table = home_sessions(frame, SIM, p_max_mw=0.011)
    assert len(table) == 1
    assert table.energy_mwh[0] == pytest.approx(0.011)
    # 18:00 local is 17:00 UTC: 17 h after the start of the index at 5 min.
    assert table.arrival[0] == 18 * 12
    assert table.departure[0] - table.arrival[0] == 12 * 12


def test_stays_without_energy_are_dropped_and_counted() -> None:
    frame = _emobpy([(10, 8, 0.0, 0), (72, 48, 3.7, 8)])
    table = home_sessions(frame, SIM, p_max_mw=0.0037)
    assert len(table) == 1
    assert table.counts["no_energy"] == 1


def test_a_stay_cut_at_the_edge_of_the_year_keeps_its_share() -> None:
    """Home from before the simulation starts: the part inside counts.

    Home 22:00 to 02:00 local (21:00 to 01:00 UTC), drawing 3.7 kW throughout;
    the index starts at 23:00 UTC, so half the stay and half its energy lie
    inside.
    """
    frame = _emobpy([(8, 16, 3.7, 16)], start="2016-01-03 20:00", days=1)
    table = home_sessions(frame, SIM, p_max_mw=0.0037)
    assert table.counts["cut_at_year_edge"] == 1
    assert table.arrival[0] == 0
    assert table.departure[0] == 2 * 12
    assert table.energy_mwh[0] == pytest.approx(0.0037 * 4 / 2)


def test_scaling_reaches_the_target_and_reports_its_factor() -> None:
    frame = _emobpy([(72, 48, 11.0, 4), (168, 40, 11.0, 2)])
    table = scale_to_annual(home_sessions(frame, SIM, 0.011), 0.008, SIM)
    assert table.total_energy_mwh == pytest.approx(0.008)
    assert table.scale == pytest.approx(0.008 / 0.0165)


def test_scaling_up_past_what_fits_is_capped_and_counted() -> None:
    frame = _emobpy([(72, 4, 11.0, 4)])  # one hour at home, full power
    table = scale_to_annual(home_sessions(frame, SIM, 0.011), 0.05, SIM)
    assert table.counts["reduced_after_scaling"] == 1
    assert table.energy_mwh[0] == pytest.approx(0.011)


def test_every_session_fits_at_nominal_power() -> None:
    frame = _emobpy([(72, 48, 11.0, 30), (168, 20, 3.7, 20), (250, 30, 11.0, 5)])
    for target in (0.01, 0.2):
        table = scale_to_annual(home_sessions(frame, SIM, 0.011), target, SIM)
        hours = (table.departure - table.arrival) * 5 / 60
        assert np.all(table.energy_mwh <= hours * table.p_max_mw + 1e-12)


def test_the_spring_change_is_read_as_wall_clock_time() -> None:
    """Home from 01:00 to 04:00 local on 27 March 2016 is two hours in UTC."""
    index = pd.date_range("2016-03-26 23:00", periods=12 * 24, freq="5min", tz="UTC")
    frame = _emobpy([(4, 12, 3.7, 4)], start="2016-03-27 00:00", days=1)
    table = home_sessions(frame, index, p_max_mw=0.0037)
    assert (table.departure[0] - table.arrival[0]) * 5 == 120


def _write_run(tmp_path, frame, **entry):
    path = tmp_path / "load_15.parquet"
    frame.to_parquet(path, index=False)
    manifest = {
        "charge_points": [
            {
                "asset_id": "load:15",
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "nominal_power_kw": 11.0,
                "annual_energy_mwh": 1.36,
                "battery_capacity_kwh": 58.0,
                "driver": "fulltime",
                "vehicle": ["Volkswagen", "ID.3", 2020],
                "weeks": [],
                **entry,
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_a_run_is_read_and_verified(tmp_path) -> None:
    _write_run(tmp_path, _emobpy([(72, 48, 11.0, 4)]))
    vehicles = read_emobpy_run(tmp_path)
    assert vehicles["load:15"].driver == "fulltime"
    assert vehicles["load:15"].annual_energy_mwh == pytest.approx(1.36)


def test_a_run_is_verified_against_a_committed_manifest(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_run(run, _emobpy([(72, 48, 11.0, 4)]))
    committed = tmp_path / "committed.manifest.json"
    committed.write_text((run / "manifest.json").read_text(encoding="utf-8"))
    (run / "manifest.json").unlink()
    assert "load:15" in read_emobpy_run(run, manifest=committed)
    _write_run(run, _emobpy([(72, 48, 3.7, 4)]))  # other data, own manifest
    with pytest.raises(ValueError, match="SHA-256"):
        read_emobpy_run(run, manifest=committed)


def test_a_changed_file_fails_the_checksum(tmp_path) -> None:
    _write_run(tmp_path, _emobpy([(72, 48, 11.0, 4)]), sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        read_emobpy_run(tmp_path)


def test_a_missing_run_says_how_to_make_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate.py"):
        read_emobpy_run(tmp_path / "nothing")
