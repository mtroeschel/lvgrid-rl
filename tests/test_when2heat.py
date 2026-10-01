"""The when2heat adapter (M4, step 4.3a)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lvgrid_rl.data.sources.when2heat import (
    SINKS,
    cop_on_index,
    heat_source_of,
    read_cop,
    thermal_demand_mw,
)

RAW = Path("data/raw/when2heat/when2heat-2023-07-27.csv")


def _write(path: Path, index: pd.DatetimeIndex, value=lambda i: 3.0 + 0.01 * i) -> Path:
    """A file in when2heat's format: semicolons, decimal commas, DE and AT columns."""
    columns = {"utc_timestamp": index.strftime("%Y-%m-%dT%H:%M:%SZ")}
    columns["cet_cest_timestamp"] = index.tz_convert("Europe/Berlin").strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    for country in ("AT", "DE"):
        for source in ("ASHP", "GSHP", "WSHP"):
            for sink in SINKS:
                columns[f"{country}_COP_{source}_{sink}"] = [
                    f"{value(i):.2f}".replace(".", ",") for i in range(len(index))
                ]
    pd.DataFrame(columns).to_csv(path, sep=";", index=False)
    return path


def _hours(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="1h", tz="UTC")


def test_heat_sources_follow_the_simbench_profile_names() -> None:
    assert heat_source_of("Air_Parallel_2") == "ASHP"
    assert heat_source_of("Soil_Alternative_2") == "GSHP"
    with pytest.raises(ValueError, match="not a SimBench heat pump"):
        heat_source_of("H0-A")


def test_decimal_commas_are_read_as_numbers(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _hours("2016-01-01", "2016-01-02"))
    cop = read_cop(path, verify=False)
    assert cop.column("GSHP", "floor").iloc[1] == pytest.approx(3.01)
    assert set(cop.frame.columns) == {f"{s}_{k}" for s in ("ASHP", "GSHP") for k in SINKS}


def test_the_missing_autumn_hour_is_filled_and_reported(tmp_path) -> None:
    """01:00 UTC on the last Sunday of October -- missing in every year."""
    index = _hours("2016-10-29", "2016-10-31")
    gap = pd.Timestamp("2016-10-30 01:00", tz="UTC")
    path = _write(tmp_path / "w.csv", index.drop(gap), value=lambda i: 3.0 + 0.05 * i)
    cop = read_cop(path, verify=False)
    assert cop.filled_utc == (gap,)
    assert cop.frame.index.equals(index)
    before, after = (
        cop.frame.loc[gap - pd.Timedelta("1h")],
        cop.frame.loc[gap + pd.Timedelta("1h")],
    )
    assert cop.frame.loc[gap, "GSHP_floor"] == pytest.approx(
        (before["GSHP_floor"] + after["GSHP_floor"]) / 2
    )


def test_any_other_gap_is_refused(tmp_path) -> None:
    index = _hours("2016-06-01", "2016-06-03")
    path = _write(
        tmp_path / "w.csv", index.drop(pd.Timestamp("2016-06-02 12:00", tz="UTC"))
    )
    with pytest.raises(ValueError, match="gaps other than the autumn"):
        read_cop(path, verify=False)


def test_a_cop_outside_the_plausible_range_is_refused(tmp_path) -> None:
    path = _write(
        tmp_path / "w.csv", _hours("2016-01-01", "2016-01-02"), value=lambda i: 0.5
    )
    with pytest.raises(ValueError, match="outside"):
        read_cop(path, verify=False)


def test_a_different_file_fails_the_checksum(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _hours("2016-01-01", "2016-01-02"))
    with pytest.raises(ValueError, match="SHA-256"):
        read_cop(path)


def test_cop_is_interpolated_linearly_onto_the_simulation_step(tmp_path) -> None:
    path = _write(
        tmp_path / "w.csv",
        _hours("2016-01-01", "2016-01-02"),
        value=lambda i: 3.0 + 0.05 * i,
    )
    hourly = read_cop(path, verify=False).column("ASHP", "floor")
    index = pd.date_range("2016-01-01 00:00", "2016-01-01 02:00", freq="5min", tz="UTC")
    fine = cop_on_index(hourly, index)
    assert fine.iloc[0] == pytest.approx(3.0)
    assert fine.loc[pd.Timestamp("2016-01-01 00:30", tz="UTC")] == pytest.approx(3.025)
    assert fine.loc[pd.Timestamp("2016-01-01 01:00", tz="UTC")] == pytest.approx(3.05)


def test_extrapolating_an_efficiency_is_refused(tmp_path) -> None:
    path = _write(tmp_path / "w.csv", _hours("2016-01-01", "2016-01-02"))
    hourly = read_cop(path, verify=False).column("ASHP", "floor")
    index = pd.date_range("2015-12-31 23:00", periods=3, freq="5min", tz="UTC")
    with pytest.raises(ValueError, match="not covered"):
        cop_on_index(hourly, index)


def test_thermal_demand_is_electrical_times_cop() -> None:
    index = pd.date_range("2016-01-01", periods=4, freq="5min", tz="UTC")
    el = pd.Series([0.0, 0.002, 0.004, 0.001], index=index)
    cop = pd.Series([3.0, 3.5, 4.0, 4.5], index=index)
    assert np.allclose(thermal_demand_mw(el, cop), [0.0, 0.007, 0.016, 0.0045])
    with pytest.raises(ValueError, match="share one index"):
        thermal_demand_mw(el, cop.shift(1, freq="5min"))
    with pytest.raises(ValueError, match="must not feed in"):
        thermal_demand_mw(-el - 0.001, cop)


@pytest.mark.skipif(not RAW.exists(), reason="run scripts/fetch_when2heat.py first")
def test_the_real_dataset_covers_the_simbench_year_after_repair() -> None:
    """The published file: checksum, one filled hour per year, plausible COP."""
    cop = read_cop(RAW)
    assert len(cop.filled_utc) == 15  # 2008 to 2022, one autumn hour each
    assert all(ts.month == 10 and ts.hour == 1 for ts in cop.filled_utc)
    year = cop.frame.loc["2015-12-31 23:00":"2016-12-31 23:00"]
    assert len(year) == 366 * 24 + 1
    # Ground source is the more efficient one on average, floor the better sink.
    assert year["GSHP_floor"].mean() > year["ASHP_floor"].mean()
    assert year["ASHP_floor"].mean() > year["ASHP_radiator"].mean()
    simbench = pd.date_range(
        "2015-12-31 23:00", "2016-12-31 22:55", freq="5min", tz="UTC"
    )
    assert not cop_on_index(year["GSHP_floor"], simbench).isna().any()
