"""Tests of the data layer (M1).

Tests that need real SimBench data are skipped when the optional ``sim`` extra
is not installed. CI installs it, so that this layer is actually exercised
there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lvgrid_rl.data.cache import CacheKey, ProfileCache, frame_hash
from lvgrid_rl.data.resample import Policy, energy_error, resample_frame, resample_series
from lvgrid_rl.data.sources.simbench import (
    AssetCategory,
    categorize,
    ev_rated_power_kw,
    split_pq,
)
from lvgrid_rl.data.timebase import (
    ALLOWED_SIM_DT_MIN,
    PQ_WINDOW_MIN,
    TimeBase,
    to_utc_index,
)

simbench = pytest.importorskip("simbench", reason="extra 'sim' not installed")

SIMBENCH_CODE = "1-LV-rural1--2-sw"


@pytest.fixture(scope="module")
def simbench_data():
    """Prepared SimBench grid, loaded once per test module."""
    from lvgrid_rl.data.sources.simbench import load_simbench

    return load_simbench(SIMBENCH_CODE)


# ---------------------------------------------------------------------------
# Time base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ALLOWED_SIM_DT_MIN)
def test_allowed_step_sizes_divide_the_pq_window(dt: int) -> None:
    assert PQ_WINDOW_MIN % dt == 0
    assert TimeBase(dt, control_dt_min=60).sim_steps_per_pq_window == PQ_WINDOW_MIN // dt


def test_fifteen_minutes_is_rejected() -> None:
    """15 does not divide 10 -- the native SimBench resolution, of all things."""
    with pytest.raises(ValueError, match="EN 50160"):
        TimeBase(sim_dt_min=15, control_dt_min=15)


def test_control_step_must_be_a_multiple_of_the_simulation_step() -> None:
    with pytest.raises(ValueError, match="multiple"):
        TimeBase(sim_dt_min=5, control_dt_min=7)


def test_control_step_may_be_offset_against_the_pq_grid() -> None:
    """A 15-minute control cycle on 5-minute physics is valid and realistic."""
    tb = TimeBase(sim_dt_min=5, control_dt_min=15)
    assert tb.sim_steps_per_control_step == 3
    assert tb.control_steps_per_week == 672


def test_suggested_gamma_covers_the_weekly_horizon() -> None:
    """Gamma is derived from the criterion horizon, not a free parameter."""
    tb = TimeBase(sim_dt_min=5, control_dt_min=15)
    effective_horizon = 1.0 / (1.0 - tb.suggested_gamma())
    assert effective_horizon == pytest.approx(tb.control_steps_per_week)


def test_dst_transition_is_resolved_into_a_regular_utc_index() -> None:
    """The core of the time zone handling, on synthetic data.

    On 2016-10-30 the hour 02:00 repeats in German local time. Read naively the
    index is not unique; after conversion it must be strictly monotonic and
    gapless.
    """
    repeated = pd.date_range("2016-10-30 02:00", "2016-10-30 02:45", freq="15min")
    # Ordering as in the source file: first the block in summer time, then the
    # same block in standard time. ``ambiguous="infer"`` needs exactly this
    # contiguous sequence -- a naively sorted index that interleaves the two
    # blocks cannot be resolved.
    doubled = pd.DatetimeIndex(
        list(pd.date_range("2016-10-30 00:00", "2016-10-30 01:45", freq="15min"))
        + list(repeated)
        + list(repeated)
        + list(pd.date_range("2016-10-30 03:00", "2016-10-30 04:00", freq="15min"))
    )
    assert not doubled.is_unique

    utc = to_utc_index(doubled)
    assert utc.is_monotonic_increasing
    assert utc.is_unique
    assert set(pd.Series(utc).diff().dropna().unique()) == {pd.Timedelta("15min")}


def test_already_localized_index_is_only_converted() -> None:
    idx = pd.date_range("2016-06-01", periods=4, freq="15min", tz="Europe/Berlin")
    assert str(to_utc_index(idx).tz) == "UTC"


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _power_series(freq: str = "15min", n: int = 96) -> pd.Series:
    idx = pd.date_range("2016-06-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    return pd.Series(rng.random(n), index=idx, name="p")


def test_upsampling_power_conserves_energy_exactly() -> None:
    """Piecewise constant preserves the energy of the source interval."""
    s = _power_series()
    up = resample_series(s, pd.Timedelta("5min"), Policy.MEAN)
    assert len(up) == 3 * len(s)
    assert energy_error(s, up) == pytest.approx(0.0, abs=1e-12)


def test_downsampling_power_conserves_energy_exactly() -> None:
    s = _power_series(freq="5min", n=288)
    down = resample_series(s, pd.Timedelta("15min"), Policy.MEAN)
    assert len(down) == len(s) // 3
    assert energy_error(s, down) == pytest.approx(0.0, abs=1e-12)


def test_upsampling_covers_the_full_last_source_interval() -> None:
    """Otherwise a partial interval is missing at the end of the year."""
    s = _power_series(n=4)
    up = resample_series(s, pd.Timedelta("5min"), Policy.MEAN)
    assert up.index[-1] == s.index[-1] + pd.Timedelta("10min")


def test_linear_policy_interpolates_instead_of_stepping() -> None:
    idx = pd.date_range("2016-06-01", periods=3, freq="15min", tz="UTC")
    s = pd.Series([0.0, 30.0, 60.0], index=idx)
    up = resample_series(s, pd.Timedelta("5min"), Policy.LINEAR)
    assert up.iloc[1] == pytest.approx(10.0)


def test_majority_policy_keeps_availability_binary() -> None:
    idx = pd.date_range("2016-06-01", periods=6, freq="5min", tz="UTC")
    s = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], index=idx)
    down = resample_series(s, pd.Timedelta("15min"), Policy.MAJORITY)
    assert set(down.unique()) <= {0.0, 1.0}
    assert down.iloc[0] == 1.0  # 2 von 3
    assert down.iloc[1] == 0.0  # 1 von 3


def test_non_integer_step_ratio_is_rejected() -> None:
    """Otherwise the conversion would be an interpolation with a grid offset."""
    s = _power_series()
    with pytest.raises(ValueError, match="multiple"):
        resample_series(s, pd.Timedelta("7min"), Policy.MEAN)


def test_irregular_index_is_rejected_with_a_dst_hint() -> None:
    idx = pd.DatetimeIndex(["2016-06-01 00:00", "2016-06-01 00:15", "2016-06-01 01:00"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx.tz_localize("UTC"))
    with pytest.raises(ValueError, match="daylight saving"):
        resample_series(s, pd.Timedelta("5min"), Policy.MEAN)


# ---------------------------------------------------------------------------
# SimBench adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("Air_Semi-Parallel_2", AssetCategory.HEAT_PUMP),
        ("Soil_Alternative_2", AssetCategory.HEAT_PUMP),
        ("HLS_A_22.0", AssetCategory.EV_CHARGER),
        ("PV8", AssetCategory.PV),
        ("Storage_PV5_L1-A", AssetCategory.STORAGE),
        ("H0-A", AssetCategory.HOUSEHOLD),
        ("L2-A", AssetCategory.AGRICULTURE),
        ("", AssetCategory.OTHER),
    ],
)
def test_profile_names_map_to_categories(profile: str, expected: AssetCategory) -> None:
    assert categorize(profile) == expected


def test_ev_charger_rated_power_is_read_from_the_profile_name() -> None:
    assert ev_rated_power_kw("HLS_A_3.7") == 3.7
    assert ev_rated_power_kw("HLS_A_22.0") == 22.0
    assert ev_rated_power_kw("PV8") is None


def test_ev_chargers_are_not_controllable_yet() -> None:
    """Charge points are uncontrollable for now.

    No admissible shift can be derived from a fixed load curve; the session
    model arrives with emobpy in M5.
    """
    assert not AssetCategory.EV_CHARGER.is_controllable
    assert AssetCategory.HEAT_PUMP.is_controllable
    assert AssetCategory.PV.is_controllable
    assert AssetCategory.STORAGE.is_controllable


def test_simbench_profiles_form_a_regular_utc_year(simbench_data) -> None:
    idx = simbench_data.profiles.index
    assert idx.is_monotonic_increasing and idx.is_unique
    assert str(idx.tz) == "UTC"
    assert len(idx) == 366 * 96, "2016 is a leap year, 15-minute grid"
    assert set(pd.Series(idx).diff().dropna().unique()) == {pd.Timedelta("15min")}


def test_scenario_two_contains_heat_pumps_and_chargers(simbench_data) -> None:
    """Counter-check to the original claim that SimBench has no EV profiles."""
    cats = {a.category for a in simbench_data.assets}
    assert AssetCategory.HEAT_PUMP in cats
    assert AssetCategory.EV_CHARGER in cats
    assert AssetCategory.STORAGE in cats


def test_ratings_follow_the_consumer_sign_convention(simbench_data) -> None:
    """PV feeds in and cannot draw; storage can do both."""
    for asset in simbench_data.assets:
        r = asset.ratings
        if asset.element_table == "sgen":
            assert r.p_max_mw == 0.0 and r.p_min_mw <= 0.0
        elif asset.element_table == "storage":
            assert r.p_min_mw < 0.0 < r.p_max_mw
        else:
            assert r.p_min_mw == 0.0 and r.p_max_mw >= 0.0


def test_connection_points_include_generator_only_buses(simbench_data) -> None:
    """Decision D7: generators count, not only loads."""
    load_buses = {a.bus for a in simbench_data.assets if a.element_table == "load"}
    all_buses = set(simbench_data.connection_point_buses)
    assert load_buses <= all_buses
    gen_buses = {a.bus for a in simbench_data.assets if a.element_table == "sgen"}
    assert gen_buses <= all_buses


def test_every_asset_references_an_existing_profile(simbench_data) -> None:
    for asset in simbench_data.assets:
        if asset.profile_name:
            assert asset.profile_name in simbench_data.profiles.columns


def test_resampling_the_real_year_conserves_energy(simbench_data) -> None:
    tb = TimeBase(sim_dt_min=5, control_dt_min=15)
    resampled = resample_frame(simbench_data.profiles, tb, policies={})
    assert len(resampled) == 3 * len(simbench_data.profiles)
    for column in ("H0-A", "PV8", "Soil_Alternative_2", "HLS_A_22.0"):
        assert energy_error(
            simbench_data.profiles[column], resampled[column]
        ) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Cache and manifest
# ---------------------------------------------------------------------------


def _small_frame() -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=24, freq="5min", tz="UTC", name="t")
    return pd.DataFrame({"a": np.arange(24.0), "b": np.arange(24.0) * 2}, index=idx)


def test_cache_key_is_stable_and_sensitive() -> None:
    base = CacheKey("simbench", "1.6.1", "1-LV-rural1--2-sw", 5)
    assert base.digest() == CacheKey("simbench", "1.6.1", "1-LV-rural1--2-sw", 5).digest()
    assert base.digest() != CacheKey("simbench", "1.6.1", "1-LV-rural1--2-sw", 1).digest()
    assert base.digest() != CacheKey("simbench", "1.6.2", "1-LV-rural1--2-sw", 5).digest()


def test_pipeline_version_is_part_of_the_key() -> None:
    """Otherwise a corrected rule would have no effect on a stale cache."""
    a = CacheKey("s", "1", "d", 5, pipeline_version="1")
    b = CacheKey("s", "1", "d", 5, pipeline_version="2")
    assert a.digest() != b.digest()


def test_cache_round_trip_preserves_content(tmp_path) -> None:
    cache = ProfileCache(tmp_path)
    key = CacheKey("simbench", "1.6.1", "testnetz", 5)
    frame = _small_frame()
    assert not cache.has(key)
    manifest = cache.store(key, frame, notes="Test")
    assert cache.has(key)
    loaded, loaded_manifest = cache.load(key)
    # check_freq=False: Parquet does not store the index freq attribute.
    pd.testing.assert_frame_equal(frame, loaded, check_freq=False)
    assert loaded_manifest.content_hash == manifest.content_hash
    assert loaded_manifest.n_rows == 24


def test_tampered_cache_entry_is_detected(tmp_path) -> None:
    """The content hash is the basis of the reproducibility statement."""
    cache = ProfileCache(tmp_path)
    key = CacheKey("simbench", "1.6.1", "testnetz", 5)
    cache.store(key, _small_frame())
    tampered = _small_frame()
    tampered.iloc[0, 0] = 999.0
    tampered.to_parquet(cache.path_for(key), compression="zstd")
    with pytest.raises(ValueError, match="Content hash"):
        cache.load(key)


def test_frame_hash_ignores_storage_details_but_not_content() -> None:
    frame = _small_frame()
    assert frame_hash(frame) == frame_hash(frame.copy())
    other = frame.copy()
    other.iloc[3, 1] += 1e-9
    assert frame_hash(frame) != frame_hash(other)


def test_missing_cache_entry_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ProfileCache(tmp_path).load(CacheKey("s", "1", "d", 5))


# ---------------------------------------------------------------------------
# Reactive power profiles
# ---------------------------------------------------------------------------


def test_reactive_profiles_are_carried_separately(simbench_data) -> None:
    """Load profiles come as separate ``*_pload`` and ``*_qload`` series.

    They are kept as their own column group because they need their own
    resampling rule and because reference method B3 (Q(U) characteristic)
    depends on them.
    """
    assert not simbench_data.q_profiles.empty
    assert simbench_data.q_profiles.index.equals(simbench_data.profiles.index)
    # Every reactive series has an active counterpart, but not the other way
    # round: generators and storage carry no reactive profile in SimBench.
    assert set(simbench_data.q_profiles.columns) <= set(simbench_data.profiles.columns)


def test_reactive_profiles_may_be_negative(simbench_data) -> None:
    """Capacitive behaviour is in the data and must not be assumed away.

    A validation rule demanding non-negative reactive power would reject real
    households.
    """
    assert float(simbench_data.q_profiles["H0-A"].min()) < 0.0


def test_only_loads_carry_reactive_profiles(simbench_data) -> None:
    pv = next(a for a in simbench_data.assets if a.category == AssetCategory.PV)
    household = next(
        a for a in simbench_data.assets if a.category == AssetCategory.HOUSEHOLD
    )
    assert simbench_data.q_profile_for(pv) is None
    assert simbench_data.q_profile_for(household) is not None


def test_merge_and_split_round_trip(simbench_data) -> None:
    """The cache stores one frame, so its content hash covers both groups."""
    merged = simbench_data.merged()
    assert merged.shape[1] == (
        simbench_data.profiles.shape[1] + simbench_data.q_profiles.shape[1]
    )
    active, reactive = split_pq(merged)
    pd.testing.assert_frame_equal(active, simbench_data.profiles)
    pd.testing.assert_frame_equal(reactive, simbench_data.q_profiles)


def test_reactive_profiles_survive_resampling(simbench_data) -> None:
    tb = TimeBase(sim_dt_min=5, control_dt_min=15)
    resampled = resample_frame(simbench_data.merged(), tb, policies={})
    _, reactive = split_pq(resampled)
    assert reactive.shape[1] == simbench_data.q_profiles.shape[1]
    assert energy_error(
        simbench_data.q_profiles["H0-A"], reactive["H0-A"]
    ) == pytest.approx(0.0, abs=1e-9)


def test_pipeline_version_changed_with_reactive_power() -> None:
    """The cache key must change when the column set changes.

    Adding the reactive columns changes the content hash.
    """
    from lvgrid_rl.data.cache import PIPELINE_VERSION

    assert PIPELINE_VERSION == "2"
