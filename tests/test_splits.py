"""Tests of the week characterisation and the stratified split (M3, decision D13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lvgrid_rl.env.splits import (
    FEATURE_COLUMNS,
    STEPS_PER_WEEK_15MIN,
    SplitSpec,
    WeekSplit,
    build_split,
    weekly_features,
)

pytest.importorskip("simbench", reason="extra 'sim' not installed")

CODE = "1-LV-rural1--2-sw"
SET_NAMES = ("train", "val", "test", "stress", "holdout", "embargoed")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Week features of the working scenario, computed once."""
    from lvgrid_rl.data.sources.simbench import load_simbench

    return weekly_features(load_simbench(CODE))


@pytest.fixture(scope="module")
def split(features: pd.DataFrame) -> WeekSplit:
    """The default split."""
    return build_split(features)


# ---------------------------------------------------------------------------
# Week characterisation
# ---------------------------------------------------------------------------


def test_only_complete_weeks_are_characterised(features: pd.DataFrame) -> None:
    """A standard-conforming pass rate needs complete calendar weeks.

    The SimBench year starts with a 2015 week-53 stub and ends with a truncated
    week 52, leaving 51 usable weeks.
    """
    assert len(features) == 51
    assert (features["n_steps"] == STEPS_PER_WEEK_15MIN).all()


def test_all_documented_features_are_present(features: pd.DataFrame) -> None:
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert features[list(FEATURE_COLUMNS)].notna().all().all()


def test_energy_features_are_magnitudes(features: pd.DataFrame) -> None:
    """Energy features are reported as magnitudes.

    That keeps the stratification axes monotone and readable; only the flow
    features are signed.
    """
    for column in (
        "pv_energy_mwh",
        "heat_pump_energy_mwh",
        "ev_energy_mwh",
        "base_load_energy_mwh",
    ):
        assert (features[column] >= 0).all()


def test_seasonality_is_visible_in_the_features(features: pd.DataFrame) -> None:
    """Sanity check against a unit or sign error in the scaling.

    Summer weeks must carry more PV and less heat pump energy than winter weeks.
    """
    month = pd.DatetimeIndex(features["start_utc"]).month
    summer = features[(month >= 6) & (month <= 8)]
    winter = features[(month == 12) | (month <= 2)]
    assert summer["pv_energy_mwh"].mean() > 2 * winter["pv_energy_mwh"].mean()
    assert winter["heat_pump_energy_mwh"].mean() > summer["heat_pump_energy_mwh"].mean()


def test_stratification_axes_are_not_redundant(features: pd.DataFrame) -> None:
    """The reason the default axes are PV and EV rather than PV and heat pump.

    PV energy and heat pump energy are strongly anticorrelated -- both are
    seasonal proxies -- so stratifying on them leaves corners of the grid empty.
    EV energy is essentially uncorrelated and therefore adds a real second
    dimension.
    """
    corr = features[list(FEATURE_COLUMNS)].corr(method="spearman")
    assert corr.loc["pv_energy_mwh", "heat_pump_energy_mwh"] < -0.7
    assert abs(corr.loc["pv_energy_mwh", "ev_energy_mwh"]) < 0.3


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def test_sets_are_disjoint_and_cover_every_week(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    collected = [w for name in SET_NAMES for w in split.weeks_of(name)]
    assert len(collected) == len(set(collected)), "sets overlap"
    assert set(collected) == set(features.index)


def test_split_is_deterministic(features: pd.DataFrame) -> None:
    """The split must be identical across runs, agents and baselines."""
    a, b = build_split(features), build_split(features)
    assert a.test == b.test and a.train == b.train


def test_a_different_seed_gives_a_different_split(features: pd.DataFrame) -> None:
    other = build_split(features, SplitSpec(seed=99))
    assert other.test != build_split(features).test


def test_test_set_covers_every_occupied_stratum(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    """The M3 acceptance criterion.

    Without this check the defect that motivated the module -- a test set sitting
    entirely in the lowest PV decile -- would reappear under a different
    mechanism.
    """
    occupied = {v for v in split.strata.values() if v >= 0}
    assert len(occupied) == split.spec.n_bins**2 == 9
    assert split.strata_covered("test") == occupied


def test_test_set_spans_the_annual_pv_range(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    """The test set must span the annual PV range.

    The concrete failure of the chronological split was that all eight of its
    test weeks sat below the 30th PV percentile.
    """
    ranks = features["pv_energy_mwh"].rank(pct=True).loc[list(split.test)]
    assert ranks.min() < 0.10
    assert ranks.max() > 0.90


def test_holdout_month_is_withheld_entirely(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    month = pd.DatetimeIndex(features["start_utc"]).month
    expected = set(features.index[month == split.spec.holdout_month])
    assert set(split.holdout) == expected
    for name in ("train", "val", "test"):
        assert not (set(split.weeks_of(name)) & expected)


def test_holdout_can_be_disabled(features: pd.DataFrame) -> None:
    split = build_split(features, SplitSpec(holdout_month=None))
    assert split.holdout == ()


def test_stress_weeks_are_the_extremes_and_are_reserved(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    """Curated extremes belong in their own set.

    Blending them into the test aggregate makes the number unreadable: a pass
    rate of 0.85 could mean failure on the one extreme week or a little failure
    everywhere.
    """
    assert len(split.stress) == len(split.spec.stress_axes)
    pv_rank = features["pv_energy_mwh"].rank(pct=True)
    assert pv_rank.loc[list(split.stress)].max() == 1.0
    for name in ("train", "val", "test"):
        assert not (set(split.stress) & set(split.weeks_of(name)))


def test_embargo_removes_weeks_adjacent_to_a_test_week(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    order = list(features.index)
    test_positions = {order.index(w) for w in split.test}
    for week in split.train:
        distance = min(abs(order.index(week) - t) for t in test_positions)
        assert distance > split.spec.embargo_weeks, f"{week} neighbours a test week"
    for week in split.embargoed:
        distance = min(abs(order.index(week) - t) for t in test_positions)
        assert distance <= split.spec.embargo_weeks


def test_embargo_can_be_switched_off_to_measure_its_cost(
    features: pd.DataFrame,
) -> None:
    """The embargo must be switchable off.

    It costs roughly a third of the training weeks, so being able to quantify
    what it buys matters.
    """
    with_embargo = build_split(features)
    without = build_split(features, SplitSpec(embargo_weeks=0))
    assert without.embargoed == ()
    assert len(without.train) > len(with_embargo.train)
    assert without.test == with_embargo.test, "the draw must not change"


# ---------------------------------------------------------------------------
# Configuration and serialisation
# ---------------------------------------------------------------------------


def test_unknown_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown stratification axis"):
        SplitSpec(axes=("pv_energy_mwh", "not_a_feature"))


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitSpec(fractions={"train": 0.5, "val": 0.2, "test": 0.2})


def test_two_identical_axes_are_rejected() -> None:
    with pytest.raises(ValueError, match="two different features"):
        SplitSpec(axes=("pv_energy_mwh", "pv_energy_mwh"))


def test_overlapping_sets_are_rejected() -> None:
    """Structural guard: a construction bug must not produce a silent leak."""
    with pytest.raises(ValueError, match="overlap"):
        WeekSplit(
            train=((2016, 1),),
            val=(),
            test=((2016, 1),),
            stress=(),
            holdout=(),
            embargoed=(),
            spec=SplitSpec(),
        )


def test_round_trip_through_json(split: WeekSplit) -> None:
    """The split reads back exactly.

    It is committed to the configuration, so any drift would change results
    silently.
    """
    restored = WeekSplit.from_json(split.to_json())
    for name in SET_NAMES:
        assert restored.weeks_of(name) == split.weeks_of(name)
    assert restored.spec == split.spec
    assert restored.strata == split.strata


def test_written_split_is_readable(split: WeekSplit, tmp_path) -> None:
    path = split.write(tmp_path / "nested" / "split.json")
    assert path.exists()
    assert WeekSplit.from_json(path.read_text(encoding="utf-8")).test == split.test


def test_coverage_report_lists_every_set(
    split: WeekSplit, features: pd.DataFrame
) -> None:
    report = split.coverage_report(features)
    assert set(report.index) == set(SET_NAMES)
    assert report.loc["test", "n_weeks"] == len(split.test)
    assert np.isclose(report.loc["test", "pv_energy_mwh_q_max"], 1.0, atol=0.15)
