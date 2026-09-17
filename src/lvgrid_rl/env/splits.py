"""Week characterisation and the stratified evaluation split (decision D13).

The split unit is the week, because that is the assessment interval of EN 50160
and because a standard-conforming pass rate needs complete calendar weeks with a
zero initial budget.

**Why not a chronological block split.** The obvious design -- train Jan-Sep,
validate Oct, test Nov-Dec -- puts the whole test set in one corner of the
operating envelope. Measured on ``1-LV-rural1--2-sw`` over the 51 complete weeks
of 2016, the annual PV quantiles of the eight Nov-Dec weeks are 0.29, 0.12, 0.16,
0.18, 0.06, 0.14, 0.10 and 0.00: every one of them below the 30th percentile, one
of them the weakest PV week of the year. Since the voltage band violations under
``moderate_growth`` are PV-driven overvoltage, such a test set measures the
headline KPI exactly where the problem does not occur.

The usual justification for a chronological split is leakage avoidance plus
deployment fidelity. This is not a forecasting task -- the policy is a controller,
and the question is whether it works in situations it has not seen. With a single
year of data a chronological split unavoidably confounds "unseen data" with
"unseen season". The leakage concern is real at the adjacency level, though, and
is addressed here by an embargo rather than by a block cut.

See section 6.5 of the architecture document.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from lvgrid_rl.data.sources.simbench import SimBenchData

__all__ = [
    "STEPS_PER_WEEK_15MIN",
    "FEATURE_COLUMNS",
    "weekly_features",
    "SplitSpec",
    "WeekSplit",
    "build_split",
]

STEPS_PER_WEEK_15MIN = 7 * 24 * 4
"""672 quarter-hour steps in a complete week, used to reject partial weeks."""

FEATURE_COLUMNS = (
    "pv_energy_mwh",
    "heat_pump_energy_mwh",
    "ev_energy_mwh",
    "base_load_energy_mwh",
    "peak_reverse_flow_mw",
    "peak_net_load_mw",
)
"""Characterisation features. All derived from **exogenous quantities only**.

No controller, no power flow, no KPI enters here. That is deliberate: a split
stratified on an outcome would leak the label into the assignment, and the test
set would no longer be an independent sample.
"""


def weekly_features(data: SimBenchData) -> pd.DataFrame:
    """Characterise every complete calendar week of a scenario.

    Args:
        data: Output of the SimBench adapter. The normalised profiles are scaled
            with each asset's nominal power, so the features describe the
            scenario as it will actually be simulated.

    Returns:
        One row per complete week, indexed by ``(iso_year, iso_week)``, with the
        columns of :data:`FEATURE_COLUMNS` plus ``start_utc`` and ``n_steps``.
        Partial weeks at the start and end of the series are dropped -- in the
        SimBench year those are the 2015 week 53 stub and the truncated week 52,
        leaving 51 usable weeks.

    Signs follow the consumer convention internally, but the energy features are
    reported as magnitudes, which keeps the stratification axes monotone and easy
    to read. Only the flow features are signed.
    """
    from lvgrid_rl.data.sources.simbench import AssetCategory

    profiles = data.profiles
    dt_hours = 0.25

    by_category: dict[AssetCategory, np.ndarray] = {}
    for asset in data.assets:
        if not asset.profile_name or asset.profile_name not in profiles.columns:
            continue
        series = profiles[asset.profile_name].to_numpy() * asset.p_nom_mw
        acc = by_category.setdefault(asset.category, np.zeros(len(profiles)))
        acc += series

    def energy(category: AssetCategory) -> np.ndarray:
        return by_category.get(category, np.zeros(len(profiles)))

    pv = energy(AssetCategory.PV)
    heat = energy(AssetCategory.HEAT_PUMP)
    ev = energy(AssetCategory.EV_CHARGER)
    base = (
        energy(AssetCategory.HOUSEHOLD)
        + energy(AssetCategory.AGRICULTURE)
        + energy(AssetCategory.COMMERCIAL)
    )
    # Signed net exchange at the connection points, consumer convention:
    # positive means the grid supplies, negative means reverse flow.
    net = base + heat + ev - pv

    frame = pd.DataFrame(
        {
            "pv_mw": pv,
            "heat_mw": heat,
            "ev_mw": ev,
            "base_mw": base,
            "net_mw": net,
        },
        index=profiles.index,
    )
    # Group by ISO week. The keys are passed as an array rather than a list of
    # tuples: pandas would interpret a list of tuples as column labels.
    iso = profiles.index.isocalendar()
    keys = pd.MultiIndex.from_arrays(
        [iso.year.to_numpy(), iso.week.to_numpy()], names=["iso_year", "iso_week"]
    )
    grouped = frame.groupby(keys, sort=True)

    rows = []
    for (iso_year, iso_week), block in grouped:
        if len(block) != STEPS_PER_WEEK_15MIN:
            continue
        rows.append(
            {
                "iso_year": int(iso_year),
                "iso_week": int(iso_week),
                "start_utc": block.index[0],
                "n_steps": len(block),
                "pv_energy_mwh": float(block["pv_mw"].sum() * dt_hours),
                "heat_pump_energy_mwh": float(block["heat_mw"].sum() * dt_hours),
                "ev_energy_mwh": float(block["ev_mw"].sum() * dt_hours),
                "base_load_energy_mwh": float(block["base_mw"].sum() * dt_hours),
                "peak_reverse_flow_mw": float(max(-block["net_mw"].min(), 0.0)),
                "peak_net_load_mw": float(max(block["net_mw"].max(), 0.0)),
            }
        )

    out = pd.DataFrame(rows).set_index(["iso_year", "iso_week"])
    return out.sort_index()


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Configuration of the stratified split.

    Args:
        axes: The two stratification axes. Two axes with three bins give nine
            strata; three axes would give 27, which is too fine for 51 weeks.

            The default pairs PV energy with EV energy, which is not the obvious
            choice and was picked after measuring the rank correlations on
            ``1-LV-rural1--2-sw``: PV energy and heat pump energy are Spearman
            **-0.81** correlated, and peak reverse flow is **+0.89** correlated
            with PV energy. All three are seasonal proxies, so stratifying on two
            of them leaves the high-PV/high-heat and low-PV/low-heat corners
            empty -- seven of nine strata occupied, and the second axis adds
            almost nothing. EV energy is the only feature essentially
            uncorrelated with the rest (-0.05 against PV, 0.07 against heat
            pump), so pairing it with PV fills all nine strata.

            PV is kept as the first axis because the voltage problem in this
            scenario is PV-driven overvoltage. For an undervoltage-dominated
            scenario the first axis should be the heat pump energy instead
            (architecture document, §13.2, open question 8).
        n_bins: Quantile bins per axis.
        fractions: Target shares for train, validation and test among the weeks
            that remain after the stress weeks and the held-out month are
            removed.
        embargo_weeks: Training weeks within this distance of a test week are
            dropped. Weather autocorrelation operates on the scale of days, so
            adjacent weeks are genuinely similar.

            This is not free. Measured on the 51 SimBench weeks with nine test
            weeks: an embargo of one week costs nine training weeks (26 down to
            17), an embargo of two costs fifteen. The default of one is a
            deliberate trade of roughly a third of the training weeks against an
            adjacency-inflated test result; the episode sampler can still start
            anywhere inside a training week, so the number of distinct episodes
            is far larger than the week count suggests. Set to 0 to measure how
            much the embargo actually changes the reported KPI.
        holdout_month: Calendar month withheld entirely, as a harder test of
            temporal generalisation. Season enters the observation through the
            sin/cos features, so under an interleaved split the agent has seen
            seasonally similar weeks for every test week. ``None`` disables it.
        n_stress_weeks: How many extreme weeks to reserve per stress axis.
        stress_axes: Axes whose maxima define the stress weeks.
        seed: Draw seed. Separate from the training and scenario seeds so the
            split stays identical across runs, agents and baselines.
    """

    axes: tuple[str, str] = ("pv_energy_mwh", "ev_energy_mwh")
    n_bins: int = 3
    fractions: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.60, "val": 0.15, "test": 0.25}
    )
    embargo_weeks: int = 1
    holdout_month: int | None = 2
    n_stress_weeks: int = 1
    stress_axes: tuple[str, ...] = (
        "pv_energy_mwh",
        "heat_pump_energy_mwh",
        "peak_reverse_flow_mw",
    )
    seed: int = 20240101

    def __post_init__(self) -> None:
        for axis in (*self.axes, *self.stress_axes):
            if axis not in FEATURE_COLUMNS:
                raise ValueError(
                    f"Unknown stratification axis {axis!r}; known: {FEATURE_COLUMNS}"
                )
        if len(set(self.axes)) != 2:
            raise ValueError("axes must name two different features")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        total = sum(self.fractions.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"fractions must sum to 1.0, got {total}")
        if self.holdout_month is not None and not 1 <= self.holdout_month <= 12:
            raise ValueError("holdout_month must be a calendar month or None")


@dataclass(frozen=True, slots=True)
class WeekSplit:
    """The resulting assignment of weeks to evaluation sets.

    Weeks are identified by ``(iso_year, iso_week)`` pairs. The five sets are
    disjoint; ``embargoed`` holds weeks removed from training because they
    neighbour a test week and belong to no set.

    The three reportable sets are kept apart on purpose and are never aggregated
    into one number. A blended pass rate of 0.85 cannot be read: it could mean
    failure on the single extreme week or a little failure everywhere.
    """

    train: tuple[tuple[int, int], ...]
    val: tuple[tuple[int, int], ...]
    test: tuple[tuple[int, int], ...]
    stress: tuple[tuple[int, int], ...]
    holdout: tuple[tuple[int, int], ...]
    embargoed: tuple[tuple[int, int], ...]
    spec: SplitSpec
    strata: Mapping[str, int] = field(default_factory=dict)
    """Stratum label per week, as ``"year-week" -> stratum index``."""

    def __post_init__(self) -> None:
        sets = {
            "train": set(self.train),
            "val": set(self.val),
            "test": set(self.test),
            "stress": set(self.stress),
            "holdout": set(self.holdout),
            "embargoed": set(self.embargoed),
        }
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = sets[a] & sets[b]
                if overlap:
                    raise ValueError(f"Sets {a} and {b} overlap: {sorted(overlap)}")

    def weeks_of(self, name: str) -> tuple[tuple[int, int], ...]:
        """Weeks of one set by name."""
        return getattr(self, name)

    def strata_covered(self, name: str) -> set[int]:
        """Stratum indices represented in one set."""
        return {self.strata[f"{y}-{w}"] for y, w in self.weeks_of(name)}

    def to_json(self) -> str:
        """Serialise for committing to the configuration."""
        payload = {
            "spec": asdict(self.spec),
            "sets": {
                name: [list(week) for week in self.weeks_of(name)]
                for name in ("train", "val", "test", "stress", "holdout", "embargoed")
            },
            "strata": dict(self.strata),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> WeekSplit:
        """Read back a committed split."""
        payload = json.loads(text)
        spec_raw = dict(payload["spec"])
        spec_raw["axes"] = tuple(spec_raw["axes"])
        spec_raw["stress_axes"] = tuple(spec_raw["stress_axes"])
        sets = {
            name: tuple(tuple(week) for week in weeks)
            for name, weeks in payload["sets"].items()
        }
        return cls(spec=SplitSpec(**spec_raw), strata=payload["strata"], **sets)

    def write(self, path: Path) -> Path:
        """Write the split to disk and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def coverage_report(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per set: size, strata covered, and the span of each axis in quantiles.

        This is the acceptance check for M3: the test set must cover all strata
        and its PV quantiles must span the year. Without it the defect that
        motivated this module -- a test set sitting entirely in the lowest PV
        decile -- would simply reappear under a different mechanism.
        """
        rows = []
        for name in ("train", "val", "test", "stress", "holdout", "embargoed"):
            weeks = self.weeks_of(name)
            if not weeks:
                rows.append({"set": name, "n_weeks": 0})
                continue
            row: dict[str, object] = {
                "set": name,
                "n_weeks": len(weeks),
                "n_strata": len(self.strata_covered(name)),
            }
            for axis in self.spec.axes:
                ranks = features[axis].rank(pct=True)
                row[f"{axis}_q_min"] = float(ranks.loc[list(weeks)].min())
                row[f"{axis}_q_max"] = float(ranks.loc[list(weeks)].max())
            rows.append(row)
        return pd.DataFrame(rows).set_index("set")


def _week_ordinal(week: tuple[int, int], order: Sequence[tuple[int, int]]) -> int:
    return order.index(week)


def build_split(features: pd.DataFrame, spec: SplitSpec | None = None) -> WeekSplit:
    """Build the stratified split from week features.

    The steps, in order: withhold the contiguous month; reserve the stress weeks;
    bin the remainder into strata; draw from each stratum proportionally; then
    apply the embargo. The order matters -- reserving stress weeks after
    stratification would distort the strata, and embargoing before the draw would
    have nothing to embargo against.

    Args:
        features: Output of :func:`weekly_features`.
        spec: Configuration; defaults to :class:`SplitSpec`.

    Returns:
        The assignment, with a stratum label per week.
    """
    spec = spec or SplitSpec()
    order = list(features.index)
    rng = np.random.default_rng(spec.seed)

    holdout: list[tuple[int, int]] = []
    if spec.holdout_month is not None:
        months = pd.DatetimeIndex(features["start_utc"]).month
        holdout = [
            w for w, m in zip(order, months, strict=True) if m == spec.holdout_month
        ]

    remaining = [w for w in order if w not in set(holdout)]

    stress: list[tuple[int, int]] = []
    for axis in spec.stress_axes:
        ranked = features.loc[remaining, axis].sort_values(ascending=False)
        taken = 0
        for week in ranked.index:
            if week in stress:
                continue
            stress.append(week)
            taken += 1
            if taken == spec.n_stress_weeks:
                break
    remaining = [w for w in remaining if w not in set(stress)]

    # Stratify on the remaining weeks only, so the extremes do not distort the
    # quantile boundaries.
    labels = {}
    binned = []
    for axis in spec.axes:
        values = features.loc[remaining, axis]
        binned.append(
            pd.qcut(values.rank(method="first"), spec.n_bins, labels=False).to_numpy()
        )
    for i, week in enumerate(remaining):
        labels[f"{week[0]}-{week[1]}"] = int(binned[0][i] * spec.n_bins + binned[1][i])
    for week in (*holdout, *stress):
        labels.setdefault(f"{week[0]}-{week[1]}", -1)

    assignment: dict[str, list[tuple[int, int]]] = {"train": [], "val": [], "test": []}
    for stratum in sorted({labels[f"{w[0]}-{w[1]}"] for w in remaining}):
        members = [w for w in remaining if labels[f"{w[0]}-{w[1]}"] == stratum]
        rng.shuffle(members)
        n = len(members)
        # Largest-remainder apportionment keeps the target shares even for
        # strata with two or three members, where naive rounding would starve
        # the test set.
        exact = {k: n * v for k, v in spec.fractions.items()}
        counts = {k: int(np.floor(v)) for k, v in exact.items()}
        for key in sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)[
            : n - sum(counts.values())
        ]:
            counts[key] += 1
        cursor = 0
        for key in ("train", "val", "test"):
            assignment[key].extend(members[cursor : cursor + counts[key]])
            cursor += counts[key]

    test_ordinals = {_week_ordinal(w, order) for w in assignment["test"]}
    embargoed = [
        w
        for w in assignment["train"]
        if any(
            abs(_week_ordinal(w, order) - t) <= spec.embargo_weeks for t in test_ordinals
        )
    ]
    train = [w for w in assignment["train"] if w not in set(embargoed)]

    def ordered(weeks: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(weeks, key=lambda w: _week_ordinal(w, order)))

    return WeekSplit(
        train=ordered(train),
        val=ordered(assignment["val"]),
        test=ordered(assignment["test"]),
        stress=ordered(stress),
        holdout=ordered(holdout),
        embargoed=ordered(embargoed),
        spec=spec,
        strata=labels,
    )
