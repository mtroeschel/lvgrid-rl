"""Episode sampling on top of the committed week split.

Two requirements pull in opposite directions. The EN 50160 criterion refers to a
whole week, so a standard-conforming assessment needs complete calendar weeks
with a zero initial budget. Week-long episodes, on the other hand, are long and
expensive to train on.

The resolution is **budget randomisation**: training episodes are 2-7 days and
start with a randomly drawn amount of budget already consumed, so the agent sees
every budget regime without episodes having to span a week. Evaluation episodes
are complete weeks starting from zero, so the reported KPI remains
standard-conforming (§6.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from lvgrid_rl.env.splits import WeekSplit

__all__ = ["EpisodeMode", "EpisodeSpec", "Episode", "EpisodeSampler"]


class EpisodeMode(StrEnum):
    """How episodes are drawn from a set of weeks."""

    TRAIN = "train"
    """2-7 days, start drawn inside a week, randomised initial budget."""
    EVALUATE = "evaluate"
    """Complete weeks in fixed order, zero initial budget."""


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Episode configuration.

    Args:
        mode: Training or evaluation.
        length_days: Episode length in training mode.
        max_initial_budget_frac: Upper bound of the randomly drawn initial
            budget, as a fraction of the permitted windows. Drawing up to 1.2
            rather than 1.0 is deliberate: the agent must also learn to behave
            sensibly once the budget is already spent, and that regime occurs in
            operation.
        randomise_budget: Whether to draw an initial budget at all. Disabled for
            evaluation.
    """

    mode: EpisodeMode = EpisodeMode.TRAIN
    length_days: int = 3
    max_initial_budget_frac: float = 1.2
    randomise_budget: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.length_days <= 7:
            raise ValueError("length_days must be between 1 and 7")
        if self.max_initial_budget_frac < 0.0:
            raise ValueError("max_initial_budget_frac must not be negative")


@dataclass(frozen=True, slots=True)
class Episode:
    """One episode, resolved to simulation step indices.

    Args:
        week: The ISO week it was drawn from.
        start_step: First simulation step, as an index into the profile frame.
        n_steps: Number of simulation steps.
        initial_k95_violations: Pre-consumed budget per assessed bus.
        windows_elapsed: Assessment windows already elapsed in the week, kept
            consistent with the pre-consumed budget so that week progress and
            budget use do not contradict each other in the observation.
    """

    week: tuple[int, int]
    start_step: int
    n_steps: int
    initial_k95_violations: np.ndarray
    windows_elapsed: int


class EpisodeSampler:
    """Draws episodes from one set of the committed split.

    Args:
        split: The committed week split.
        set_name: Which set to draw from (``train``, ``val``, ``test``,
            ``stress``, ``holdout``).
        index: Time index of the profile frame, used to resolve weeks to step
            indices.
        steps_per_day: Simulation steps per day, from ``sim_dt``.
        n_buses: Number of assessed connection points.
        budget_windows: Permitted violating windows per bus and week.
        spec: Episode configuration.
    """

    def __init__(
        self,
        split: WeekSplit,
        set_name: str,
        index: pd.DatetimeIndex,
        steps_per_day: int,
        n_buses: int,
        budget_windows: int,
        spec: EpisodeSpec | None = None,
    ) -> None:
        self.split = split
        self.set_name = set_name
        self.spec = spec or EpisodeSpec()
        self.steps_per_day = steps_per_day
        self.n_buses = n_buses
        self.budget_windows = budget_windows

        iso = index.isocalendar()
        keys = pd.MultiIndex.from_arrays(
            [iso.year.to_numpy(), iso.week.to_numpy()], names=["y", "w"]
        )
        positions = pd.Series(np.arange(len(index)), index=keys)
        self._week_bounds: dict[tuple[int, int], tuple[int, int]] = {}
        for week in split.weeks_of(set_name):
            if week not in positions.index:
                continue
            block = positions.loc[week]
            self._week_bounds[week] = (int(block.iloc[0]), int(block.iloc[-1]) + 1)
        if not self._week_bounds:
            raise ValueError(
                f"Set {set_name!r} contains no week present in the time index"
            )
        self._weeks = list(self._week_bounds)
        self._cursor = 0

    @property
    def n_weeks(self) -> int:
        """Weeks available in this set."""
        return len(self._weeks)

    def sample(self, rng: np.random.Generator) -> Episode:
        """Draw the next episode.

        In evaluation mode the weeks are returned in a fixed order and the
        generator is unused, so that every agent and every baseline sees exactly
        the same episodes.
        """
        if self.spec.mode is EpisodeMode.EVALUATE:
            week = self._weeks[self._cursor % len(self._weeks)]
            self._cursor += 1
            start, end = self._week_bounds[week]
            return Episode(
                week=week,
                start_step=start,
                n_steps=end - start,
                initial_k95_violations=np.zeros(self.n_buses, dtype=np.int32),
                windows_elapsed=0,
            )

        week = self._weeks[int(rng.integers(len(self._weeks)))]
        start, end = self._week_bounds[week]
        n_steps = self.spec.length_days * self.steps_per_day
        latest = max(end - n_steps, start)
        start_step = int(rng.integers(start, latest + 1)) if latest > start else start

        if self.spec.randomise_budget:
            frac = float(rng.uniform(0.0, self.spec.max_initial_budget_frac))
            used = int(round(frac * self.budget_windows))
            violations = np.full(self.n_buses, used, dtype=np.int32)
            # The elapsed windows must be at least the consumed ones, otherwise
            # the observation would report a budget spent in windows that never
            # happened.
            windows_elapsed = int(rng.integers(used, max(used, 1008 - 1) + 1))
        else:
            violations = np.zeros(self.n_buses, dtype=np.int32)
            windows_elapsed = 0

        return Episode(
            week=week,
            start_step=start_step,
            n_steps=min(n_steps, end - start_step),
            initial_k95_violations=violations,
            windows_elapsed=windows_elapsed,
        )

    def reset_cursor(self) -> None:
        """Restart the evaluation order."""
        self._cursor = 0
