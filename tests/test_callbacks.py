"""Tests of the training progress display."""

from __future__ import annotations

import io

import pytest

pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")

from lvgrid_rl.experiment.callbacks import (  # noqa: E402
    TrainingProgress,
    format_duration,
)


class _FakeModel:
    """Stand-in for an SB3 model; the callback must not need more than this."""

    def __init__(self, num_timesteps: int = 0, episodes=None) -> None:
        self.num_timesteps = num_timesteps
        self.ep_info_buffer = episodes if episodes is not None else []


def _callback(total: int, stream, model: _FakeModel) -> TrainingProgress:
    callback = TrainingProgress(total, interval_s=0.0, stream=stream)
    callback.model = model  # type: ignore[assignment]
    callback.on_training_start({}, {})
    return callback


def test_duration_formatting() -> None:
    assert format_duration(0) == "0:00:00"
    assert format_duration(3725) == "1:02:05"


def test_unknown_duration_is_not_printed_as_zero() -> None:
    """A run that has not produced a rate yet must not claim to be finished."""
    assert format_duration(float("nan")) == "--:--:--"
    assert format_duration(float("inf")) == "--:--:--"
    assert format_duration(-1) == "--:--:--"


def test_progress_reports_fraction_and_steps() -> None:
    stream = io.StringIO()
    model = _FakeModel()
    callback = _callback(1000, stream, model)
    model.num_timesteps = 250
    callback.on_step()
    output = stream.getvalue()
    assert "25.0%" in output
    assert "250/1,000 steps" in output


def test_resumed_runs_count_only_this_session() -> None:
    """A resumed run counts only the steps of this session.

    Otherwise it would start at an arbitrary percentage and its estimate would
    be wrong throughout.
    """
    stream = io.StringIO()
    model = _FakeModel(num_timesteps=400)
    callback = _callback(1000, stream, model)
    model.num_timesteps = 700
    callback.on_step()
    # 300 of the remaining 600 steps are done, not 700 of 1000.
    assert "50.0%" in stream.getvalue()


def test_missing_monitor_shows_as_not_available() -> None:
    """A missing monitor wrapper shows as not available.

    An empty episode buffer means the wrapper is missing, and that should be
    visible rather than shown as a zero return.
    """
    stream = io.StringIO()
    model = _FakeModel()
    callback = _callback(100, stream, model)
    model.num_timesteps = 10
    callback.on_step()
    assert "n/a" in stream.getvalue()


def test_episode_return_is_averaged_over_the_buffer() -> None:
    stream = io.StringIO()
    model = _FakeModel(episodes=[{"r": -100.0}, {"r": -200.0}])
    callback = _callback(100, stream, model)
    model.num_timesteps = 10
    callback.on_step()
    assert "-150.0" in stream.getvalue()


def test_final_line_ends_with_a_newline() -> None:
    """The final line ends with a newline.

    A log file must not end mid-line, and the results table that follows must
    start on its own line.
    """
    stream = io.StringIO()
    model = _FakeModel()
    callback = _callback(100, stream, model)
    model.num_timesteps = 100
    callback.on_training_end()
    assert stream.getvalue().endswith("\n")
    assert "100.0%" in stream.getvalue()


def test_updates_are_throttled_by_wall_time() -> None:
    """Updates are throttled by wall time.

    By time rather than step count, because the step rate is exactly what is
    unknown in advance.
    """
    stream = io.StringIO()
    model = _FakeModel()
    callback = TrainingProgress(1000, interval_s=3600.0, stream=stream)
    callback.model = model  # type: ignore[assignment]
    callback.on_training_start({}, {})
    for step in range(1, 20):
        model.num_timesteps = step
        callback.on_step()
    assert stream.getvalue() == "", "no update should fit into the interval"


def test_callback_never_stops_training() -> None:
    """A display must not be able to abort a four-hour run."""
    stream = io.StringIO()
    model = _FakeModel()
    callback = _callback(100, stream, model)
    model.num_timesteps = 50
    assert callback.on_step() is True
