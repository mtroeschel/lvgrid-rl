"""Tests of the training callbacks: progress display and dual ascent."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("stable_baselines3", reason="extra 'rl' not installed")

from lvgrid_rl.env.reward import RewardConfig, RewardMode  # noqa: E402
from lvgrid_rl.experiment.callbacks import (  # noqa: E402
    DEFAULT_DUALS,
    DualSpec,
    LagrangianCallback,
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


# ---------------------------------------------------------------------------
# LagrangianCallback
# ---------------------------------------------------------------------------


class _FakeVecEnv:
    """Records what the callback pushes to the workers."""

    def __init__(self) -> None:
        self.pushed: list[dict] = []

    def env_method(self, name: str, values: dict) -> None:
        assert name == "set_multipliers"
        self.pushed.append(dict(values))


class _FakeLogger:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def record(self, key: str, value: float) -> None:
        self.records[key] = value


def _lagrangian(**kwargs) -> tuple[LagrangianCallback, _FakeVecEnv]:
    config = RewardConfig(mode=RewardMode.LAGRANGIAN)
    callback = LagrangianCallback(config, **kwargs)
    env = _FakeVecEnv()
    callback.model = SimpleNamespace(  # type: ignore[assignment]
        logger=_FakeLogger(), get_env=lambda: env, num_timesteps=0
    )
    return callback, env


def _rollout(callback: LagrangianCallback, costs: list[dict]) -> None:
    for step in costs:
        callback.locals = {"infos": [{f"cost/{k}": v for k, v in step.items()}]}
        callback._on_step()
    callback._on_rollout_end()


def test_multiplier_rises_above_the_limit_and_falls_below_it() -> None:
    """Dual ascent: ``lambda += eta * (J - d)``, clipped at zero."""
    duals = {
        "en50160_k95": DualSpec(learning_rate=10.0, lambda_max=50.0),
        "en50160_k100": DualSpec(learning_rate=1.0, lambda_max=50.0),
        "thermal_overload": DualSpec(learning_rate=1.0, lambda_max=50.0),
    }
    callback, _ = _lagrangian(duals=duals, ema_decay=0.0)
    _rollout(callback, [{"en50160_k95": 0.15}, {"en50160_k95": 0.25}])
    assert callback.multipliers["en50160_k95"] == pytest.approx(10.0 * (0.20 - 0.05))
    _rollout(callback, [{"en50160_k95": 0.0}])
    assert callback.multipliers["en50160_k95"] == pytest.approx(1.5 - 0.5)
    _rollout(callback, [{"en50160_k95": 0.0}] * 3)
    assert callback.multipliers["en50160_k95"] == pytest.approx(0.5)
    _rollout(callback, [{"en50160_k95": 0.0}])
    assert callback.multipliers["en50160_k95"] == pytest.approx(0.0, abs=1e-12)
    _rollout(callback, [{"en50160_k95": 0.0}])
    assert callback.multipliers["en50160_k95"] == 0.0  # clipped, not negative


def test_multiplier_is_bounded_and_the_bound_is_visible() -> None:
    """A multiplier at its bound is a fixed weight in disguise; it must show."""
    duals = dict(DEFAULT_DUALS)
    duals["thermal_overload"] = DualSpec(learning_rate=100.0, lambda_max=5.0)
    callback, _ = _lagrangian(duals=duals, ema_decay=0.0)
    _rollout(callback, [{"thermal_overload": 1.0}])
    assert callback.multipliers["thermal_overload"] == 5.0
    assert callback.history[-1]["thermal_overload"]["multiplier"] == 5.0


def test_estimate_is_smoothed_across_rollouts() -> None:
    duals = {
        name: DualSpec(learning_rate=1.0, lambda_max=100.0) for name in DEFAULT_DUALS
    }
    callback, _ = _lagrangian(duals=duals, ema_decay=0.5)
    _rollout(callback, [{"thermal_overload": 1.0}])
    _rollout(callback, [{"thermal_overload": 0.0}])
    assert callback.history[-1]["thermal_overload"]["cost_estimate"] == pytest.approx(0.5)
    assert callback.multipliers["thermal_overload"] == pytest.approx(1.5)


def test_diverged_steps_do_not_poison_the_estimate() -> None:
    """A diverged power flow reports NaN costs; they are skipped, not averaged."""
    callback, _ = _lagrangian(ema_decay=0.0)
    _rollout(
        callback,
        [{"thermal_overload": float("nan")}, {"thermal_overload": 0.01}],
    )
    assert callback.history[-1]["thermal_overload"]["cost_mean"] == pytest.approx(0.01)


def test_multipliers_reach_the_workers_after_every_rollout() -> None:
    callback, env = _lagrangian(ema_decay=0.0)
    _rollout(callback, [{"thermal_overload": 0.05}])
    assert env.pushed[-1] == callback.multipliers
    assert env.pushed[-1]["thermal_overload"] > 0.0


def test_fixed_weight_mode_is_refused() -> None:
    """In fixed_weights mode the multipliers would silently be ignored."""
    with pytest.raises(ValueError, match="LAGRANGIAN"):
        LagrangianCallback(RewardConfig())


def test_every_constraint_needs_dual_settings() -> None:
    config = RewardConfig(mode=RewardMode.LAGRANGIAN)
    with pytest.raises(ValueError, match="thermal_overload"):
        LagrangianCallback(config, duals={"en50160_k95": DEFAULT_DUALS["en50160_k95"]})


def test_off_policy_algorithms_are_refused() -> None:
    """A replay buffer would mix rewards computed with different multipliers."""
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

    callback, _ = _lagrangian()
    callback.model = OffPolicyAlgorithm.__new__(OffPolicyAlgorithm)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="on-policy"):
        callback._on_training_start()


def test_history_is_written_at_the_end(tmp_path) -> None:
    callback, _ = _lagrangian(ema_decay=0.0, history_path=tmp_path / "lagrangian.json")
    _rollout(callback, [{"thermal_overload": 0.05}])
    callback._on_training_end()
    payload = json.loads((tmp_path / "lagrangian.json").read_text(encoding="utf-8"))
    assert payload["final_multipliers"] == callback.multipliers
    assert len(payload["history"]) == 1
