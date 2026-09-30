"""Training callbacks.

Two of them. :class:`TrainingProgress` is a progress display with a
remaining-time estimate: a training run takes hours, and without feedback there
is no way to tell a slow run from a hung one -- or to notice that the throughput
dropped because something else started competing for the machine.
:class:`LagrangianCallback` updates the Lagrange multipliers of the constrained
reward formulation (architecture section 6.4).

**Why not Stable-Baselines3's own progress bar.** ``model.learn(progress_bar=True)``
needs ``tqdm`` *and* ``rich``, and it renders as a live-updating widget that turns
into thousands of garbled lines when the output is redirected to a log file --
which is what one does with an overnight sweep. This implementation has no
dependencies beyond the standard library, adapts to whether it is writing to a
terminal, and reports the one quantity a bar cannot: the mean episode return, so
that a run which is progressing but not learning is visible as such.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from lvgrid_rl.env.reward import RewardConfig, RewardMode

__all__ = ["DualSpec", "LagrangianCallback", "TrainingProgress", "format_duration"]


def format_duration(seconds: float) -> str:
    """Format a duration as ``h:mm:ss``, or ``--:--:--`` if unknown.

    >>> format_duration(3725)
    '1:02:05'
    >>> format_duration(float("nan"))
    '--:--:--'
    """
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class TrainingProgress(BaseCallback):
    """Prints progress, throughput and an estimated time to completion.

    Args:
        total_timesteps: The training budget, for the percentage and estimate.
        interval_s: Minimum wall-clock seconds between updates. Throttled by
            time rather than by step count because the step rate is exactly what
            is unknown in advance.
        stream: Where to write. Defaults to stderr so that a redirected stdout
            keeps only the results table.
        force_plain: Write newline-terminated lines even on a terminal. Useful
            when the output is piped through something that mangles carriage
            returns.

    The estimate uses the throughput since training started rather than an
    instantaneous rate. Power flow times vary with the operating point -- a
    winter night solves faster than a summer noon -- so an instantaneous estimate
    would swing by a factor of two and be useless.
    """

    def __init__(
        self,
        total_timesteps: int,
        interval_s: float = 10.0,
        stream: TextIO | None = None,
        force_plain: bool = False,
    ) -> None:
        super().__init__()
        self.total_timesteps = total_timesteps
        self.interval_s = interval_s
        self.stream = stream if stream is not None else sys.stderr
        self._started = 0.0
        self._last_print = 0.0
        self._start_timesteps = 0
        self._interactive = (not force_plain) and bool(
            getattr(self.stream, "isatty", lambda: False)()
        )

    def _on_training_start(self) -> None:
        self._started = time.perf_counter()
        self._last_print = 0.0
        # A resumed run starts above zero; the estimate must count only the
        # steps of this session.
        self._start_timesteps = self.model.num_timesteps

    def _mean_episode_return(self) -> float:
        """Mean return over the recent episodes, or NaN if none finished yet.

        Requires the environment to be wrapped in a monitor; without one the
        buffer stays empty and the field reads ``n/a``, which is itself a useful
        signal that the wrapper is missing.
        """
        buffer = getattr(self.model, "ep_info_buffer", None)
        if not buffer:
            return float("nan")
        returns = [entry["r"] for entry in buffer if "r" in entry]
        return float(np.mean(returns)) if returns else float("nan")

    def _line(self) -> str:
        done = self.model.num_timesteps - self._start_timesteps
        total = max(self.total_timesteps - self._start_timesteps, 1)
        elapsed = time.perf_counter() - self._started
        fraction = min(done / total, 1.0)
        rate = done / elapsed if elapsed > 0 else float("nan")
        remaining = (total - done) / rate if rate > 0 else float("nan")
        reward = self._mean_episode_return()
        reward_text = f"{reward:9.1f}" if np.isfinite(reward) else "      n/a"
        return (
            f"[{fraction:6.1%}] {done:>8,}/{total:,} steps | "
            f"{rate:5.1f} steps/s | elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(remaining)} | ep_rew {reward_text}"
        )

    def _emit(self, final: bool = False) -> None:
        line = self._line()
        if self._interactive and not final:
            self.stream.write("\r" + line)
        else:
            self.stream.write(("\r" if self._interactive else "") + line + "\n")
        self.stream.flush()

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._started - self._last_print >= self.interval_s:
            self._last_print = now - self._started
            self._emit()
        return True

    def _on_training_end(self) -> None:
        self._emit(final=True)


@dataclass(frozen=True, slots=True)
class DualSpec:
    """Dual ascent settings for one constraint term.

    Args:
        learning_rate: Step size of the multiplier per rollout, in reward units
            per unit of cost excess. Well below the policy's own rate of change,
            because the dual problem oscillates easily (§6.4).
        lambda_max: Upper bound of the multiplier. A multiplier that ends at
            this bound means the constraint was never met, and the run has
            degenerated into a fixed penalty weight of ``lambda_max``; that is
            reported, not hidden.
        initial: Starting value.
    """

    learning_rate: float
    lambda_max: float
    initial: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.lambda_max <= 0:
            raise ValueError("learning_rate and lambda_max must be positive")
        if not 0.0 <= self.initial <= self.lambda_max:
            raise ValueError("initial must lie in [0, lambda_max]")


DEFAULT_DUALS: Mapping[str, DualSpec] = {
    # The step sizes are per rollout, and a rollout is long: train.py runs PPO
    # with the Stable-Baselines3 default of 2,048 steps per worker, 16,384 with
    # eight workers, so a 600,000-step run has only about 37 dual updates.
    #
    # Voltage costs are rates, bounded by 2 / 1.5. A multiplier of 1 prices a
    # violating window at the worst bus at two thirds of a megawatt-hour of
    # curtailment -- some fifty times the mean PV yield of a control step on the
    # test weeks.
    "en50160_k95": DualSpec(learning_rate=4.0, lambda_max=50.0),
    "en50160_k100": DualSpec(learning_rate=4.0, lambda_max=50.0),
    # An untrained policy produces a thermal cost of about 0.01 per control step
    # on the training episodes. The M4.0 validation ran with the M3 definition,
    # which was three times the integral, and with a step of 80 and a bound of
    # 200. Preserving that trajectory needs the multiplier three times larger
    # (the cost is a third) and so the step nine times larger: 720 and 600. The
    # multipliers of M4.0 read 34 to 44 in the old units, 101 to 131 in these.
    "thermal_overload": DualSpec(learning_rate=720.0, lambda_max=600.0),
}
"""Starting values for the validation on the M3 setup, not tuned values."""


class LagrangianCallback(BaseCallback):
    """Dual ascent on the constraint costs, once per rollout.

    ``lambda <- clip(lambda + eta * (J_c - d), 0, lambda_max)``, where ``J_c``
    is an exponentially smoothed estimate of the mean cost per control step and
    ``d`` the term's limit from the reward configuration. Because the voltage
    costs are rates (:meth:`RewardComposer.compute`), ``d = 0.05`` for K95 is the
    standard's criterion in the cost's own unit.

    The update runs at the end of a rollout, after collection and before the
    policy update. With an on-policy algorithm every reward in a rollout was
    therefore computed with one and the same multiplier, and the non-stationarity
    is confined to the boundary between rollouts. With an off-policy algorithm
    this is **not** sufficient -- the replay buffer would mix multipliers -- and
    the callback refuses to run there rather than produce a quietly biased
    result.

    With a limit of zero, as for the thermal term, ``J_c - d`` is never negative
    and the multiplier can only grow. It then settles at the smallest value that
    drives the cost to zero, or runs into ``lambda_max``. Both outcomes are
    informative and are visible in the history.

    Args:
        reward: The reward configuration of the environments. Must be in
            ``lagrangian`` mode; in ``fixed_weights`` mode the multipliers would
            be ignored and the run would silently be a different experiment.
        duals: Dual settings per constraint term; every constraint term of
            ``reward`` needs one.
        ema_decay: Weight of the previous estimate in the smoothing of ``J_c``.
        history_path: Where to write the multiplier history as JSON at the end
            of training, or ``None``.
    """

    def __init__(
        self,
        reward: RewardConfig,
        duals: Mapping[str, DualSpec] = DEFAULT_DUALS,
        ema_decay: float = 0.8,
        history_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        if reward.mode is not RewardMode.LAGRANGIAN:
            raise ValueError(
                f"reward mode is {reward.mode.value!r}; the multipliers would be "
                "ignored. Use RewardMode.LAGRANGIAN."
            )
        missing = set(reward.constraint) - set(duals)
        if missing:
            raise ValueError(f"No dual settings for constraint terms: {sorted(missing)}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        self.limits = {name: spec.limit for name, spec in reward.constraint.items()}
        self.duals = {name: duals[name] for name in self.limits}
        self.ema_decay = ema_decay
        self.history_path = Path(history_path) if history_path is not None else None
        self.multipliers = {name: d.initial for name, d in self.duals.items()}
        self.history: list[dict] = []
        self._estimate: dict[str, float] = {}
        self._reset_sums()

    def _reset_sums(self) -> None:
        self._sums = dict.fromkeys(self.limits, 0.0)
        self._counts = dict.fromkeys(self.limits, 0)

    def _push(self) -> None:
        self.training_env.env_method("set_multipliers", dict(self.multipliers))

    def _on_training_start(self) -> None:
        from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

        if isinstance(self.model, OffPolicyAlgorithm):
            raise TypeError(
                "LagrangianCallback supports on-policy algorithms only: a replay "
                "buffer holds rewards computed with stale multipliers. Store the "
                "costs separately and recompute the reward on sampling first."
            )
        self._push()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            for name in self.limits:
                value = info.get(f"cost/{name}")
                # A diverged power flow reports NaN costs; it carries its own
                # penalty and must not poison the estimate.
                if value is not None and np.isfinite(value):
                    self._sums[name] += float(value)
                    self._counts[name] += 1
        return True

    def _on_rollout_end(self) -> None:
        entry: dict = {"timesteps": int(self.num_timesteps)}
        for name, limit in self.limits.items():
            if self._counts[name] == 0:
                continue
            mean = self._sums[name] / self._counts[name]
            previous = self._estimate.get(name)
            estimate = (
                mean
                if previous is None
                else self.ema_decay * previous + (1.0 - self.ema_decay) * mean
            )
            self._estimate[name] = estimate
            dual = self.duals[name]
            updated = self.multipliers[name] + dual.learning_rate * (estimate - limit)
            self.multipliers[name] = float(np.clip(updated, 0.0, dual.lambda_max))
            entry[name] = {
                "cost_mean": mean,
                "cost_estimate": estimate,
                "limit": limit,
                "multiplier": self.multipliers[name],
            }
            self.logger.record(f"lagrangian/cost/{name}", mean)
            self.logger.record(f"lagrangian/multiplier/{name}", self.multipliers[name])
        self.history.append(entry)
        self._reset_sums()
        self._push()

    def _on_training_end(self) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(
            json.dumps(
                {
                    "ema_decay": self.ema_decay,
                    "duals": {name: asdict(d) for name, d in self.duals.items()},
                    "final_multipliers": self.multipliers,
                    "history": self.history,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
