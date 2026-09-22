"""Training callbacks.

Currently one: a progress display with a remaining-time estimate. A training run
takes hours, and without feedback there is no way to tell a slow run from a hung
one -- or to notice that the throughput dropped because something else started
competing for the machine.

**Why not Stable-Baselines3's own progress bar.** ``model.learn(progress_bar=True)``
needs ``tqdm`` *and* ``rich``, and it renders as a live-updating widget that turns
into thousands of garbled lines when the output is redirected to a log file --
which is what one does with an overnight sweep. This implementation has no
dependencies beyond the standard library, adapts to whether it is writing to a
terminal, and reports the one quantity a bar cannot: the mean episode return, so
that a run which is progressing but not learning is visible as such.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

__all__ = ["TrainingProgress", "format_duration"]


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
