"""Adapter making a trained policy usable by the evaluation runner.

The runner knows only :class:`~lvgrid_rl.core.protocols.Controller`, which is
what makes the comparison against reference methods structurally fair. A policy,
however, consumes the observation vector rather than the information set, so it
needs the one piece the rule-based methods do not: what the environment last
observed.

That is handled by an optional ``set_observation`` hook which the runner calls
when a controller offers it. The alternative -- widening ``Controller.act`` to
take the observation as well -- would push a policy-specific concern into every
baseline signature.
"""

from __future__ import annotations

import numpy as np

from lvgrid_rl.core.information import InformationSet

__all__ = ["PolicyController"]


class PolicyController:
    """Wraps a Stable-Baselines3 model as a controller.

    Args:
        model: A trained SB3 model.
        deterministic: Whether to use the deterministic action. Evaluation runs
            deterministically; leaving it stochastic would mix policy quality
            with sampling noise in the reported KPIs.
        name: Label for the results table.
    """

    def __init__(self, model, deterministic: bool = True, name: str = "policy") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name
        self._observation: np.ndarray | None = None

    def set_observation(self, observation: np.ndarray) -> None:
        """Receive the environment's current observation."""
        self._observation = observation

    def reset(self, info: InformationSet) -> None:
        """Nothing to reset; the observation arrives through the hook."""

    def act(self, info: InformationSet) -> np.ndarray:
        """Predict an action from the last observation."""
        if self._observation is None:
            raise RuntimeError(
                "No observation received. The runner must call set_observation "
                "before act for policy controllers."
            )
        action, _ = self.model.predict(
            self._observation, deterministic=self.deterministic
        )
        return np.asarray(action, dtype=np.float64)
