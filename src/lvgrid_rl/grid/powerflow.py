"""Power flow engine with a side-effect free hypothetical call.

Implements :class:`lvgrid_rl.core.protocols.PowerFlowEngine`. The hypothetical
call is invariant I5 of the extensibility contract: backup trajectories of a
later predictive safety filter and the falsification search of the verification
step both need to evaluate candidate operating points without disturbing the
running simulation.

The implementation keeps a second, dedicated grid object for that purpose rather
than snapshotting and restoring the live one. That is both faster and safer:
there is no restore step that could be incomplete, and pandapower writes a
number of result tables whose full extent is easy to miss.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from lvgrid_rl.core.protocols import Setpoint
from lvgrid_rl.core.schemas import GridState
from lvgrid_rl.grid.loader import GridModel

if TYPE_CHECKING:  # pragma: no cover
    from pandapower.auxiliary import pandapowerNet

__all__ = ["PandapowerEngine"]

# Element tables whose active and reactive power the engine writes.
_WRITABLE = ("load", "sgen", "storage")


class PandapowerEngine:
    """Newton-Raphson power flow on a prepared :class:`GridModel`.

    Args:
        model: The prepared grid.
        warm_start: Use the previous solution as the initial guess. Roughly
            halves the iteration count in a quasi-static time series, where
            consecutive operating points are close together.
        numba: Enable pandapower's numba acceleration when available.

    Non-convergence is a defined event, not an exception: :meth:`run` returns a
    :class:`GridState` with ``converged=False``. The environment turns that into
    a reward penalty and an ``info`` flag. Raising here would make a single
    numerically awkward step abort a whole training run.
    """

    def __init__(
        self, model: GridModel, warm_start: bool = True, numba: bool = False
    ) -> None:
        self.model = model
        self.warm_start = warm_start
        self.numba = numba
        self._net: pandapowerNet = model.net
        self._shadow: pandapowerNet = copy.deepcopy(model.net)
        self._solved_once = False
        # Positional lookup per element table, resolved once.
        self._positions = {
            table: {
                f"{table}:{idx}": pos for pos, idx in enumerate(self._net[table].index)
            }
            for table in _WRITABLE
            if table in self._net
        }

    # -- internal helpers ---------------------------------------------------

    def _apply(self, net: pandapowerNet, setpoints: Mapping[str, Setpoint]) -> None:
        """Write setpoints into a grid, converting to pandapower sign conventions.

        Internally everything uses the consumer reference direction (see
        :data:`lvgrid_rl.core.units.SIGN_CONVENTION`). pandapower counts ``sgen``
        the other way round, so exactly one sign flip happens here -- and this is
        the only place in the codebase where it happens.
        """
        for asset_id, sp in setpoints.items():
            table, _, _ = asset_id.partition(":")
            pos = self._positions[table][asset_id]
            sign = -1.0 if table == "sgen" else 1.0
            net[table].iloc[pos, net[table].columns.get_loc("p_mw")] = sign * sp.p_mw
            if "q_mvar" in net[table].columns:
                net[table].iloc[pos, net[table].columns.get_loc("q_mvar")] = (
                    sign * sp.q_mvar
                )

    def _solve(self, net: pandapowerNet, t_index: int, warm: bool) -> GridState:
        import pandapower as pp

        kwargs = {"numba": self.numba}
        if warm:
            kwargs["init"] = "results"
        try:
            pp.runpp(net, **kwargs)
        except Exception:
            # Retry from a flat start before giving up: a warm start from a
            # distant previous operating point is the most common cause of a
            # failure that a flat start still solves.
            try:
                pp.runpp(net, numba=self.numba, init="flat")
            except Exception:
                return GridState(
                    t_index=t_index,
                    vm_pu=np.full(len(net.bus), np.nan),
                    line_loading_percent=np.full(len(net.line), np.nan),
                    trafo_loading_percent=np.full(len(net.trafo), np.nan),
                    p_slack_mw=float("nan"),
                    losses_mw=float("nan"),
                    converged=False,
                )
        losses = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
        return GridState(
            t_index=t_index,
            vm_pu=net.res_bus.vm_pu.to_numpy(copy=True),
            line_loading_percent=net.res_line.loading_percent.to_numpy(copy=True),
            trafo_loading_percent=net.res_trafo.loading_percent.to_numpy(copy=True),
            p_slack_mw=float(net.res_ext_grid.p_mw.sum()),
            losses_mw=losses,
            converged=True,
        )

    # -- PowerFlowEngine protocol ------------------------------------------

    def run(self, setpoints: Mapping[str, Setpoint], t_index: int) -> GridState:
        """Run the power flow on the live grid and advance its state."""
        self._apply(self._net, setpoints)
        state = self._solve(
            self._net, t_index, warm=self.warm_start and self._solved_once
        )
        if state.converged:
            self._solved_once = True
        return state

    def run_hypothetical(
        self, setpoints: Mapping[str, Setpoint], t_index: int
    ) -> GridState:
        """Run the power flow without altering the live grid (invariant I5).

        The shadow grid is synchronised from the live grid first, so that the
        hypothetical operating point differs only in the setpoints passed in.
        """
        for table in _WRITABLE:
            if table not in self._net or len(self._net[table]) == 0:
                continue
            for column in ("p_mw", "q_mvar", "in_service"):
                if column in self._net[table].columns:
                    self._shadow[table][column] = self._net[table][column].to_numpy(
                        copy=True
                    )
        self._apply(self._shadow, setpoints)
        return self._solve(self._shadow, t_index, warm=False)
