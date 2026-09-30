"""Action mode 2: one network per asset kind, an individual setpoint per asset.

Decision D14 (architecture section 13): the parameters are shared in the
**weights**, not in the action space. The action space is the flat one of mode 1
-- one power setpoint per asset, affine and invertible (invariant I4), so the
certified arm's ``store_executed`` coupling stays possible -- and the network
that produces it is built so that its size does not depend on the number of
assets.

**Structure.** The observation comes in the ``per_asset`` layout (section 6.2):
a global block, then one block per asset of a fixed width per kind.

* A global encoder embeds the global block; one block encoder per kind embeds
  each asset's block.
* The **context** is the global embedding together with the *mean* of the block
  embeddings of every kind. Mean pooling is what makes the context independent
  of how many assets a kind has.
* One actor head per kind maps the context and an asset's own embedding to that
  asset's action components. Every asset of a kind goes through the same head.
* The critic maps the context to a value.
* The exploration noise is shared per kind as well: one log standard deviation
  per kind and action component, not per asset.

Two properties follow and are tested: **equivariance** -- swapping two assets of
a kind in the observation swaps their actions and leaves everything else
unchanged -- and **size independence** -- the parameter set is the same for a
grid with three batteries and one with thirteen, so trained weights load into
either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from lvgrid_rl.env.obs import ObservationLayout

__all__ = ["SharedAssetExtractor", "SharedAssetPolicy"]


def _mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    """Linear layers with an activation after every layer but the last."""
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class SharedAssetExtractor(nn.Module):
    """Takes the place of SB3's ``MlpExtractor``.

    ``forward_actor`` returns the mean actions directly, in action order, so the
    policy's ``action_net`` is the identity; ``forward_critic`` returns a latent
    for SB3's ``value_net``.

    Args:
        layout: The per-asset observation layout with action ranges.
        hidden: Width of every hidden layer and embedding.
        activation: Activation function.
    """

    def __init__(
        self,
        layout: ObservationLayout,
        hidden: int = 64,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.kinds = layout.kinds
        if not self.kinds:
            raise ValueError("the layout has no asset blocks")
        self.global_dim = layout.global_dim
        self.hidden = hidden
        action_dim = max(b.action_stop for b in layout.blocks)
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = hidden

        self.global_encoder = _mlp([layout.global_dim, hidden, hidden], activation)
        self.block_encoders = nn.ModuleDict()
        self.actor_heads = nn.ModuleDict()
        self.action_widths: dict[str, int] = {}
        context_dim = hidden * (1 + len(self.kinds))
        for kind in self.kinds:
            blocks = layout.blocks_of(kind)
            obs_widths = {b.obs_stop - b.obs_start for b in blocks}
            act_widths = {b.action_stop - b.action_start for b in blocks}
            if len(obs_widths) != 1 or len(act_widths) != 1:
                raise ValueError(
                    f"assets of kind {kind!r} differ in block or action width; a "
                    "shared network needs them equal"
                )
            (obs_width,) = obs_widths
            (act_width,) = act_widths
            self.action_widths[kind] = act_width
            self.block_encoders[kind] = _mlp([obs_width, hidden, hidden], activation)
            self.actor_heads[kind] = _mlp(
                [context_dim + hidden, hidden, act_width], activation
            )
            # Index tensors, registered as buffers so they follow the device and
            # are not parameters: they describe this grid, not the policy.
            obs_index = th.tensor(
                [list(range(b.obs_start, b.obs_stop)) for b in blocks], dtype=th.long
            )
            act_index = th.tensor(
                [list(range(b.action_start, b.action_stop)) for b in blocks],
                dtype=th.long,
            )
            self.register_buffer(f"obs_index_{kind}", obs_index, persistent=False)
            self.register_buffer(f"act_index_{kind}", act_index, persistent=False)
        self.critic = _mlp([context_dim, hidden, hidden], activation)
        self.activation = activation()
        self.action_dim = action_dim

    def _index(self, prefix: str, kind: str) -> th.Tensor:
        return getattr(self, f"{prefix}_index_{kind}")

    def _encode(self, obs: th.Tensor) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        """Context of shape ``(batch, context)`` and embeddings per kind."""
        global_code = self.activation(self.global_encoder(obs[:, : self.global_dim]))
        codes: dict[str, th.Tensor] = {}
        pooled = [global_code]
        for kind in self.kinds:
            blocks = obs[:, self._index("obs", kind)]  # (batch, n_kind, width)
            code = self.activation(self.block_encoders[kind](blocks))
            codes[kind] = code
            pooled.append(code.mean(dim=1))
        return th.cat(pooled, dim=-1), codes

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        """Mean actions, one set per asset from its kind's head, in action order."""
        context, codes = self._encode(obs)
        out = obs.new_zeros(obs.shape[0], self.action_dim)
        for kind in self.kinds:
            code = codes[kind]
            expanded = context.unsqueeze(1).expand(-1, code.shape[1], -1)
            actions = self.actor_heads[kind](th.cat([expanded, code], dim=-1))
            index = self._index("act", kind)
            out[:, index.reshape(-1)] = actions.reshape(obs.shape[0], -1)
        return out

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        """Latent for SB3's value head, from the context alone."""
        context, _ = self._encode(obs)
        return self.activation(self.critic(context))

    def forward(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Actor and critic outputs, as SB3's ``MlpExtractor`` returns them."""
        return self.forward_actor(obs), self.forward_critic(obs)


class SharedAssetPolicy(ActorCriticPolicy):
    """PPO policy for action mode 2.

    Args:
        layout: The per-asset observation layout, from
            ``env.observation_layout``. Filled in by
            :func:`lvgrid_rl.agents.factory.make_agent`.
        hidden: Width of the hidden layers.
        All other arguments as for :class:`ActorCriticPolicy`.
    """

    def __init__(
        self,
        *args: Any,
        layout: ObservationLayout | None = None,
        hidden: int = 64,
        **kwargs: Any,
    ) -> None:
        if layout is None:
            raise ValueError(
                "SharedAssetPolicy needs the per-asset observation layout; build "
                "the environment with layout 'per_asset' and pass "
                "env.observation_layout as policy_kwargs['layout']"
            )
        self.layout = layout
        self.hidden = hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        if self.features_dim != max(b.obs_stop for b in self.layout.blocks):
            raise ValueError(
                f"observation has {self.features_dim} features, the layout "
                f"{max(b.obs_stop for b in self.layout.blocks)}"
            )
        self.mlp_extractor = SharedAssetExtractor(
            self.layout, hidden=self.hidden, activation=self.activation_fn
        )

    def _build(self, lr_schedule) -> None:
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise NotImplementedError("SharedAssetPolicy supports Gaussian actions only")
        super()._build(lr_schedule)
        extractor: SharedAssetExtractor = self.mlp_extractor  # type: ignore[assignment]
        # The mean actions come out of the heads already; a dense action_net on
        # top would mix all assets and undo the sharing.
        self.action_net = nn.Identity()
        # One log standard deviation per kind and component, not per asset.
        del self.log_std
        self.kind_log_std = nn.ParameterDict(
            {
                kind: nn.Parameter(th.full((width,), float(self.log_std_init)))
                for kind, width in extractor.action_widths.items()
            }
        )
        if self.ortho_init:
            # SB3 gave the whole extractor the hidden-layer gain. The last layer
            # of every actor head is the action output and gets SB3's small
            # output gain, so the initial policy is near the centre of the box.
            for head in extractor.actor_heads.values():
                last = head[-1]
                nn.init.orthogonal_(last.weight, gain=0.01)
                nn.init.zeros_(last.bias)
        # The parameters changed after SB3 built its optimizer.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _expanded_log_std(self) -> th.Tensor:
        extractor: SharedAssetExtractor = self.mlp_extractor  # type: ignore[assignment]
        out = th.zeros(extractor.action_dim, device=self.device)
        for kind in extractor.kinds:
            index = extractor._index("act", kind)  # noqa: SLF001
            out = out.index_put(
                (index.reshape(-1),),
                self.kind_log_std[kind].repeat(index.shape[0]),
            )
        return out

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self._expanded_log_std())

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(layout=self.layout, hidden=self.hidden)
        return data

    @staticmethod
    def parameter_count(policy: nn.Module) -> int:
        """Number of trainable parameters, for the size-independence check."""
        return int(np.sum([p.numel() for p in policy.parameters() if p.requires_grad]))
