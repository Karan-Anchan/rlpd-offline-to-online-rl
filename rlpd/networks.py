"""Actor and critic networks.   OWNER: Member 1 (algorithm).

Skeleton only — signatures are agreed, bodies are Member 1's to implement.
RLPD critic specifics: LayerNorm on, ensemble of `ensemble_size` Q-heads, MLP
of `num_layers` x `hidden_width`. Kept here so the package imports cleanly and
the rest of the team can integrate against the names before the bodies exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_width: int = 256, num_layers: int = 2):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        raise NotImplementedError("Member 1: build the policy MLP + Gaussian head")

    def forward(self, obs: torch.Tensor):
        """Return (action, log_prob) with the reparameterisation trick."""
        raise NotImplementedError


class EnsembleCritic(nn.Module):
    """Ensemble of Q-networks with LayerNorm (the RLPD critic)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_width: int = 256,
        num_layers: int = 2,
        ensemble_size: int = 10,
        layernorm: bool = True,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        raise NotImplementedError("Member 1: build the LayerNorm ensemble critic")

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values, shape (ensemble_size, N, 1)."""
        raise NotImplementedError
