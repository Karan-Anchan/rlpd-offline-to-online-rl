"""Actor and LayerNorm ensemble critic."""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_width: int = 256, num_layers: int = 2):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        raise NotImplementedError("TODO: build the policy MLP + Gaussian head")

    def forward(self, obs: torch.Tensor):
        """Return (action, log_prob) with the reparameterisation trick."""
        raise NotImplementedError


class EnsembleCritic(nn.Module):
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
        raise NotImplementedError("TODO: build the LayerNorm ensemble critic")

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values, shape (ensemble_size, N, 1)."""
        raise NotImplementedError
