"""Actor and LayerNorm ensemble critic."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Squashed-Gaussian log-std clamp
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes: List[int], layernorm: bool, activation=nn.ReLU) -> nn.Sequential:
    """[Linear -> (LayerNorm) -> activation] per hidden layer; bare Linear output.

    `sizes` is [input, hidden..., output]; e.g. [11, 256, 256, 1] builds two
    hidden layers of width 256 and a linear head with no activation.
    """
    layers: List[nn.Module] = []
    num_linear = len(sizes) - 1
    for i in range(num_linear):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_hidden = i < num_linear - 1  # the last Linear is the output head
        if is_hidden:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Squashed-Gaussian (tanh) policy, SAC-style.

    obs -> trunk (num_layers hidden layers) -> mu head and log_std head.
    An action is sampled from Normal(mu, std) and squashed through tanh so it
    lands in [-1, 1].
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_width: int = 256, num_layers: int = 2,
        log_std_min: float = LOG_STD_MIN,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # SAC samples its own actions, so a low floor (-20) is safe; IQL/AWR scores fixed
        # dataset actions, where a collapsed std makes log_prob explode -> use a higher floor.
        self.log_std_min = log_std_min
        # Trunk ends with an activation so the mu/log_std heads read a nonlinear
        # feature; num_layers hidden layers total (paper: 2 for locomotion).
        self.trunk = mlp([obs_dim] + [hidden_width] * num_layers, layernorm=False)
        self.trunk.append(nn.ReLU())
        self.mu = nn.Linear(hidden_width, act_dim)
        self.log_std = nn.Linear(hidden_width, act_dim)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (action, log_prob). action is tanh-squashed to [-1, 1]."""
        # Gaussian parameters from the trunk features.
        features = self.trunk(obs)
        mu = self.mu(features)
        log_std = self.log_std(features).clamp(self.log_std_min, LOG_STD_MAX)
        dist = torch.distributions.Normal(mu, log_std.exp())

        # Pre-squash sample u (reparameterised so gradients flow), then squash.
        u = mu if deterministic else dist.rsample()
        action = torch.tanh(u)

        # Log-probability of the squashed action
        log_prob: Optional[torch.Tensor] = None
        if with_logprob:
            log_prob = dist.log_prob(u).sum(-1, keepdim=True)
            log_prob -= 2.0 * (
                torch.log(torch.tensor(2.0, device=u.device)) - u - F.softplus(-2.0 * u)
            ).sum(-1, keepdim=True)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log pi(a|s) for a given tanh-squashed action (IQL / AWR)."""
        features = self.trunk(obs)
        mu = self.mu(features)
        log_std = self.log_std(features).clamp(self.log_std_min, LOG_STD_MAX)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = torch.atanh(action.clamp(-0.999999, 0.999999))
        logp = dist.log_prob(u).sum(-1, keepdim=True)
        logp -= 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u)).sum(-1, keepdim=True)
        return logp


class ValueNet(nn.Module):
    """State-value V(s) for IQL."""

    def __init__(self, obs_dim: int, hidden_width: int = 256, num_layers: int = 2):
        super().__init__()
        self.net = mlp([obs_dim] + [hidden_width] * num_layers + [1], layernorm=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class EnsembleLinear(nn.Module):
    """E independent Linear layers evaluated in parallel with one batched matmul."""

    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        for e in range(ensemble_size):
            ref = nn.Linear(in_features, out_features)  # default kaiming-uniform init
            self.weight.data[e] = ref.weight.data.t()
            self.bias.data[e, 0] = ref.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, N, in) -> (E, N, out)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleCritic(nn.Module):
    """E independent critics with LayerNorm, computed in parallel (default E=10)."""

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
        sizes = [obs_dim + act_dim] + [hidden_width] * num_layers + [1]

        layers: List[nn.Module] = []
        num_linear = len(sizes) - 1
        for i in range(num_linear):
            layers.append(EnsembleLinear(sizes[i], sizes[i + 1], ensemble_size))
            is_hidden = i < num_linear - 1
            if is_hidden:
                if layernorm:
                    # Normalises per (member, sample) over the feature dim; the
                    # affine params are shared across members (standard, harmless).
                    layers.append(nn.LayerNorm(sizes[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values, shape (ensemble_size, N, 1)."""
        x = torch.cat([obs, action], dim=-1)  # (N, in)
        x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)  # (E, N, in)
        return self.net(x)
