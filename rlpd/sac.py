"""RLPD/SAC agent implementing the Agent protocol."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .interfaces import Batch
from .networks import Actor, EnsembleCritic


class RLPDAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
        self.obs_dim, self.act_dim, self.device = obs_dim, act_dim, device
        self.cfg = cfg
        raise NotImplementedError(
            "TODO: build actor, ensemble critic + target, temperature, optimizers"
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def update(self, batch: Batch) -> Dict[str, float]:
        """Return metrics incl. critic_loss, actor_loss, alpha, mean_q."""
        raise NotImplementedError
