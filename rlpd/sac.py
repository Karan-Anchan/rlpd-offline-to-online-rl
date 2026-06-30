"""RLPD / SAC agent.   OWNER: Member 1 (algorithm).

Implements the `Agent` protocol from rlpd.interfaces. RLPD = SAC + three changes
(symmetric sampling lives in the buffer, LayerNorm + large ensemble live in the
critic, high UTD lives in the training loop). Skeleton only.

Integration point: rlpd/stubs.py:StubAgent is the placeholder the loop uses
today. Swap to RLPDAgent here once act()/update() are implemented.
"""

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
            "Member 1: build actor, ensemble critic + target, temperature, optimizers"
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def update(self, batch: Batch) -> Dict[str, float]:
        """One UTD-scaled update step. Must return at least critic_loss,
        actor_loss, alpha, mean_q."""
        raise NotImplementedError
