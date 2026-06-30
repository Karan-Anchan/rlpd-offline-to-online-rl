"""Trivial stubs so each workstream can run before the real pieces exist.

`MockBuffer` lets Member 1 develop the agent with no data dependency; `StubAgent`
lets Member 3 build and run the training loop with no algorithm dependency. Both
honour the frozen interfaces. Delete once the real versions are integrated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .interfaces import Batch


class MockBuffer:
    """Returns random batches in the contract shape. No real storage."""

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.obs_dim, self.act_dim, self.device = obs_dim, act_dim, device

    def add(self, *args) -> None:
        pass

    def sample(self, n: int) -> Batch:
        rand = lambda d: torch.randn(n, d, dtype=torch.float32, device=self.device)
        return {
            "obs": rand(self.obs_dim),
            "action": rand(self.act_dim),
            "reward": rand(1),
            "next_obs": rand(self.obs_dim),
            "done": torch.zeros(n, 1, dtype=torch.float32, device=self.device),
        }


class StubAgent:
    """Random policy, no-op update. Drop-in for the real agent during wiring."""

    def __init__(self, act_dim: int):
        self.act_dim = act_dim

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, self.act_dim).astype(np.float32)

    def update(self, batch: Batch) -> Dict[str, float]:
        return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 1.0, "mean_q": 0.0}
