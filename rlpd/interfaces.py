"""Frozen interfaces — the single source of truth for how the pieces fit together.

Do not change these signatures silently. The batch contract below is what the
buffers (Member 2) produce and what the agent (Member 1) consumes; the training
loop (Member 3) wires them with no knowledge of either implementation.

The batch contract — a dict of torch tensors, all float32, on the agent's device:

    obs       (N, obs_dim)   float32
    action    (N, act_dim)   float32
    reward    (N, 1)         float32
    next_obs  (N, obs_dim)   float32
    done      (N, 1)         float32   <- termination only, NOT truncation

`done` flags termination only: a time-limit truncation must not zero the TD
bootstrap target, or learning is quietly corrupted.
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import numpy as np
import torch

# A batch is exactly the five keys documented above.
Batch = Dict[str, torch.Tensor]

BATCH_KEYS = ("obs", "action", "reward", "next_obs", "done")


@runtime_checkable
class Buffer(Protocol):
    """Produces batches that satisfy the contract above."""

    def add(self, o, a, r, no, d) -> None:
        """Store one transition (obs, action, reward, next_obs, done)."""
        ...

    def sample(self, n: int) -> Batch:
        """Return a batch of `n` transitions as the contract dict."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Consumes batches; acts in the environment."""

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Return an action for a single observation (numpy, env-ready)."""
        ...

    def update(self, batch: Batch) -> Dict[str, float]:
        """One gradient step on a batch; return scalar metrics (incl. mean_q)."""
        ...


def check_batch(batch: Batch, obs_dim: int, act_dim: int) -> None:
    """Assert a batch satisfies the contract. Cheap; call it in tests/integration."""
    assert set(batch) == set(BATCH_KEYS), f"batch keys {set(batch)} != {set(BATCH_KEYS)}"
    n = batch["obs"].shape[0]
    expected = {
        "obs": (n, obs_dim),
        "action": (n, act_dim),
        "reward": (n, 1),
        "next_obs": (n, obs_dim),
        "done": (n, 1),
    }
    for k, shape in expected.items():
        assert tuple(batch[k].shape) == shape, f"{k}: {tuple(batch[k].shape)} != {shape}"
        assert batch[k].dtype == torch.float32, f"{k}: dtype {batch[k].dtype} != float32"
