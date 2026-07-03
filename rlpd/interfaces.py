"""Batch contract and the Buffer/Agent protocols.

Batch: dict of float32 torch tensors on the agent's device:
    obs (N, obs_dim), action (N, act_dim), reward (N, 1),
    next_obs (N, obs_dim), done (N, 1)   # done = termination only, not truncation
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import numpy as np
import torch

Batch = Dict[str, torch.Tensor]
BATCH_KEYS = ("obs", "action", "reward", "next_obs", "done")


@runtime_checkable
class Buffer(Protocol):
    def add(self, o, a, r, no, d) -> None: ...
    def sample(self, n: int) -> Batch: ...


@runtime_checkable
class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def update(self, batch: Batch) -> Dict[str, float]: ...


def check_batch(batch: Batch, obs_dim: int, act_dim: int) -> None:
    """Assert a batch matches the contract (shapes + float32)."""
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
