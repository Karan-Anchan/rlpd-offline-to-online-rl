"""Online replay buffer + symmetric sampler.   OWNER: Member 2.

Implements the `Buffer` protocol. `sample()` returns the frozen batch contract:
float32 tensors of shape (N, obs_dim)/(N, act_dim)/(N, 1) on `device`, with
`done` carrying termination only (the loader enforces this on the offline side).
"""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import Batch


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, act_dim), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.capacity, self.device = capacity, device
        self.size = self.ptr = 0

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = o
        self.action[i] = a
        self.reward[i] = r
        self.next_obs[i] = no
        self.done[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> Batch:
        idx = np.random.randint(0, self.size, size=n)
        t = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)
        return {
            "obs": t(self.obs),
            "action": t(self.action),
            "reward": t(self.reward),
            "next_obs": t(self.next_obs),
            "done": t(self.done),
        }


def symmetric_sample(online: ReplayBuffer, offline: ReplayBuffer, batch_size: int,
                     ratio: float = 0.5) -> Batch:
    """Concatenate `ratio` of the batch from online and the rest from offline.

    `ratio` is the online fraction (paper default 0.5 = the 50/50 split). It is
    exposed so the symmetric-sampling ablation can sweep it without touching the
    training loop; the loop calls this with the configured ratio.
    """
    n_online = round(batch_size * ratio)
    n_offline = batch_size - n_online
    a, b = online.sample(n_online), offline.sample(n_offline)
    return {k: torch.cat([a[k], b[k]], dim=0) for k in a}
