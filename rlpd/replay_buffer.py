"""Online replay buffer + symmetric sampler."""

from __future__ import annotations

import numpy as np
import torch

from .interfaces import Batch

# Field names in batch-contract order (see rlpd/interfaces.py).
FIELDS = ("obs", "action", "reward", "next_obs", "done")


class ReplayBuffer:
    """Fixed-capacity ring buffer backed by numpy arrays.

    `ptr` is the next write position; once full, new rows overwrite the oldest.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, act_dim), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

    def add(self, o, a, r, no, d) -> None:
        i = self.ptr
        self.obs[i] = o
        self.action[i] = a
        self.reward[i] = r
        self.next_obs[i] = no
        self.done[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def save(self, path: str) -> None:
        """Persist contents + ring-buffer position (only the filled rows)."""
        n = self.size
        np.savez_compressed(
            path,
            obs=self.obs[:n],
            action=self.action[:n],
            reward=self.reward[:n],
            next_obs=self.next_obs[:n],
            done=self.done[:n],
            ptr=self.ptr,
            size=self.size,
        )

    def load(self, path: str) -> None:
        with np.load(path) as data:
            n = int(data["size"])
            if n > self.capacity:
                raise ValueError(
                    f"checkpoint has {n} rows, buffer capacity is {self.capacity}"
                )
            for name in FIELDS:
                getattr(self, name)[:n] = data[name]
            self.size = n
            self.ptr = int(data["ptr"]) % self.capacity

    def sample(self, n: int) -> Batch:
        """n rows drawn uniformly (with replacement) as float32 device tensors."""
        idx = np.random.randint(0, self.size, size=n)
        batch: Batch = {}
        for name in FIELDS:
            rows = getattr(self, name)[idx]
            batch[name] = torch.as_tensor(rows, dtype=torch.float32, device=self.device)
        return batch


def symmetric_sample(
    online: ReplayBuffer, offline: ReplayBuffer, batch_size: int, ratio: float = 0.5
) -> Batch:
    """Batch of `ratio` online rows + the rest offline (ratio=0.5 is the 50/50 split)."""
    n_online = round(batch_size * ratio)
    online_part = online.sample(n_online)
    offline_part = offline.sample(batch_size - n_online)

    batch: Batch = {}
    for name in FIELDS:
        batch[name] = torch.cat([online_part[name], offline_part[name]], dim=0)
    return batch


def combined_sample_many(
    online: ReplayBuffer, offline: ReplayBuffer, batch_size: int, count: int
) -> list[Batch]:
    """`count` batches drawn from the online+offline union, proportional to size (SACfD)."""
    total = batch_size * count
    n_on = round(total * online.size / (online.size + offline.size))
    parts = ([online.sample(n_on)] if n_on > 0 else []) + [offline.sample(total - n_on)]
    cat = {name: torch.cat([p[name] for p in parts], dim=0) for name in FIELDS}
    perm = torch.randperm(total, device=cat["obs"].device)
    return [{name: cat[name][perm[i * batch_size:(i + 1) * batch_size]] for name in FIELDS}
            for i in range(count)]


def symmetric_sample_many(
    online: ReplayBuffer,
    offline: ReplayBuffer,
    batch_size: int,
    count: int,
    ratio: float = 0.5,
) -> list[Batch]:
    """`count` symmetric batches drawn with one buffer read + one transfer each side."""
    n_online = round(batch_size * ratio)
    n_offline = batch_size - n_online

    # One big draw per side, covering all `count` batches.
    online_part = online.sample(n_online * count)
    offline_part = offline.sample(n_offline * count)

    # Reshape each side to (count, rows_per_batch, dim) and glue them together,
    # giving one (count, batch_size, dim) tensor per field.
    stacked: dict[str, torch.Tensor] = {}
    for name in FIELDS:
        online_rows = online_part[name].view(count, n_online, -1)
        offline_rows = offline_part[name].view(count, n_offline, -1)
        stacked[name] = torch.cat([online_rows, offline_rows], dim=1)

    # Batch i is row i of every stacked tensor — a view, no copy.
    batches: list[Batch] = []
    for i in range(count):
        batches.append({name: stacked[name][i] for name in FIELDS})
    return batches
