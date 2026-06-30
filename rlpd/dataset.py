"""Minari offline datasets -> a ReplayBuffer.   OWNER: Member 2.

Loads a Minari dataset and fills a ReplayBuffer of all its transitions, ready to
be one half of `symmetric_sample`. The `done` flag uses **termination only**:
truncation (time-limit) must not zero the TD bootstrap or learning is corrupted.
"""

from __future__ import annotations

import minari

from .replay_buffer import ReplayBuffer


def load_offline_buffer(dataset_id: str, obs_dim: int, act_dim: int,
                        device: str = "cpu") -> ReplayBuffer:
    ds = minari.load_dataset(dataset_id, download=True)
    buf = ReplayBuffer(ds.total_steps, obs_dim, act_dim, device)
    for ep in ds.iterate_episodes():
        obs, act, rew = ep.observations, ep.actions, ep.rewards
        term = ep.terminations
        for t in range(len(act)):
            buf.add(obs[t], act[t], rew[t], obs[t + 1], float(term[t]))
    return buf
