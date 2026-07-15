"""Load Minari offline datasets into a ReplayBuffer."""

from __future__ import annotations

import minari
import numpy as np

from .replay_buffer import ReplayBuffer

# Minari ships {medium, simple, expert} per env (no "random").
_SLUG = {
    "Hopper-v5": "mujoco/hopper",
    "HalfCheetah-v5": "mujoco/halfcheetah",
    "Walker2d-v5": "mujoco/walker2d",
    "Humanoid-v5": "mujoco/humanoid",
}
QUALITIES = ("medium", "simple", "expert")


def dataset_id_for_env(env_id: str, quality: str = "expert") -> str:
    if env_id not in _SLUG:
        raise KeyError(f"no offline dataset registered for {env_id!r}; known: {sorted(_SLUG)}")
    if quality not in QUALITIES:
        raise ValueError(f"unknown dataset quality {quality!r}; known: {QUALITIES}")
    return f"{_SLUG[env_id]}/{quality}-v0"


def load_offline_buffer(dataset_id: str, obs_dim: int, act_dim: int,
                        device: str = "cpu") -> ReplayBuffer:
    ds = minari.load_dataset(dataset_id, download=True)
    buf = ReplayBuffer(ds.total_steps, obs_dim, act_dim, device)
    for ep in ds.iterate_episodes():
        obs, act, rew = ep.observations, ep.actions, ep.rewards
        term = ep.terminations
        for t in range(len(act)):
            buf.add(obs[t], act[t], rew[t], obs[t + 1], float(term[t]))  # termination only
    return buf


def load_offline_buffer_for_env(env_id: str, obs_dim: int, act_dim: int,
                                device: str = "cpu") -> ReplayBuffer:
    return load_offline_buffer(dataset_id_for_env(env_id), obs_dim, act_dim, device)


def mean_return(dataset_id: str) -> float:
    """Mean episode return of a Minari dataset (normalization reference)."""
    ds = minari.load_dataset(dataset_id, download=True)
    return float(np.mean([float(ep.rewards.sum()) for ep in ds.iterate_episodes()]))
