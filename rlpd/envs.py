"""Environment creation + wrappers."""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym


def make_env(env_id: str, seed: Optional[int] = None) -> gym.Env:
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def env_dims(env: gym.Env) -> Tuple[int, int]:
    return env.observation_space.shape[0], env.action_space.shape[0]
