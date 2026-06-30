"""Environment creation + wrappers.   OWNER: Member 2.

One place to make envs so wrapper changes (especially for the Humanoid port) are
documented in a single file. Returns standard Gymnasium MuJoCo envs for now.
"""

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
    """Return (obs_dim, act_dim) for a flat Box obs/action env."""
    return env.observation_space.shape[0], env.action_space.shape[0]
