"""Load Minari offline datasets into a ReplayBuffer."""

from __future__ import annotations

import minari

from .replay_buffer import ReplayBuffer

# env id -> Minari dataset id (expert class, minari 0.5.2).
DATASET_IDS = {
    "Hopper-v5": "mujoco/hopper/expert-v0",            # obs 11, act 3, ~999k steps
    "HalfCheetah-v5": "mujoco/halfcheetah/expert-v0",  # obs 17, act 6,  1.0M steps
    "Walker2d-v5": "mujoco/walker2d/expert-v0",        # obs 17, act 6, ~999k steps
}


def dataset_id_for_env(env_id: str) -> str:
    try:
        return DATASET_IDS[env_id]
    except KeyError:
        raise KeyError(
            f"no offline dataset registered for {env_id!r}; known: {sorted(DATASET_IDS)}"
        ) from None


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
