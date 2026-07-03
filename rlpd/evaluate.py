"""Deterministic evaluation rollout; returns (mean, std) raw return."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def evaluate(agent, env, episodes: int) -> Tuple[float, float]:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))
