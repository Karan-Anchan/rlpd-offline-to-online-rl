"""Roll out a trained checkpoint — the presentation demo.

    python demo.py                                   # config's run checkpoint, 5 episodes
    python demo.py --checkpoint checkpoints/NR1_Hopper_RLPD_seed0.pt --episodes 10
    python demo.py --video figures/hopper.mp4        # also save an mp4 (best-effort)

Prints per-episode raw + normalized return for a deterministic (greedy) policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wandb_logger import build_run_name, load_config, normalized_score
from rlpd.envs import make_env, env_dims
from rlpd.sac import RLPDAgent


def rollout(agent, env, episodes: int) -> tuple[list[float], list[int]]:
    returns, lengths = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total, steps = False, 0.0, 0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
    return returns, lengths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint", default=None, help="default: checkpoints/<run_name>.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video", default=None, help="path to save an mp4 (best-effort)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    env_id = cfg["env"]["id"]
    ckpt = Path(args.checkpoint) if args.checkpoint else Path("checkpoints") / f"{build_run_name(cfg)}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt} — train a run first")

    if args.video:
        import gymnasium as gym
        from gymnasium.wrappers import RecordVideo
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(12345)
        env = RecordVideo(env, video_folder=str(out.parent), name_prefix=out.stem,
                          episode_trigger=lambda i: True)
    else:
        env = make_env(env_id, seed=12345)

    obs_dim, act_dim = env_dims(env)
    agent = RLPDAgent(obs_dim, act_dim, cfg, args.device)
    state = torch.load(ckpt, map_location=args.device, weights_only=False)
    agent.load_state_dict(state["agent"])
    trained_step = state.get("step", "?")

    returns, lengths = rollout(agent, env, args.episodes)
    env.close()

    print(f"{env_id}  |  checkpoint {ckpt.name}  (env_step {trained_step})")
    for i, (ret, ln) in enumerate(zip(returns, lengths)):
        norm = normalized_score(env_id, ret)
        norm_s = f"{norm:6.1f}%" if norm is not None else "   n/a"
        print(f"  ep {i}: return {ret:8.1f}  normalized {norm_s}  length {ln}")
    mean = float(np.mean(returns))
    mean_norm = normalized_score(env_id, mean)
    tail = f" ({mean_norm:.1f}% normalized)" if mean_norm is not None else ""
    print(f"  mean over {args.episodes}: {mean:.1f}{tail}, mean length {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()
