"""Training loop: wires env, buffers, agent, and logging via the frozen interfaces."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from rlpd.envs import make_env, env_dims
from rlpd.evaluate import evaluate
from rlpd.replay_buffer import ReplayBuffer, symmetric_sample_many
from rlpd.stubs import MockBuffer, StubAgent
from wandb_logger import WandbLogger, build_run_name


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch so a (seed, config) pair reproduces a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_agent(cfg: dict, obs_dim: int, act_dim: int, device: str, use_stubs: bool):
    if use_stubs:
        return StubAgent(act_dim)
    from rlpd.sac import RLPDAgent
    return RLPDAgent(obs_dim, act_dim, cfg, device)


def build_offline_buffer(cfg: dict, obs_dim: int, act_dim: int, device: str, use_stubs: bool):
    if use_stubs:
        return MockBuffer(obs_dim, act_dim, device)
    from rlpd.dataset import load_offline_buffer, dataset_id_for_env
    dataset_id = cfg.get("dataset", {}).get("minari_id") or dataset_id_for_env(cfg["env"]["id"])
    return load_offline_buffer(dataset_id, obs_dim, act_dim, device)


# --------------------------------------------------------------- checkpointing

def default_checkpoint_path(cfg: dict) -> Path:
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    return ckpt_dir / f"{build_run_name(cfg)}.pt"


def save_checkpoint(path: Path, step: int, agent, online: ReplayBuffer,
                    wandb_run_id: str | None) -> None:
    """Atomic write (tmp + replace) so a preemption mid-save can't corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    buffer_path = path.with_suffix(".buffer.npz")
    online.save(str(buffer_path) + ".tmp.npz")
    os.replace(str(buffer_path) + ".tmp.npz", buffer_path)

    state = {
        "step": step,
        "agent": agent.state_dict(),
        "wandb_run_id": wandb_run_id,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)


def load_checkpoint(path: Path, agent, online: ReplayBuffer, device: str) -> dict:
    """Restore agent/buffer/RNG in place; return {step, wandb_run_id}."""
    state = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(state["agent"])
    online.load(path.with_suffix(".buffer.npz"))

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])

    return {"step": state["step"], "wandb_run_id": state.get("wandb_run_id")}


# --------------------------------------------------------------------- training

def train(cfg: dict, device: str = "cuda", use_stubs: bool = False,
          max_steps: int | None = None, resume: str | None = None) -> None:
    algo = cfg["algo"]
    tr = cfg["training"]
    seed = cfg["experiment"]["seed"]
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+

    # --- build every component ---
    env = make_env(cfg["env"]["id"], seed=seed)
    eval_env = make_env(cfg["env"]["id"], seed=seed + 10_000)
    obs_dim, act_dim = env_dims(env)

    online = ReplayBuffer(tr["total_env_steps"], obs_dim, act_dim, device)
    offline = build_offline_buffer(cfg, obs_dim, act_dim, device, use_stubs)
    agent = build_agent(cfg, obs_dim, act_dim, device, use_stubs)

    # --- resume from a checkpoint, if asked ---
    ckpt_path = default_checkpoint_path(cfg)
    ckpt_every = int(tr.get("checkpoint_every_steps", 0))
    can_checkpoint = hasattr(agent, "state_dict")  # StubAgent has no state
    start_step = 0
    wandb_run_id = None
    if resume is not None:
        resume_path = ckpt_path if resume == "auto" else Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"no checkpoint to resume from at {resume_path}")
        restored = load_checkpoint(resume_path, agent, online, device)
        start_step = restored["step"] + 1
        wandb_run_id = restored["wandb_run_id"]
        print(f"resumed from {resume_path} at env_step {start_step}")

    logger = WandbLogger(cfg, run_id=wandb_run_id)

    batch_size = algo["batch_online"] + algo["batch_offline"]
    ratio = algo["symmetric_sampling_ratio"]
    utd = algo["utd"]
    total = max_steps if max_steps is not None else tr["total_env_steps"]

    # --- main loop ---
    # The env can't be checkpointed, so a resumed run starts on a fresh episode.
    obs, _ = env.reset(seed=seed)
    for step in range(start_step, total):
        # 1. Act: random exploration during warm-up, the policy afterwards.
        if step < tr["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        # 2. Step the env and store the transition (termination only, never truncation).
        next_obs, reward, term, trunc, _ = env.step(action)
        online.add(obs, action, reward, next_obs, float(term))
        if term or trunc:
            obs = env.reset()[0]
        else:
            obs = next_obs

        # 3. Gradient updates. High UTD applies to the critic only; the actor and
        #    temperature update once per env step, on the last minibatch
        #    (REDQ / official RLPD).
        if step >= tr["start_steps"]:
            batches = symmetric_sample_many(online, offline, batch_size, utd, ratio=ratio)
            for i, batch in enumerate(batches):
                is_last = i == utd - 1
                metrics = agent.update(batch, update_actor=is_last)
            logger.log_train(metrics, env_step=step)

        # 4. Periodic deterministic evaluation.
        if step % tr["eval_every_steps"] == 0 and step > 0:
            mean_ret, _ = evaluate(agent, eval_env, tr["eval_episodes"])
            logger.log_eval(mean_ret, env_step=step)

        # 5. Periodic checkpoint (overwrites the same file).
        if can_checkpoint and ckpt_every > 0 and (step + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_path, step, agent, online, logger.run_id)

    # Final checkpoint so a finished run can also be inspected/extended later.
    if can_checkpoint and ckpt_every > 0 and total > start_step:
        save_checkpoint(ckpt_path, total - 1, agent, online, logger.run_id)

    logger.finish()
    env.close()
    eval_env.close()
