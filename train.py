"""Training loop — wires env + buffers + agent + logging.   OWNER: Member 3 (with M1).

This is the integration point. It knows only the frozen interfaces, so it runs
today against the stubs and keeps running unchanged as the real pieces land.

Two swap points, both marked SWAP below:
  * the agent     : StubAgent  -> rlpd.sac.RLPDAgent
  * the offline buffer: MockBuffer -> rlpd.dataset.load_offline_buffer

Run via run.py.
"""

from __future__ import annotations

from rlpd.envs import make_env, env_dims
from rlpd.evaluate import evaluate
from rlpd.replay_buffer import ReplayBuffer, symmetric_sample
from rlpd.stubs import MockBuffer, StubAgent
from wandb_logger import WandbLogger


def build_agent(cfg: dict, obs_dim: int, act_dim: int, device: str, use_stubs: bool):
    if use_stubs:
        return StubAgent(act_dim)
    # SWAP (Member 1): from rlpd.sac import RLPDAgent; return RLPDAgent(obs_dim, act_dim, cfg, device)
    from rlpd.sac import RLPDAgent
    return RLPDAgent(obs_dim, act_dim, cfg, device)


def build_offline_buffer(cfg: dict, obs_dim: int, act_dim: int, device: str, use_stubs: bool):
    if use_stubs:
        return MockBuffer(obs_dim, act_dim, device)
    # SWAP (Member 2's loader): real Minari offline data.
    from rlpd.dataset import load_offline_buffer
    return load_offline_buffer(cfg["dataset"]["minari_id"], obs_dim, act_dim, device)


def train(cfg: dict, device: str = "cuda", use_stubs: bool = False,
          max_steps: int | None = None) -> None:
    algo, tr = cfg["algo"], cfg["training"]
    seed = cfg["experiment"]["seed"]

    env = make_env(cfg["env"]["id"], seed=seed)
    eval_env = make_env(cfg["env"]["id"], seed=seed + 10_000)
    obs_dim, act_dim = env_dims(env)

    online = ReplayBuffer(tr["total_env_steps"], obs_dim, act_dim, device)
    offline = build_offline_buffer(cfg, obs_dim, act_dim, device, use_stubs)
    agent = build_agent(cfg, obs_dim, act_dim, device, use_stubs)
    logger = WandbLogger(cfg)

    batch_size = algo["batch_online"] + algo["batch_offline"]
    ratio = algo["symmetric_sampling_ratio"]
    total = max_steps if max_steps is not None else tr["total_env_steps"]

    obs, _ = env.reset(seed=seed)
    for step in range(total):
        if step < tr["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, term, trunc, _ = env.step(action)
        online.add(obs, action, reward, next_obs, float(term))  # termination only
        obs = next_obs if not (term or trunc) else env.reset()[0]

        if step >= tr["start_steps"]:
            for _ in range(algo["utd"]):
                batch = symmetric_sample(online, offline, batch_size, ratio=ratio)
                metrics = agent.update(batch)
            logger.log_train(metrics, env_step=step)

        if step % tr["eval_every_steps"] == 0 and step > 0:
            mean_ret, _ = evaluate(agent, eval_env, tr["eval_episodes"])
            logger.log_eval(mean_ret, env_step=step)

    logger.finish()
    env.close()
    eval_env.close()
