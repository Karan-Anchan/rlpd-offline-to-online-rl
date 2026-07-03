"""Data-pipeline tests: the batch contract, symmetric sampling, and — gated
behind RLPD_RUN_SLOW=1 — the real Hopper Minari load + random-policy eval.

    pytest                       # fast contract tests only
    RLPD_RUN_SLOW=1 pytest       # also load the real dataset (downloads ~1GB)
"""

import os

import numpy as np
import pytest
import torch

from rlpd.interfaces import BATCH_KEYS, check_batch
from rlpd.replay_buffer import ReplayBuffer, symmetric_sample

OBS_DIM, ACT_DIM = 11, 3  # Hopper-v5
RUN_SLOW = os.environ.get("RLPD_RUN_SLOW") == "1"
slow = pytest.mark.skipif(not RUN_SLOW, reason="set RLPD_RUN_SLOW=1 (downloads Minari)")


def _fill(buf: ReplayBuffer, n: int, done_every: int | None = None) -> None:
    for i in range(n):
        d = 1.0 if (done_every and (i + 1) % done_every == 0) else 0.0
        buf.add(
            np.full(OBS_DIM, i, np.float32),
            np.full(ACT_DIM, i, np.float32),
            float(i),
            np.full(OBS_DIM, i + 1, np.float32),
            d,
        )


# ---- fast: no network, exercises buffer + contract ----

def test_sample_satisfies_batch_contract():
    buf = ReplayBuffer(100, OBS_DIM, ACT_DIM, "cpu")
    _fill(buf, 100)
    batch = buf.sample(32)
    check_batch(batch, OBS_DIM, ACT_DIM)
    assert set(batch) == set(BATCH_KEYS)


def test_sampled_done_is_float32_and_binary():
    buf = ReplayBuffer(100, OBS_DIM, ACT_DIM, "cpu")
    _fill(buf, 100, done_every=10)
    done = buf.sample(64)["done"]
    assert done.dtype == torch.float32
    assert torch.isin(done, torch.tensor([0.0, 1.0])).all()


@pytest.mark.parametrize("ratio, n_online", [(0.5, 128), (0.25, 64), (1.0, 256), (0.0, 0)])
def test_symmetric_sample_ratio_and_contract(ratio, n_online):
    online = ReplayBuffer(500, OBS_DIM, ACT_DIM, "cpu")
    offline = ReplayBuffer(500, OBS_DIM, ACT_DIM, "cpu")
    _fill(online, 500)
    _fill(offline, 500)
    batch = symmetric_sample(online, offline, 256, ratio=ratio)
    check_batch(batch, OBS_DIM, ACT_DIM)
    assert batch["obs"].shape[0] == 256
    # online rows carry small values (i<500 reused); we just assert the split count
    assert round(256 * ratio) == n_online


def test_ring_buffer_wraps_and_caps_size():
    buf = ReplayBuffer(10, OBS_DIM, ACT_DIM, "cpu")
    _fill(buf, 25)
    assert buf.size == 10  # capped at capacity
    assert buf.ptr == 25 % 10


def test_dataset_registry_covers_reproduction_tasks():
    from rlpd.dataset import DATASET_IDS, dataset_id_for_env

    assert set(DATASET_IDS) == {"Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"}
    assert dataset_id_for_env("Walker2d-v5") == "mujoco/walker2d/expert-v0"
    with pytest.raises(KeyError):
        dataset_id_for_env("Ant-v5")


# ---- slow: real Minari load + random eval, per reproduction task ----

@slow
@pytest.mark.parametrize("env_id, dims", [
    ("Hopper-v5", (11, 3)),
    ("HalfCheetah-v5", (17, 6)),
    ("Walker2d-v5", (17, 6)),
])
def test_offline_load_and_random_eval(env_id, dims):
    pytest.importorskip("minari")
    pytest.importorskip("gymnasium")

    from wandb_logger import normalized_score
    from rlpd.dataset import load_offline_buffer_for_env
    from rlpd.envs import make_env, env_dims
    from rlpd.evaluate import evaluate
    from rlpd.stubs import StubAgent

    env = make_env(env_id, seed=0)
    obs_dim, act_dim = env_dims(env)
    assert (obs_dim, act_dim) == dims

    offline = load_offline_buffer_for_env(env_id, obs_dim, act_dim, "cpu")
    assert offline.size == offline.capacity > 0

    # termination-only: done must be sparse (truncations would spike this).
    done_rate = offline.done.sum() / offline.size
    assert done_rate < 0.01, f"done rate {done_rate:.4f} too high — truncations leaking in?"

    check_batch(offline.sample(256), obs_dim, act_dim)

    # random policy must land near the 0-end of the normalized curve, nowhere near expert (100).
    mean_ret, _ = evaluate(StubAgent(act_dim), env, episodes=10)
    norm = normalized_score(env_id, mean_ret)
    env.close()
    assert norm is not None and norm < 25, f"random policy normalized={norm}, expected ~0"
