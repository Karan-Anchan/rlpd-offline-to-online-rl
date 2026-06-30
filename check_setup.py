"""Pipeline check: GPU, MuJoCo env, Minari dataset, and W&B. Does not train.

    python check_setup.py
    python check_setup.py --wandb-online
    python check_setup.py --skip-minari
"""

import argparse
import platform
import sys
import traceback

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def check_system():
    section("0. System")
    print(f"  Python   : {platform.python_version()} ({sys.executable})")
    print(f"  Platform : {platform.platform()}")


def check_torch():
    section("1. PyTorch / GPU")
    try:
        import torch
    except Exception as e:
        record("import torch", False, repr(e))
        return
    record("import torch", True, f"torch {torch.__version__}, cuda {torch.version.cuda}")

    if not torch.cuda.is_available():
        record("cuda available", False, "CPU-only build? reinstall from the cu128 index")
        return
    record("cuda available", True, torch.cuda.get_device_name(0))

    major, minor = torch.cuda.get_device_capability(0)
    record("compute capability", True, f"sm_{major}{minor}")

    # A kernel must actually launch; an unsupported build fails here.
    try:
        x = torch.randn(4096, 4096, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        record("gpu matmul", True, f"ran, sum={y:.1f}")
    except Exception as e:
        record("gpu matmul", False, repr(e))


def check_env():
    section("2. MuJoCo / Gymnasium env")
    try:
        import gymnasium as gym
    except Exception as e:
        record("import gymnasium", False, repr(e))
        return
    record("import gymnasium", True, f"gymnasium {gym.__version__}")

    try:
        env = gym.make("Hopper-v5")
    except Exception as e:
        record("make Hopper-v5", False, repr(e))
        return
    record("make Hopper-v5", True)

    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    env.close()
    record("reset + 10 steps", True, f"obs={env.observation_space.shape}, act={env.action_space.shape}")


def check_minari():
    section("3. Minari offline dataset")
    try:
        import minari
    except Exception as e:
        record("import minari", False, repr(e))
        return
    record("import minari", True, f"minari {minari.__version__}")

    try:
        remote = minari.list_remote_datasets()
        # Anchor on mujoco/hopper ("hopper" alone also matches Atari "choppercommand").
        hopper = sorted(k for k in remote if k.lower().startswith("mujoco/hopper"))
        dataset_id = next((k for k in hopper if "expert" in k), hopper[0])
        record("find dataset", True, dataset_id)
    except Exception as e:
        record("find dataset", False, repr(e))
        return

    try:
        ds = minari.load_dataset(dataset_id, download=True)
        record("download + load", True, f"{ds.total_episodes} episodes, {ds.total_steps} steps")
        ep = next(ds.iterate_episodes())
        record("sample transition", True,
               f"obs{tuple(ep.observations.shape)} act{tuple(ep.actions.shape)} rew{tuple(ep.rewards.shape)}")
    except Exception as e:
        record("download + load", False, repr(e))


def check_wandb(online):
    section("4. W&B logging")
    try:
        import wandb
    except Exception as e:
        record("import wandb", False, repr(e))
        return
    record("import wandb", True, f"wandb {wandb.__version__}")

    entity, project = None, "rlpd-lab"
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            wb = (yaml.safe_load(f) or {}).get("wandb", {})
        entity, project = wb.get("entity"), wb.get("project", project)
    except Exception:
        pass

    mode = "online" if online else "offline"
    try:
        run = wandb.init(entity=entity, project=project, name="NR1_Hopper_test_seed0",
                         mode=mode, config={"purpose": "pipeline_check"})
        wandb.log({"check/dummy_metric": 1.0})
        run.finish()
        record("wandb init+log", True, f"mode={mode} -> {entity}/{project}")
    except Exception as e:
        record("wandb init+log", False, repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--skip-minari", action="store_true")
    args = parser.parse_args()

    check_system()
    check_torch()
    check_env()
    if not args.skip_minari:
        check_minari()
    check_wandb(args.wandb_online)

    section("SUMMARY")
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    for name, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {len(RESULTS) - n_fail} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
