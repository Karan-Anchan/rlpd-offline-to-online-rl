"""CLI launcher: read config.yaml, apply overrides, start training.

    python run.py                                  # train from config.yaml
    python run.py --set env.id=Walker2d-v5 experiment.seed=1
    python run.py --stub --steps 50 --wandb-offline # wiring check, no GPU/data needed
"""

from __future__ import annotations

import argparse

from wandb_logger import load_config
from rlpd.settings import apply_setting
import train as train_mod


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def apply_overrides(cfg: dict, assignments: list[str]) -> dict:
    """Apply dotted key=value overrides, e.g. algo.utd=1."""
    for item in assignments:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _coerce(raw)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stub", action="store_true", help="use StubAgent + MockBuffer (wiring test)")
    p.add_argument("--steps", type=int, default=None, help="cap env steps (short runs)")
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from a checkpoint: --resume [path], no path = the "
                        "run's default checkpoint (checkpoints/<run_name>.pt)")
    args = p.parse_args()

    cfg = apply_setting(apply_overrides(load_config(args.config), args.set))
    if args.wandb_offline:
        cfg.setdefault("wandb", {})["mode"] = "offline"

    train_mod.train(cfg, device=args.device, use_stubs=args.stub, max_steps=args.steps,
                    resume=args.resume)


if __name__ == "__main__":
    main()
