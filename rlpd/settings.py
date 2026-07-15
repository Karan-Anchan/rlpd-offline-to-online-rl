"""Per-setting (RLPD/SACfD/IQL) and per-env config profiles."""

from __future__ import annotations


def apply_setting(cfg: dict) -> dict:
    """Fill setting- and env-specific hyperparameters. Call after --set overrides."""
    setting = cfg["experiment"]["setting"]
    cfg.setdefault("sampling", "symmetric")

    if setting == "SACfD":  # vanilla SAC over a single offline+online buffer
        cfg["algo"].update(ensemble_size=2, layernorm=False, utd=1)
        cfg["sampling"] = "combined"
    elif setting == "IQL":
        cfg["algo"]["utd"] = 1
        cfg["sampling"] = "combined"
        cfg["training"]["start_steps"] = 0  # pretrained policy acts from step 0

    if cfg["env"]["id"] == "Humanoid-v5":  # larger obs; 1M budget across all methods
        cfg.setdefault("env_specific", {})["num_layers"] = 3
        cfg["training"]["total_env_steps"] = 1_000_000

    return cfg
