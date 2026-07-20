"""Per-setting (RLPD/SACfD/IQL) and per-env config profiles."""

from __future__ import annotations

# Named RLPD ablations: base RLPD with one component changed, run as
# experiment.setting=ablation-<name>. Each is (config section, key, value).
ABLATIONS = {
    "no-layernorm": ("algo", "layernorm", False),
    "utd1": ("algo", "utd", 1),
    "ensemble2": ("algo", "ensemble_size", 2),
    "ratio0": ("algo", "symmetric_sampling_ratio", 0.0),   # offline-only sampling
    "ratio1": ("algo", "symmetric_sampling_ratio", 1.0),   # online-only sampling
}
ABLATION_HUMANOID_STEPS = 500_000  # reduced budget; compare vs RLPD truncated to the same step


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

    # Ablations are base RLPD with one knob changed; applied last so they win over
    # the env profile (incl. the reduced Humanoid budget).
    if setting.startswith("ablation-"):
        name = setting[len("ablation-"):]
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
        if cfg["env"]["id"] == "Humanoid-v5":
            cfg["training"]["total_env_steps"] = ABLATION_HUMANOID_STEPS
        section, key, value = ABLATIONS[name]
        cfg[section][key] = value

    return cfg
