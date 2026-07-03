# rlpd-offline-to-online-rl

Reproduction and humanoid extension of **RLPD** — *Efficient Online Reinforcement
Learning with Offline Data* (Ball, Smith, Kostrikov & Levine, ICML 2023) — in PyTorch,
using [Minari](https://minari.farama.org/) offline datasets.

## Scope

- **Reproduce** on `Hopper-v5`, `HalfCheetah-v5`, `Walker2d-v5` (3 seeds each).
- **Extend** to `Humanoid-v5` / `HumanoidStandup-v5`.
- **Ablate** the core design choices (symmetric sampling ratio, LayerNorm, ensemble size,
  UTD) plus one humanoid-specific ablation.
- **Baselines:** SAC + offline data, and IQL + finetuning.

RLPD is SAC plus three changes: 50/50 symmetric sampling of offline/online data, LayerNorm
in the critic, and a large critic ensemble with a high update-to-data ratio.

## Setup

Requires Python 3.11+ and an NVIDIA GPU with a recent driver (CUDA 12.8+ for RTX 50-series).

```powershell
./setup.ps1
```

Creates `.venv`, installs PyTorch + `requirements.txt`, and runs the setup check. PyTorch is
installed separately from the CUDA 12.8 index (see `setup.ps1`). Reactivate later with
`.\.venv\Scripts\Activate.ps1`.

## Setup check

Verifies GPU, MuJoCo env, Minari dataset, and logging without training:

```powershell
python check_setup.py
python check_setup.py --wandb-online
python check_setup.py --skip-minari
```

## Config and logging

`config.yaml` holds the hyperparameters (paper Table 1) and per-environment knobs.

Logging via [W&B](https://wandb.ai/) through `wandb_logger.py`: runs are named
`NR1_[Env]_[Setting]_[Seed]`, returns are logged raw and normalized on an `env_step`
x-axis, and seed/dataset/versions/commit are recorded for reproducibility.

```powershell
wandb login
```

Then set `wandb.entity` and `wandb.project` in `config.yaml`.

## Layout

Modules code against the frozen interfaces in `rlpd/interfaces.py` (the batch
contract + `Buffer`/`Agent` protocols) and the stubs in `rlpd/stubs.py`.

```
rlpd/
  interfaces.py     batch contract + protocols
  stubs.py          MockBuffer, StubAgent
  networks.py       Actor, EnsembleCritic
  sac.py            RLPD/SAC agent
  replay_buffer.py  online buffer + symmetric sampler
  dataset.py        Minari -> offline buffer
  envs.py           env creation + wrappers
  evaluate.py       eval loop
train.py            training loop (wires interfaces)
run.py              CLI launcher (config + overrides)
wandb_logger.py     logging
config.yaml         hyperparameters
requirements.txt    dependencies (PyTorch separate, see setup.ps1)
setup.ps1           environment bootstrap
check_setup.py      pipeline verification
```

Wiring check (no GPU or dataset needed):

```powershell
python run.py --stub --steps 60 --wandb-offline --device cpu
```

## License

[MIT](LICENSE).

## References

- Ball, Smith, Kostrikov, Levine. *Efficient Online Reinforcement Learning with Offline
  Data.* ICML 2023. https://github.com/ikostrikov/rlpd
- Minari: https://minari.farama.org/
- Gymnasium MuJoCo: https://gymnasium.farama.org/environments/mujoco/
