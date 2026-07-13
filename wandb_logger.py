"""W&B logging.

Run naming NR1_[Env]_[Setting]_[Seed], raw and normalized returns on an env-step
x-axis, and run provenance (seed, dataset id, versions, git commit).
"""

import csv
import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

RESULTS_DIR = Path("results")  # local, gitignored; mirrors eval metrics for plotting
BUDGET_CSV = RESULTS_DIR / "budget.csv"  # one row per finished run (compute accounting)

# D4RL reference scores for 100 * (score - min) / (max - min); min=random, max=expert.
# Minari carries no reference scores; the v5/TQC expert datasets exceed 100 (~119-149%).
REFERENCE_SCORES = {
    "HalfCheetah": (-280.178953, 12135.0),
    "Hopper": (-20.272305, 3234.3),
    "Walker2d": (1.629008, 4592.3),
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env_short_name(env_id: str) -> str:
    return env_id.split("-")[0]


def normalized_score(env_id: str, raw_return: float) -> Optional[float]:
    key = env_short_name(env_id)
    if key not in REFERENCE_SCORES:
        return None
    lo, hi = REFERENCE_SCORES[key]
    return 100.0 * (raw_return - lo) / (hi - lo)


def build_run_name(cfg: dict) -> str:
    exp = cfg["experiment"]
    return f"{exp['group_id']}_{env_short_name(cfg['env']['id'])}_{exp['setting']}_seed{exp['seed']}"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict:
    versions = {}
    for pkg in ("torch", "gymnasium", "minari", "numpy"):
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            versions[pkg] = "n/a"
    return versions


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


class WandbLogger:
    def __init__(self, cfg: dict, extra_config: Optional[dict] = None,
                 run_id: Optional[str] = None):
        """`run_id` continues an existing W&B run (checkpoint resume)."""
        import wandb

        self.wandb = wandb
        self.env_id = cfg["env"]["id"]
        wb = cfg.get("wandb", {})

        config = dict(cfg)
        config["_meta"] = {
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        }
        if extra_config:
            config.update(extra_config)

        self.run_name = build_run_name(cfg)
        self.run = wandb.init(
            entity=wb.get("entity") or None,
            project=wb.get("project", "offline-to-online-rl"),
            name=self.run_name,
            group=cfg["experiment"]["group_id"],
            job_type=cfg["experiment"]["setting"],
            mode=wb.get("mode", "online"),
            config=config,
            id=run_id,
            resume="must" if run_id else None,
        )
        self.run_id = self.run.id
        self._last_train: dict = {}
        self._last_step = 0
        self._start = time.time()
        self.cfg = cfg
        self._eval_csv = RESULTS_DIR / f"{self.run_name}.eval.csv"

        # env_step is the x-axis for all curves.
        wandb.define_metric("env_step")
        wandb.define_metric("train/*", step_metric="env_step")
        wandb.define_metric("eval/*", step_metric="env_step")

    def log_train(self, metrics: dict, env_step: int) -> None:
        payload = {f"train/{k}": v for k, v in metrics.items()}
        payload["env_step"] = env_step
        self.wandb.log(payload)
        self._last_train = dict(metrics)
        self._last_step = max(self._last_step, env_step)

    def log_eval(self, raw_return: float, env_step: int, extra: Optional[dict] = None) -> Optional[float]:
        payload = {"eval/return_raw": raw_return, "env_step": env_step}
        norm = normalized_score(self.env_id, raw_return)
        if norm is not None:
            payload["eval/return_normalized"] = norm
        if extra:
            for k, v in extra.items():
                payload[f"eval/{k}"] = v
        self.wandb.log(payload)

        row = {"env_step": env_step, "return_raw": raw_return, "return_normalized": norm}
        row.update({k: self._last_train.get(k) for k in
                    ("mean_q", "critic_loss", "actor_loss", "alpha")})
        _append_csv(self._eval_csv, row)
        return norm

    def finish(self) -> None:
        try:
            exp = self.cfg["experiment"]
            _append_csv(BUDGET_CSV, {
                "run_name": self.run_name,
                "env": env_short_name(self.env_id),
                "setting": exp["setting"],
                "seed": exp["seed"],
                "env_steps": self._last_step,
                "wall_seconds": round(time.time() - self._start, 1),
            })
        except Exception:
            pass  # budget accounting must never break a run's teardown
        self.run.finish()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    cfg.setdefault("wandb", {})["mode"] = "offline"
    logger = WandbLogger(cfg)
    print("run name:", logger.run_name)
    logger.log_train({"critic_loss": 0.42, "mean_q": 12.3}, env_step=0)
    print("normalized(1500):", logger.log_eval(raw_return=1500.0, env_step=0))
    logger.finish()
