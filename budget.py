"""Compute-budget accounting: summarize finished runs and project what's left.

    python budget.py                      # summary + projection for the full plan
    python budget.py --throughput 13      # project even with no logged runs yet

Each finished run appends a row to results/budget.csv (see wandb_logger). The
plan below is the reproduction (3 tasks x 3 seeds) plus the humanoid extension
(3 seeds), each at the config's per-run env-step budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wandb_logger import BUDGET_CSV, load_config

# (label, number of runs) at the per-run env-step budget from config.
PLAN = [("locomotion reproduction (3 tasks x 3 seeds)", 9),
        ("humanoid extension (3 seeds)", 3)]


def summarize(df: pd.DataFrame) -> None:
    total_steps = int(df["env_steps"].sum())
    total_hours = df["wall_seconds"].sum() / 3600.0
    print(df.to_markdown(index=False))
    print(f"\n{len(df)} runs | {total_steps:,} env-steps | {total_hours:.2f} GPU-hours")


def measured_throughput(df: pd.DataFrame) -> float | None:
    valid = df[df["wall_seconds"] > 0]
    if valid.empty:
        return None
    return float(valid["env_steps"].sum() / valid["wall_seconds"].sum())


def project(throughput: float, per_run_steps: int) -> None:
    print(f"\nprojection @ {throughput:.1f} env-steps/s, {per_run_steps:,} steps/run:")
    grand = 0.0
    for label, n_runs in PLAN:
        hours = n_runs * per_run_steps / throughput / 3600.0
        grand += hours
        print(f"  {label:44s} {n_runs:2d} runs  ~{hours:5.1f} h")
    print(f"  {'TOTAL':44s} {sum(n for _, n in PLAN):2d} runs  ~{grand:5.1f} h")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--throughput", type=float, default=None,
                   help="env-steps/s for projection (default: measured from budget.csv)")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    per_run_steps = int(load_config(args.config)["training"]["total_env_steps"])

    if BUDGET_CSV.exists():
        df = pd.read_csv(BUDGET_CSV)
        summarize(df)
        throughput = args.throughput or measured_throughput(df)
    else:
        print(f"no runs logged in {BUDGET_CSV} yet.")
        throughput = args.throughput

    if throughput:
        project(throughput, per_run_steps)
    else:
        print("\npass --throughput <steps/s> to project remaining cost.")


if __name__ == "__main__":
    main()
