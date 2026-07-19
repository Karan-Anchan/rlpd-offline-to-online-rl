"""Backfill Minari-v5 normalized eval onto finished W&B runs.

W&B history is append-only, so this adds `eval/return_normalized_v5` (+ summary fields),
computed from each run's local results/<name>.eval.csv `return_raw` column via the v5 anchors
in wandb_logger.REFERENCE_SCORES. Reading raw locally avoids the flaky history-scan service and
is immune to a stale normalized column. The old D4RL `eval/return_normalized` stays (hide it).

    python wandb_backfill_v5.py                         # dry run
    python wandb_backfill_v5.py --live                  # append to every finished run
    python wandb_backfill_v5.py --live --only NR1_Hopper_RLPD_medium_seed0

Idempotent (skips runs already carrying the v5 summary) and error-tolerant (skips + reports a
run that fails), so it is safe to run now and re-run after the remaining jobs finish.
"""
from __future__ import annotations

import argparse
import csv

import wandb

from wandb_logger import RESULTS_DIR, normalized_score

ENTITY, PROJECT = "fryan-nr", "NR1"
V5_KEY = "eval/return_normalized_v5"


def v5_points(run_name: str):
    f = RESULTS_DIR / f"{run_name}.eval.csv"
    if not f.exists():
        return None
    env_short = run_name.split("_")[1]  # NR1_<Env>_...
    pts = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                step = int(float(r["env_step"]))
                v = normalized_score(env_short, float(r["return_raw"]))
            except (KeyError, ValueError, TypeError):
                continue
            if v is not None:
                pts.append((step, v))
    pts.sort()
    return pts or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these run names")
    args = ap.parse_args()

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"{len(runs)} runs in {ENTITY}/{PROJECT}", flush=True)
    done = skipped = failed = 0
    for run in runs:
        if args.only and run.name not in args.only:
            continue
        if run.state == "running":
            continue
        if run.summary.get(V5_KEY + "_final") is not None:
            skipped += 1
            continue
        pts = v5_points(run.name)
        if not pts:
            continue
        v5_last, v5_max = pts[-1][1], max(v for _, v in pts)
        print(f"  {run.name} [{run.state}] pts={len(pts)} v5 last={v5_last:.2f} max={v5_max:.2f}"
              + ("" if args.live else "  (dry run)"), flush=True)
        if not args.live:
            continue
        try:
            r = wandb.init(entity=ENTITY, project=PROJECT, id=run.id, resume="must")
            wandb.define_metric("env_step")
            wandb.define_metric(V5_KEY, step_metric="env_step")
            for step, v in pts:
                wandb.log({V5_KEY: v, "env_step": step})
            wandb.summary[V5_KEY + "_final"] = v5_last
            wandb.summary[V5_KEY + "_max"] = v5_max
            r.finish()
            done += 1
        except Exception as e:  # keep going; re-run later to retry
            print(f"    FAILED {run.name}: {e}", flush=True)
            failed += 1
    print(f"done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
