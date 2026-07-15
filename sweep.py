"""Launch the run matrix (setting x env x seed x quality), N subprocesses at a time.

    python sweep.py --quality medium --max-parallel 3
    python sweep.py --setting SACfD IQL --env Hopper-v5 --seed 0 1 2 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--setting", nargs="+", default=["RLPD", "SACfD", "IQL"])
    p.add_argument("--env", nargs="+",
                   default=["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Humanoid-v5"])
    p.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--quality", nargs="+", default=["medium"])
    p.add_argument("--max-parallel", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = list(itertools.product(args.quality, args.setting, args.env, args.seed))
    print(f"{len(jobs)} runs | {args.max_parallel} at a time")
    for q, s, e, sd in jobs:
        print(f"  {s:6s} {e:15s} seed{sd} {q}")
    if args.dry_run:
        return

    running: list[subprocess.Popen] = []
    for q, s, e, sd in jobs:
        while len(running) >= args.max_parallel:
            running = [r for r in running if r.poll() is None]
            if len(running) >= args.max_parallel:
                time.sleep(5)
        cmd = [sys.executable, str(ROOT / "run.py"), "--set",
               f"experiment.setting={s}", f"env.id={e}",
               f"experiment.seed={sd}", f"dataset.quality={q}"]
        print("launch:", " ".join(cmd[3:]))
        running.append(subprocess.Popen(cmd))

    for r in running:
        r.wait()
    print("sweep done")


if __name__ == "__main__":
    main()
