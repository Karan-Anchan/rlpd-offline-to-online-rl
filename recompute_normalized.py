"""Recompute return_normalized from return_raw across results/*.eval.csv using the current
wandb_logger.REFERENCE_SCORES. Idempotent; preserves each file's own columns.

    python recompute_normalized.py
    python recompute_normalized.py --exclude NR1_Humanoid_RLPD_medium_seed2  # skip a live run
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from wandb_logger import RESULTS_DIR, normalized_score

SUFFIX = ".eval.csv"


def recompute_file(path: Path) -> tuple[int, int]:
    env_short = path.name.split("_")[1]  # NR1_<Env>_...
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "return_raw" not in fields or "return_normalized" not in fields:
        return 0, 0
    changed = 0
    for row in rows:
        try:
            norm = normalized_score(env_short, float(row["return_raw"]))
        except (TypeError, ValueError, KeyError):
            continue
        new = "" if norm is None else f"{norm:.4f}"
        if row.get("return_normalized", "") != new:
            changed += 1
        row["return_normalized"] = new
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return len(rows), changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exclude", nargs="*", default=[],
                   help="run names (eval-CSV stem) to skip, e.g. a still-running run")
    args = p.parse_args()
    exclude = set(args.exclude)
    for path in sorted(RESULTS_DIR.glob("*" + SUFFIX)):
        run_name = path.name[: -len(SUFFIX)]
        if run_name in exclude:
            print(f"{run_name:42s} SKIPPED (active)")
            continue
        n, changed = recompute_file(path)
        print(f"{run_name:42s} rows={n:4d} updated={changed}")


if __name__ == "__main__":
    main()
