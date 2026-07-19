"""Build learning curves + a comparison table from results/*.eval.csv.

    python -m plotting.make_curves            # all figures + table into figures/
    python -m plotting.make_curves --metric return_raw

Each run writes results/NR1_<Env>_<Setting>_seed<N>.eval.csv (see wandb_logger).
Curves shade +/- std across seeds; the table reports the final eval per (env, setting).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
# Run names are NR1_<Env>_<Setting>[_<quality>]_seed<N>. The quality suffix is optional:
# the earliest runs predate it and were all trained on the expert datasets.
QUALITIES = ("simple", "medium", "expert")
RUN_RE = re.compile(
    r"^(?P<group>[^_]+)_(?P<env>[^_]+)_(?P<setting>[^_]+)"
    r"(?:_(?P<quality>" + "|".join(QUALITIES) + r"))?"
    r"_seed(?P<seed>\d+)$"
)


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Concatenate every eval CSV, tagged with env / setting / quality / seed from its name."""
    frames = []
    for csv in sorted(results_dir.glob("*.eval.csv")):
        m = RUN_RE.match(csv.name[: -len(".eval.csv")])
        if not m:
            print(f"skipping unparseable run name: {csv.name}")
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        df["env"], df["setting"], df["seed"] = m["env"], m["setting"], int(m["seed"])
        df["quality"] = m["quality"] or "expert"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Mean/std of `y` across seeds at each env_step, per (env, setting, quality)."""
    agg = (
        df.dropna(subset=[y])
        .groupby(["env", "setting", "quality", "env_step"])[y]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)  # single seed -> no band
    agg["label"] = agg["setting"] + " · " + agg["quality"]
    return agg


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, outdir: Path) -> list[Path]:
    """One figure per env: `metric` vs env_step, a line+band per setting."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    agg = _aggregate(df, metric)
    for env in sorted(agg["env"].unique()):
        sub = agg[agg["env"] == env]
        fig, ax = plt.subplots(figsize=(6, 4))
        for setting, s in sub.groupby("label"):
            s = s.sort_values("env_step")
            ax.plot(s["env_step"], s["mean"], label=setting, linewidth=2)
            ax.fill_between(s["env_step"], s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.2)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{env} — {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = outdir / f"{env}_{metric}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Final-eval performance per (env, setting, quality), averaged over seeds.

    `final_*` is the last eval of each run; `last5_normalized` averages each run's final five
    evals before averaging across seeds, which strips a lot of the per-eval noise on Hopper
    and HalfCheetah. Report both — they disagree by ~10 points where the run is still swingy.
    """
    keys = ["env", "setting", "quality", "seed"]
    last_step = df.groupby(keys)["env_step"].transform("max")
    finals = df[df["env_step"] == last_step]

    tail = (
        df.sort_values("env_step")
        .groupby(keys)
        .tail(5)
        .groupby(keys)["return_normalized"]
        .mean()
        .reset_index(name="last5_normalized")
    )

    table = (
        finals.groupby(["env", "setting", "quality"])
        .agg(
            seeds=("seed", "nunique"),
            # both ends: where they differ the seeds are ragged and the group's mean mixes
            # runs stopped at different horizons (e.g. the expert sweep: 245k / 57.5k / 57.5k)
            env_steps_min=("env_step", "min"),
            env_steps_max=("env_step", "max"),
            final_raw=("return_raw", "mean"),
            final_normalized=("return_normalized", "mean"),
            final_normalized_std=("return_normalized", "std"),
            final_mean_q=("mean_q", "mean"),
        )
        .reset_index()
    )
    tail_agg = tail.groupby(["env", "setting", "quality"])["last5_normalized"].mean().reset_index()
    table = table.merge(tail_agg, on=["env", "setting", "quality"], how="left")
    table["final_normalized_std"] = table["final_normalized_std"].fillna(0.0)
    return table.round(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    p.add_argument("--figures", type=Path, default=FIGURES_DIR)
    p.add_argument("--metric", default="return_normalized",
                   help="curve metric: return_normalized | return_raw")
    args = p.parse_args()

    df = load_runs(args.results)
    if df.empty:
        print(f"no eval CSVs in {args.results}/ yet — run a training job first.")
        return

    ylabel = {"return_normalized": "normalized return (Minari v5 expert = 100)",
              "return_raw": "raw return"}.get(args.metric, args.metric)
    written = plot_metric(df, args.metric, ylabel, args.figures)
    written += plot_metric(df, "mean_q", "mean Q (diagnostic)", args.figures)

    table = summary_table(df)
    args.figures.mkdir(parents=True, exist_ok=True)
    (args.figures / "summary_table.csv").write_text(table.to_csv(index=False), encoding="utf-8")
    md = table.to_markdown(index=False)
    (args.figures / "summary_table.md").write_text(md, encoding="utf-8")

    print(f"wrote {len(written)} figures to {args.figures}/")
    for pth in written:
        print(f"  {pth}")
    print("\nsummary:\n" + md)


if __name__ == "__main__":
    main()
