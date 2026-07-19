"""Build the two figures embedded in README.md from results/*.eval.csv.

    python -m plotting.make_readme_figures

Writes assets/returns.png (normalized return, RLPD vs baselines) and assets/mean_q.png
(critic value estimates, log scale). Both are 1x3 panels over the locomotion tasks on the
medium-quality offline datasets, shading +/- std across seeds.

These used to be produced ad hoc, so the README drifted out of sync with the data whenever
normalization changed. Regenerate them alongside `python -m plotting.make_curves`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

from plotting.make_curves import RESULTS_DIR, load_runs

ASSETS_DIR = Path("assets")
ENVS = ["Hopper", "Walker2d", "HalfCheetah"]
QUALITY = "medium"

# One colour per method, held constant across both figures.
METHODS = {
    "RLPD": "#3b82f6",
    "IQL": "#f59e0b",
    "SACfD": "#ef4444",
}


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.tick_params(colors="#4b5563", labelsize=9)


def _panels(title: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, len(ENVS), figsize=(13, 3.6), sharex=True)
    fig.suptitle(title, fontsize=11, color="#374151", y=1.02)
    return fig, list(axes)


def _series(agg, env: str, setting: str):
    """Mean/std curve for one (env, setting) on the medium datasets, or None if absent."""
    s = agg[(agg["env"] == env) & (agg["setting"] == setting) & (agg["quality"] == QUALITY)]
    return s.sort_values("env_step") if not s.empty else None


def _aggregate(df, metric: str):
    agg = (
        df.dropna(subset=[metric])
        .groupby(["env", "setting", "quality", "env_step"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def returns_figure(df, outdir: Path) -> Path:
    agg = _aggregate(df, "return_normalized")
    fig, axes = _panels(
        "Normalized return on the medium offline datasets — mean ± std over 3 seeds"
    )
    for ax, env in zip(axes, ENVS):
        ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1)
        ax.text(0.015, 101, "Minari v5 expert", fontsize=7.5, color="#6b7280",
                transform=ax.get_yaxis_transform())
        for setting, colour in METHODS.items():
            s = _series(agg, env, setting)
            if s is None:
                continue
            ax.plot(s["env_step"], s["mean"], color=colour, linewidth=1.8, label=setting)
            ax.fill_between(s["env_step"], s["mean"] - s["std"], s["mean"] + s["std"],
                            color=colour, alpha=0.15, linewidth=0)
        ax.set_title(f"{env}-v5", fontsize=10, color="#111827")
        ax.set_xlabel("environment steps", fontsize=9)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].set_ylabel("normalized return", fontsize=9)
    # lower right: the expert-line label owns the top-left corner
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "returns.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def mean_q_figure(df, outdir: Path) -> Path:
    agg = _aggregate(df, "mean_q")
    fig, axes = _panels(
        "Critic value estimates (log scale) — RLPD stays bounded; SACfD blows up on Walker2d"
    )
    for ax, env in zip(axes, ENVS):
        for setting, colour in METHODS.items():
            s = _series(agg, env, setting)
            if s is None:
                continue
            ax.plot(s["env_step"], s["mean"].clip(lower=1e-1), color=colour,
                    linewidth=1.8, label=setting)
        ax.set_yscale("log")
        ax.set_title(f"{env}-v5", fontsize=10, color="#111827")
        ax.set_xlabel("environment steps", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("mean Q", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "mean_q.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    args = p.parse_args()

    df = load_runs(args.results)
    if df.empty:
        print(f"no eval CSVs in {args.results}/ yet — run a training job first.")
        return

    for path in (returns_figure(df, args.assets), mean_q_figure(df, args.assets)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
