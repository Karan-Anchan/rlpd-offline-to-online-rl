"""Build the figures embedded in README.md from results/*.eval.csv.

    python -m plotting.make_readme_figures

Writes to assets/:
  returns.png    normalized return, RLPD vs baselines, locomotion medium datasets
  mean_q.png     critic value estimates (log), same runs
  quality.png    RLPD across simple / medium / expert offline data
  humanoid.png   RLPD vs baselines on Humanoid (return + critic)
  ablations.png  RLPD Humanoid ablations, one component off at a time (return + critic)

All aggregate mean +/- std across seeds. Regenerate whenever normalization or the data changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from plotting.make_curves import RESULTS_DIR, load_runs

ASSETS_DIR = Path("assets")
ENVS = ["Hopper", "Walker2d", "HalfCheetah"]

METHODS = {"RLPD": "#3b82f6", "IQL": "#f59e0b", "SACfD": "#ef4444"}
QUALITIES = {"simple": "#c7d2fe", "medium": "#6366f1", "expert": "#312e81"}
# Humanoid ablations: the reference first, then one component changed each.
ABLATIONS = {
    "RLPD": ("RLPD (reference)", "#111827"),
    "ablation-ratio1": ("online-only (ratio 1)", "#3b82f6"),
    "ablation-ensemble2": ("ensemble 2", "#10b981"),
    "ablation-utd1": ("UTD 1", "#f59e0b"),
    "ablation-ratio0": ("offline-only (ratio 0)", "#8b5cf6"),
    "ablation-no-layernorm": ("no LayerNorm", "#ef4444"),
}
ABLATION_BUDGET = 500_000


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.tick_params(colors="#4b5563", labelsize=9)


def _curve(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean/std of `metric` over seeds at each env_step for an already-filtered frame."""
    agg = (
        df.dropna(subset=[metric])
        .groupby("env_step")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("env_step")
    )
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def _plot(ax: plt.Axes, df: pd.DataFrame, metric: str, colour: str, label: str) -> None:
    c = _curve(df, metric)
    if c.empty:
        return
    ax.plot(c["env_step"], c["mean"], color=colour, linewidth=1.8, label=label)
    ax.fill_between(c["env_step"], c["mean"] - c["std"], c["mean"] + c["std"],
                    color=colour, alpha=0.15, linewidth=0)


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def returns_figure(df: pd.DataFrame, outdir: Path) -> Path:
    med = df[df["quality"] == "medium"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    fig.suptitle("Normalized return on the medium offline datasets — mean ± std over 3 seeds",
                 fontsize=11, color="#374151", y=1.02)
    for ax, env in zip(axes, ENVS):
        ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1)
        ax.text(0.015, 101, "Minari v5 expert", fontsize=7.5, color="#6b7280",
                transform=ax.get_yaxis_transform())
        for setting, colour in METHODS.items():
            _plot(ax, med[(med["env"] == env) & (med["setting"] == setting)],
                  "return_normalized", colour, setting)
        ax.set_title(f"{env}-v5", fontsize=10, color="#111827")
        ax.set_xlabel("environment steps", fontsize=9)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].set_ylabel("normalized return", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, outdir, "returns.png")


def mean_q_figure(df: pd.DataFrame, outdir: Path) -> Path:
    med = df[df["quality"] == "medium"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    fig.suptitle("Critic value estimates (log scale) — RLPD stays bounded; SACfD blows up on Walker2d",
                 fontsize=11, color="#374151", y=1.02)
    for ax, env in zip(axes, ENVS):
        for setting, colour in METHODS.items():
            c = _curve(med[(med["env"] == env) & (med["setting"] == setting)], "mean_q")
            if not c.empty:
                ax.plot(c["env_step"], c["mean"].clip(lower=1e-1), color=colour,
                        linewidth=1.8, label=setting)
        ax.set_yscale("log")
        ax.set_title(f"{env}-v5", fontsize=10, color="#111827")
        ax.set_xlabel("environment steps", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("mean Q", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, outdir, "mean_q.png")


def quality_figure(df: pd.DataFrame, outdir: Path) -> Path:
    rlpd = df[df["setting"] == "RLPD"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    fig.suptitle("RLPD vs. offline-data quality — better prior data → better online policy",
                 fontsize=11, color="#374151", y=1.02)
    for ax, env in zip(axes, ENVS):
        ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1)
        for quality, colour in QUALITIES.items():
            _plot(ax, rlpd[(rlpd["env"] == env) & (rlpd["quality"] == quality)],
                  "return_normalized", colour, quality)
        ax.set_title(f"{env}-v5", fontsize=10, color="#111827")
        ax.set_xlabel("environment steps", fontsize=9)
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[0].set_ylabel("normalized return", fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", title="offline data")
    return _save(fig, outdir, "quality.png")


def humanoid_figure(df: pd.DataFrame, outdir: Path) -> Path:
    h = df[(df["env"] == "Humanoid") & (df["quality"] == "medium")]
    fig, (ax_r, ax_q) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle("Humanoid-v5 (beyond the paper) — RLPD holds; SACfD's critic diverges",
                 fontsize=11, color="#374151", y=1.02)
    for setting, colour in METHODS.items():
        sub = h[h["setting"] == setting]
        _plot(ax_r, sub, "return_normalized", colour, setting)
        cq = _curve(sub, "mean_q")
        if not cq.empty:
            ax_q.plot(cq["env_step"], cq["mean"].clip(lower=1e-1), color=colour,
                      linewidth=1.8, label=setting)
    ax_r.set_title("normalized return", fontsize=10, color="#111827")
    ax_r.set_ylabel("normalized return", fontsize=9)
    ax_r.set_ylim(bottom=0)
    ax_q.set_title("mean Q (log scale)", fontsize=10, color="#111827")
    ax_q.set_ylabel("mean Q", fontsize=9)
    ax_q.set_yscale("log")
    for ax in (ax_r, ax_q):
        ax.set_xlabel("environment steps", fontsize=9)
        _style(ax)
    ax_r.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, outdir, "humanoid.png")


def ablations_figure(df: pd.DataFrame, outdir: Path) -> Path:
    h = df[(df["env"] == "Humanoid") & (df["quality"] == "medium")]
    fig, (ax_r, ax_q) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle("Humanoid RLPD ablations (500k) — online-only beats RLPD even with expert data",
                 fontsize=11, color="#374151", y=1.02)
    # base RLPD (medium, seed 0) reference + expert-data RLPD, both truncated to the ablation budget
    refs = [("RLPD (medium data)", "#111827", h[(h["setting"] == "RLPD") & (h["seed"] == 0)]),
            ("RLPD (expert data)", "#9ca3af",
             df[(df["env"] == "Humanoid") & (df["setting"] == "RLPD") & (df["quality"] == "expert")
                & (df["seed"] == 0)])]
    for label, colour, sub in refs:
        sub = sub[sub["env_step"] <= ABLATION_BUDGET]
        _plot(ax_r, sub, "return_normalized", colour, label)
        cq = _curve(sub, "mean_q")
        if not cq.empty:
            ax_q.plot(cq["env_step"], cq["mean"].clip(lower=1e-1), color=colour, linewidth=1.8)
    for setting, (label, colour) in ABLATIONS.items():
        if setting == "RLPD":
            continue  # handled by refs above
        sub = h[h["setting"] == setting]
        _plot(ax_r, sub, "return_normalized", colour, label)
        cq = _curve(sub, "mean_q")
        if not cq.empty:
            ax_q.plot(cq["env_step"], cq["mean"].clip(lower=1e-1), color=colour,
                      linewidth=1.8, label=label)
    ax_r.set_title("normalized return", fontsize=10, color="#111827")
    ax_r.set_ylabel("normalized return", fontsize=9)
    ax_r.set_ylim(bottom=0)
    ax_q.set_title("mean Q (log scale) — no-LayerNorm explodes", fontsize=10, color="#111827")
    ax_q.set_ylabel("mean Q", fontsize=9)
    ax_q.set_yscale("log")
    for ax in (ax_r, ax_q):
        ax.set_xlabel("environment steps", fontsize=9)
        _style(ax)
    ax_r.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, outdir, "ablations.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    p.add_argument("--assets", type=Path, default=ASSETS_DIR)
    args = p.parse_args()

    df = load_runs(args.results)
    if df.empty:
        print(f"no eval CSVs in {args.results}/ yet — run a training job first.")
        return

    figures = (returns_figure, mean_q_figure, quality_figure, humanoid_figure, ablations_figure)
    for fn in figures:
        print(f"wrote {fn(df, args.assets)}")


if __name__ == "__main__":
    main()
