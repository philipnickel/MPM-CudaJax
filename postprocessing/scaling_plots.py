"""Scaling-plot rendering for MPM-CudaJax benchmark sweeps.

The multirun callback writes ONE parquet at the sweep root (the combined
multirun dataframe). This module loads that parquet and renders ms/step,
throughput, and speedup-vs-baseline plots — one figure per sweep tag,
faceted by ``gpu_kind`` when more than one is present.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
plt.switch_backend("Agg")

# tag -> (x column, x-axis label, plot title)
SWEEP_SPECS: dict[str, tuple[str, str, str]] = {
    "sweep_particle_count": (
        "n_particles", "particles", "Particle-count scaling at G=96",
    ),
    "sweep_particle_density": (
        "num_grids", "grid resolution G (N=10M)", "Particle-density scaling",
    ),
    "sweep_weak_scaling": (
        "n_particles", "particles (constant active PPC)", "Weak scaling",
    ),
    "sweep_all": (
        "n_particles", "particles", "Backend comparison at benchmark",
    ),
}


def load_jobs(sweep_root: Path) -> pd.DataFrame:
    """Build the combined multirun dataframe from per-job ``results.json``."""
    files = sorted(sweep_root.rglob("results.json"))
    if not files:
        raise FileNotFoundError(f"No results.json under {sweep_root}")
    return pd.DataFrame.from_records(json.loads(p.read_text()) for p in files)


def load_parquet(*paths: Path) -> pd.DataFrame:
    """Concatenate one or more multirun parquets into a single dataframe."""
    if not paths:
        raise ValueError("load_parquet needs at least one path")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _sci(value: float, _pos) -> str:
    """Classical scientific notation: 1 × 10^n, rendered via mathtext."""
    if value <= 0:
        return ""
    exp = int(math.floor(math.log10(value)))
    mant = value / 10**exp
    if abs(mant - 1.0) < 1e-9:
        return rf"$10^{{{exp}}}$"
    return rf"${mant:.1f}\times 10^{{{exp}}}$"


def _set_title(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    """Main title above the figure + italic subtitle on the axes, both centered."""
    fig.suptitle(title, fontweight="bold")
    ax.set_title(subtitle, style="italic", color="0.35")


def _lineplot(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str,
    path: Path,
    y_log: bool = True,
    baseline_line: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    sns.lineplot(data=df, x=x, y=y, hue="kernel", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    if x == "n_particles":
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_sci))
    if y_log:
        ax.set_yscale("log")
    if baseline_line:
        ax.axhline(1.0, ls="--", lw=0.8, color="0.5")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("backend")
    _set_title(fig, ax, title, subtitle)
    _save(fig, path)


def _plot_tag(
    df: pd.DataFrame, tag: str, out_dir: Path, baseline: str, gpu_label: str
) -> None:
    x, xlabel, title = SWEEP_SPECS[tag]
    sub = df[df["tag"] == tag].copy() if "tag" in df else df.copy()
    if sub.empty or x not in sub:
        return

    _lineplot(
        sub, x=x, y="ms_per_step",
        xlabel=xlabel, ylabel="ms / step",
        title=title, subtitle=gpu_label,
        path=out_dir / f"{tag}_ms_per_step.png",
    )
    _lineplot(
        sub, x=x, y="particles_per_sec",
        xlabel=xlabel, ylabel="particles / second",
        title=f"{title} (throughput)", subtitle=gpu_label,
        path=out_dir / f"{tag}_particles_per_sec.png",
    )

    base = (
        sub[sub["kernel"] == baseline][[x, "ms_per_step"]]
        .rename(columns={"ms_per_step": "_base_ms"})
    )
    if base.empty:
        logger.warning("No %s baseline rows for tag=%s; skipping speedup", baseline, tag)
        return
    speedup = sub.merge(base, on=x, how="inner")
    speedup["speedup"] = speedup["_base_ms"] / speedup["ms_per_step"]
    _lineplot(
        speedup, x=x, y="speedup",
        xlabel=xlabel, ylabel=f"speedup vs {baseline}",
        title=f"{title} (speedup)", subtitle=gpu_label,
        path=out_dir / f"{tag}_speedup_vs_{baseline}.png",
        y_log=False, baseline_line=True,
    )


def render(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    baseline: str = "jax",
    gpu_subdirs: bool = True,
) -> list[tuple[str, str]]:
    """Render every known sweep tag present in ``df``, per gpu_kind.

    If ``gpu_subdirs`` is True (the CLI default), each gpu_kind's plots
    land in ``out_dir/<gpu_kind>/``. Set to False when ``out_dir`` is
    already gpu-specific (the callback's case)."""
    sns.set_theme(style="darkgrid", context="paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = (
        list(df.groupby("gpu_kind", dropna=False))
        if "gpu_kind" in df
        else [("gpu", df)]
    )
    rendered: list[tuple[str, str]] = []
    for gpu_kind, gpu_df in groups:
        gpu_label = str(gpu_kind) if gpu_kind is not None else "gpu"
        target = (out_dir / gpu_label) if gpu_subdirs else out_dir
        target.mkdir(parents=True, exist_ok=True)
        present = set(gpu_df["tag"].dropna().unique()) if "tag" in gpu_df else set()
        for tag in SWEEP_SPECS:
            if tag in present:
                _plot_tag(gpu_df, tag, target, baseline, gpu_label)
                rendered.append((gpu_label, tag))
    if not rendered:
        raise RuntimeError("No known sweep tags in the data — nothing to plot.")
    return rendered
