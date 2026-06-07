"""Scaling-plot rendering for MPM-CudaJax benchmark sweeps.

Both sweep groups (``sweep_particle_count`` and ``sweep_weak_scaling``) sweep
particle count on x, so the only difference between them is the title. We
hardcode the four plots we want — substep wall time, throughput, substep
speedup vs jax, and throughput improvement vs jax — and call a single
``_plot`` helper for each. No tag/spec dispatch dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
plt.switch_backend("Agg")

# Each sweep tag's (x column, x-axis label, fixed-axis suffix). The suffix is
# appended to every plot title to disambiguate which axis is held constant.
GROUP_SPECS: dict[str, tuple[str, str, str]] = {
    "sweep_particle_count": ("n_particles", "particles", "fixed G"),
    "sweep_weak_scaling": ("n_particles", "particles", "fixed PPC"),
    "sweep_particle_density": ("num_grids", "grid resolution G", "fixed N"),
}


def load_jobs(sweep_root: Path) -> pd.DataFrame:
    files = sorted(sweep_root.rglob("results.json"))
    if not files:
        raise FileNotFoundError(f"No results.json under {sweep_root}")
    return pd.DataFrame.from_records(json.loads(p.read_text()) for p in files)


def load_parquet(*paths: Path) -> pd.DataFrame:
    if not paths:
        raise ValueError("load_parquet needs at least one path")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def _kmg(value: float, _pos) -> str:
    if value <= 0:
        return ""
    if value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if value >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:g}"


def _plot(
    df: pd.DataFrame,
    *,
    x: str,
    xlabel: str,
    y: str,
    ylabel: str,
    title: str,
    subtitle: str,
    out_path: Path,
    y_log: bool = True,
    baseline_line: bool = False,
) -> None:
    """One sweep lineplot."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    sns.lineplot(data=df, x=x, y=y, hue="kernel", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    # Tick at every measured point so the axis matches the data exactly.
    xs = sorted(df[x].dropna().unique())
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_kmg(v, None) if x == "n_particles" else f"{int(round(v))}" for v in xs]
    )
    if y_log:
        ax.set_yscale("log")
    if baseline_line:
        ax.axhline(1.0, ls="--", lw=0.8, color="0.5")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("backend")
    fig.suptitle(title, fontweight="bold")
    ax.set_title(subtitle, style="italic", color="0.35")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_sweep(
    df: pd.DataFrame,
    *,
    x: str,
    xlabel: str,
    fixed: str,
    out_dir: Path,
    baseline: str,
    gpu_label: str,
) -> None:
    """Render the 3 plots for one sweep group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" ({fixed})"

    _plot(
        df, x=x, xlabel=xlabel,
        y="ms_per_step", ylabel="ms / substep",
        title="Time per step" + suffix,
        subtitle=gpu_label,
        out_path=out_dir / "ms_per_substep.png",
    )
    _plot(
        df, x=x, xlabel=xlabel,
        y="particles_per_sec", ylabel="particles / second",
        title="Particle Throughput" + suffix,
        subtitle=gpu_label,
        out_path=out_dir / "particles_per_sec.png",
    )

    base = (
        df[df["kernel"] == baseline][[x, "ms_per_step"]]
        .rename(columns={"ms_per_step": "_base_ms"})
    )
    if base.empty:
        logger.warning("No %s baseline rows; skipping speedup plot", baseline)
        return
    merged = df.merge(base, on=x, how="inner")
    merged["speedup_substep"] = merged["_base_ms"] / merged["ms_per_step"]
    _plot(
        merged, x=x, xlabel=xlabel,
        y="speedup_substep",
        ylabel=f"speedup vs {baseline}",
        title=f"Speedup over {baseline.capitalize()} Baseline" + suffix,
        subtitle=gpu_label,
        out_path=out_dir / f"speedup_vs_{baseline}.png",
        y_log=False, baseline_line=True,
    )


def render(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    baseline: str = "jax",
    gpu_subdirs: bool = True,
) -> None:
    """Render every sweep group present in ``df``, per gpu_kind.

    Layout: ``<out_dir>/<gpu_kind>/<group>/<plot>.png``. Set
    ``gpu_subdirs=False`` when ``out_dir`` is already gpu-specific."""
    sns.set_theme(style="darkgrid", context="paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = (
        list(df.groupby("gpu_kind", dropna=False))
        if "gpu_kind" in df
        else [("gpu", df)]
    )
    for gpu_kind, gpu_df in groups:
        gpu_label = str(gpu_kind) if gpu_kind is not None else "gpu"
        gpu_dir = (out_dir / gpu_label) if gpu_subdirs else out_dir
        for tag, (x, xlabel, fixed) in GROUP_SPECS.items():
            sub = gpu_df[gpu_df["tag"] == tag] if "tag" in gpu_df else pd.DataFrame()
            if sub.empty:
                continue
            _plot_sweep(
                sub,
                x=x,
                xlabel=xlabel,
                fixed=fixed,
                out_dir=gpu_dir / tag.removeprefix("sweep_"),
                baseline=baseline,
                gpu_label=gpu_label,
            )
