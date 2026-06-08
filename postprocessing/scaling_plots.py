"""Hydra-zen plot configs for scaling sweeps.

``render`` aggregates per-job results into ``results.parquet`` and calls each
configured plot partial.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from hydra.utils import instantiate
from hydra_zen import store
from omegaconf import MISSING

logger = logging.getLogger(__name__)
plt.switch_backend("Agg")


def _kmg(value: float) -> str:
    if value <= 0:
        return ""
    if value >= 1e6:
        return f"{value / 1e6:.3g}M"
    if value >= 1e3:
        return f"{value / 1e3:.3g}k"
    return f"{value:g}"


def _finalize(fig, ax, df, *, x, xlabel, ylabel, title, subtitle, out_path):
    ax.set_xscale("log", base=2)
    xs = sorted(df[x].dropna().unique())
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_kmg(v) if x == "n_particles" else f"{int(round(v))}" for v in xs]
    )
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


@store(
    name="loglog", group="plot", zen_partial=True,
    x=MISSING, xlabel=MISSING, y=MISSING, ylabel=MISSING,
    title=MISSING, filename=MISSING,
)
def loglog_plot(
    df, *, x, xlabel, y, ylabel, title, filename, gpu_label, out_dir, **_
):
    """Log-log lineplot for substep wall time and throughput."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    sns.lineplot(data=df, x=x, y=y, hue="kernel", marker="o", ax=ax)
    ax.set_yscale("log")
    _finalize(fig, ax, df, x=x, xlabel=xlabel, ylabel=ylabel,
              title=title, subtitle=gpu_label, out_path=out_dir / filename)


@store(
    name="speedup", group="plot", zen_partial=True,
    x=MISSING, xlabel=MISSING, title=MISSING, filename=MISSING,
    baseline="jax",
)
def speedup_plot(
    df, *, x, xlabel, baseline, title, filename, gpu_label, out_dir, **_
):
    """Semilog (log x, linear y) speedup-vs-baseline plot."""
    base = (
        df[df["kernel"] == baseline][[x, "ms_per_step"]]
        .rename(columns={"ms_per_step": "_base_ms"})
    )
    if base.empty:
        logger.warning("No %s baseline rows; skipping %s", baseline, filename)
        return
    merged = df.merge(base, on=x, how="inner")
    merged["speedup"] = merged["_base_ms"] / merged["ms_per_step"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    sns.lineplot(data=merged, x=x, y="speedup", hue="kernel", marker="o", ax=ax)
    ax.axhline(1.0, ls="--", lw=0.8, color="0.5")
    _finalize(fig, ax, merged, x=x, xlabel=xlabel,
              ylabel=f"speedup vs {baseline}",
              title=title, subtitle=gpu_label, out_path=out_dir / filename)


def _aggregate_results(sweep_root: Path) -> pd.DataFrame:
    files = sorted(sweep_root.rglob("results.json"))
    if not files:
        return pd.DataFrame()
    return pd.DataFrame.from_records(json.loads(p.read_text()) for p in files)


def render(sweep_root: Path, plots: Mapping[str, Any]) -> None:
    """Aggregate sweep results and render the configured plots."""
    sweep_root = Path(sweep_root)
    df = _aggregate_results(sweep_root)
    if df.empty:
        logger.warning("No results.json under %s", sweep_root)
        return
    df.to_parquet(sweep_root / "results.parquet", index=False)
    logger.info("Wrote %s with %d rows", sweep_root / "results.parquet", len(df))

    sns.set_theme(style="darkgrid", context="paper")
    gpu_label = str(df["gpu_kind"].iloc[0]) if "gpu_kind" in df else "gpu"

    tag = str(df["tag"].iloc[0]) if "tag" in df else "plots"
    out_dir = sweep_root / tag.removeprefix("sweep_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, plot_cfg in plots.items():
        try:
            partial = instantiate(plot_cfg)
            partial(df=df, gpu_label=gpu_label, out_dir=out_dir)
        except Exception as exc:
            logger.warning("Plot '%s' failed: %s", name, exc, exc_info=True)
