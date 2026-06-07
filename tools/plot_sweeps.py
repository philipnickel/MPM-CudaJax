"""Plot benchmark sweeps from the CSV files emitted by simulate.py."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)
SWEEP_SPECS = {
    "sweep_particle_count": {
        "prefix": "particle_count",
        "x": "n_particles",
        "xlabel": "particles",
        "title": "Particle-count scaling at G=96",
    },
    "sweep_particle_density": {
        "prefix": "particle_density",
        "x": "num_grids",
        "xlabel": "grid resolution G (N=10M)",
        "title": "Particle-density / grid-resolution scaling",
    },
    "sweep_weak_scaling": {
        "prefix": "weak_scaling",
        "x": "n_particles",
        "xlabel": "particles (constant active PPC)",
        "title": "Weak scaling at constant active PPC",
    },
}
NUMERIC_COLUMNS = [
    "n_particles",
    "num_grids",
    "dx",
    "grid_cells",
    "material_volume",
    "active_grid_cells",
    "particles_per_grid_cell",
    "particles_per_active_cell",
    "num_frames",
    "steps_per_frame",
    "total_steps",
    "elapsed_s",
    "ms_per_frame",
    "ms_per_step",
    "steps_per_sec",
    "particles_per_sec",
]


def _csv_paths(root: Path):
    if root.is_file():
        return [root]
    if (root / "results.csv").exists():
        return [root / "results.csv"]
    return sorted(root.glob("*/results.csv"))


def load_results(root: Path) -> pd.DataFrame:
    paths = _csv_paths(root)
    if not paths:
        raise FileNotFoundError(f"No results.csv files found under {root}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        if "gpu_kind" not in frame:
            frame["gpu_kind"] = path.parent.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if "output_dir" in df:
        df = df.drop_duplicates("output_dir", keep="last")
    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _summary(df: pd.DataFrame, x: str) -> pd.DataFrame:
    keys = ["kernel", x]
    keep_first = [
        "num_grids",
        "n_particles",
        "active_grid_cells",
        "particles_per_active_cell",
    ]
    agg = {
        "ms_per_step": "mean",
        "particles_per_sec": "mean",
        "steps_per_sec": "mean",
        "elapsed_s": "mean",
        "output_dir": "count",
    }
    for column in keep_first:
        if column in df and column not in keys:
            agg[column] = "first"
    summary = df.groupby(keys, as_index=False).agg(agg)
    return summary.rename(columns={"output_dir": "runs"})


def _format_tick(value, x):
    value = int(value)
    if x == "num_grids":
        return str(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return str(value)


def _lineplot(summary, *, x, y, xlabel, ylabel, title, path, y_log=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=summary, x=x, y=y, hue="kernel", marker="o", ax=ax)
    ax.set_xscale("log", base=2)
    if x in {"num_grids", "grid_cells", "n_particles"}:
        ticks = sorted(summary[x].dropna().unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_tick(tick, x) for tick in ticks])
    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="backend", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _speedup(summary: pd.DataFrame, x: str, baseline: str) -> pd.DataFrame:
    base = summary[summary["kernel"] == baseline][[x, "ms_per_step"]].rename(
        columns={"ms_per_step": "baseline_ms_per_step"}
    )
    if base.empty:
        return pd.DataFrame()
    speedup = summary.merge(base, on=x, how="inner")
    speedup["speedup_vs_baseline"] = (
        speedup["baseline_ms_per_step"] / speedup["ms_per_step"]
    )
    return speedup


def plot_sweep(df, *, tag, spec, figures_dir, baseline):
    sweep_df = df[df["tag"] == tag].copy()
    if sweep_df.empty:
        logger.warning("No rows for tag=%s", tag)
        return

    x = spec["x"]
    if x not in sweep_df:
        logger.warning("Skipping tag=%s because column %s is missing", tag, x)
        return

    for gpu_kind, gpu_df in sweep_df.groupby("gpu_kind", dropna=False):
        gpu_label = str(gpu_kind)
        title_gpu = gpu_label.replace("_", " ")
        out_dir = figures_dir / gpu_label
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = _summary(gpu_df, x)
        prefix = spec["prefix"]
        summary_path = out_dir / f"{prefix}_summary.csv"
        summary.sort_values(["kernel", x]).to_csv(summary_path, index=False)
        logger.info("Wrote %s", summary_path)

        _lineplot(
            summary,
            x=x,
            y="ms_per_step",
            xlabel=spec["xlabel"],
            ylabel="ms / step",
            title=f"{spec['title']}\n{title_gpu}",
            path=out_dir / f"{prefix}_ms_per_step.png",
        )
        _lineplot(
            summary,
            x=x,
            y="particles_per_sec",
            xlabel=spec["xlabel"],
            ylabel="particles / second",
            title=f"{spec['title']} throughput\n{title_gpu}",
            path=out_dir / f"{prefix}_particles_per_sec.png",
        )

        speedup = _speedup(summary, x, baseline)
        if speedup.empty:
            logger.warning("No %s baseline rows for tag=%s", baseline, tag)
            continue
        _lineplot(
            speedup,
            x=x,
            y="speedup_vs_baseline",
            xlabel=spec["xlabel"],
            ylabel=f"speedup vs {baseline}",
            title=f"{spec['title']} speedup\n{title_gpu}",
            path=out_dir / f"{prefix}_speedup_vs_{baseline}.png",
            y_log=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/sweeps"))
    parser.add_argument("--baseline", default="jax")
    parser.add_argument("--tag", action="append", choices=sorted(SWEEP_SPECS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sns.set_theme(style="whitegrid", context="talk")

    df = load_results(args.root)
    tags = args.tag or [tag for tag in SWEEP_SPECS if tag in set(df["tag"])]
    if not tags:
        raise RuntimeError(f"No known sweep tags found in {args.root}")

    for tag in tags:
        plot_sweep(
            df,
            tag=tag,
            spec=SWEEP_SPECS[tag],
            figures_dir=args.figures_dir,
            baseline=args.baseline,
        )


if __name__ == "__main__":
    main()
