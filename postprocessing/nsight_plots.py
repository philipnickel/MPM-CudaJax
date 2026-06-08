"""Render NCU analysis figures from profile_nsight.py sweep parquet files."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Optimization-path order.
BACKEND_ORDER = ["cuda_v1", "cuda_v2", "cuda_v3", "CuTile"]
COLOR = {
    "cuda_v1": "#fdae6b",
    "cuda_v2": "#fd8d3c",
    "cuda_v3": "#a63603",
    "CuTile": "#2171b5",
}
BACKEND_MARKER = {
    "cuda_v1": "o",
    "cuda_v2": "s",
    "cuda_v3": "^",
    "CuTile": "X",
}
LIMITER_NAME = {0: "registers", 1: "shared_mem", 2: "warps", 3: "blocks"}
# Stall reasons: memory, compute, latency.
STALLS = [
    ("long_scoreboard", "#d62728"),
    ("lg_throttle", "#ff7f0e"),
    ("short_scoreboard", "#1f77b4"),
    ("mio_throttle", "#17becf"),
    ("barrier", "#9467bd"),
    ("math_pipe_throttle", "#7f7f7f"),
    ("wait", "#bcbd22"),
    ("not_selected", "#2ca02c"),
]
DEFAULT_PLOTS = ("roofline", "atomics", "occupancy", "scheduler", "roofline_scaling")
MEMORY_LEVELS = (("l1", "o"), ("l2", "s"), ("hbm", "^"))
MEMORY_COLOR = {
    "l1": "#4c78a8",
    "l2": "#f58518",
    "hbm": "#54a24b",
}


_NSIGHT_VALUE_COLUMNS = {
    "Metric",
    "AvgValue",
    "StdDev",
    "MinValue",
    "MaxValue",
    "NumRuns",
    "CI95_Lower",
    "CI95_Upper",
    "RelativeStdDevPct",
    "StableMeasurement",
    "Normalized",
    "Geomean",
    "Value",
}


def _read_dataframe(path):
    if path.is_dir() or path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _wide_metric_rows(df):
    """nsight-python metric rows -> one wide row per profiled config/kernel."""
    if not {"Metric", "AvgValue"}.issubset(df.columns):
        return df
    index_cols = [c for c in df.columns if c not in _NSIGHT_VALUE_COLUMNS]
    # Shield NA index groups from pivot_table.
    null_sentinel = "__MPM_CUDAJAX_NULL_INDEX__"
    work = df.copy()
    for col in index_cols:
        mask = work[col].isna()
        if mask.any():
            work[col] = work[col].astype("object")
            work.loc[mask, col] = null_sentinel
    wide = (
        work.pivot_table(
            index=index_cols,
            columns="Metric",
            values="AvgValue",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )
    wide.columns.name = None
    for col in index_cols:
        if col in wide.columns:
            wide[col] = wide[col].replace(null_sentinel, pd.NA)
    return wide


def _col(df, name, default=0.0):
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _first_col(df, names, default=0.0):
    for name in names:
        if name in df:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _has_any_value(series):
    return series.notna().any() and (series.fillna(0) != 0).any()


def _prefer_nonzero(primary, fallback):
    primary = pd.to_numeric(primary, errors="coerce")
    fallback = pd.to_numeric(fallback, errors="coerce")
    return primary.where(primary.fillna(0) != 0, fallback)


def _derive_analysis_columns(df):
    """Add plotting columns from raw NCU counters. Existing derived columns win."""
    df = df.copy()
    n = _col(df, "n_particles")
    gpu_seconds = _col(df, "gpu__time_duration.sum") / 1e9
    sm_seconds = _col(df, "sm__cycles_elapsed.avg") / _col(
        df, "sm__cycles_elapsed.avg.per_second"
    )
    seconds = _prefer_nonzero(gpu_seconds, sm_seconds)
    if "time_ms" not in df:
        df["time_ms"] = seconds * 1e3
    if "p2g_mparticles_per_s" not in df:
        df["p2g_mparticles_per_s"] = (n / seconds) / 1e6

    fadd = _first_col(
        df,
        [
            "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        ],
    )
    fmul = _first_col(
        df,
        [
            "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        ],
    )
    ffma = _first_col(
        df,
        [
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        ],
    )
    flops = fadd + fmul + 2.0 * ffma
    if "fp32_flops" not in df:
        df["fp32_flops"] = flops
    if "gflops_per_s" not in df:
        df["gflops_per_s"] = flops / seconds / 1e9

    bytes_by_level = {
        "l1": _prefer_nonzero(
            _col(df, "l1tex__t_bytes.sum"),
            _col(df, "l1tex__lsu_writeback_active_mem_lg.sum") * 128.0,
        ),
        "l2": _prefer_nonzero(
            _col(df, "lts__t_bytes.sum"),
            _col(df, "lts__lts2xbar_cycles_active.sum") * 32.0,
        ),
        "hbm": _col(df, "dram__bytes.sum"),
    }
    for level, nbytes in bytes_by_level.items():
        if f"bytes_{level}" not in df:
            df[f"bytes_{level}"] = nbytes
        if f"ai_{level}_flop_per_byte" not in df:
            df[f"ai_{level}_flop_per_byte"] = flops / nbytes

    ffma_peak = _col(df, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")
    sm_hz = _col(df, "sm__cycles_elapsed.avg.per_second")
    if "peak_compute_gflops" not in df:
        df["peak_compute_gflops"] = ffma_peak * 2.0 * sm_hz / 1e9
    for level, peak_name, fallback_name, clock_name, fallback_width in (
        (
            "l1",
            "l1tex__t_bytes.sum.peak_sustained",
            "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained",
            "l1tex__cycles_elapsed.avg.per_second",
            128.0,
        ),
        (
            "l2",
            "lts__t_bytes.sum.peak_sustained",
            "lts__lts2xbar_cycles_active.sum.peak_sustained",
            "lts__cycles_elapsed.avg.per_second",
            32.0,
        ),
        (
            "hbm",
            "dram__bytes.sum.peak_sustained",
            None,
            "dram__cycles_elapsed.avg.per_second",
            1.0,
        ),
    ):
        if f"peak_{level}_gbps" not in df:
            peak_per_cycle = _col(df, peak_name)
            if not _has_any_value(peak_per_cycle) and fallback_name is not None:
                peak_per_cycle = _col(df, fallback_name) * fallback_width
            df[f"peak_{level}_gbps"] = (
                peak_per_cycle * _col(df, clock_name) / 1e9
            )

    for metric, column in [
        ("dram__bytes.sum", "dram_bytes_per_particle"),
        ("dram__bytes_read.sum", "dram_read_bytes_per_particle"),
        ("dram__bytes_write.sum", "dram_write_bytes_per_particle"),
    ]:
        if column not in df:
            df[column] = _col(df, metric) / n

    for op in ("ld", "st", "atom", "red"):
        req = _col(df, f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum")
        sec = _col(df, f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum")
        if f"global_{op}_requests_per_particle" not in df:
            df[f"global_{op}_requests_per_particle"] = req / n
        if f"global_{op}_sectors_per_particle" not in df:
            df[f"global_{op}_sectors_per_particle"] = sec / n
        if f"global_{op}_sectors_per_request" not in df:
            df[f"global_{op}_sectors_per_request"] = sec / req

    scatter_req = sum(
        _col(df, f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum")
        for op in ("red", "atom", "st")
    )
    scatter_sec = sum(
        _col(df, f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum")
        for op in ("red", "atom", "st")
    )
    if "global_scatter_requests_per_particle" not in df:
        df["global_scatter_requests_per_particle"] = scatter_req / n
    if "global_scatter_sectors_per_particle" not in df:
        df["global_scatter_sectors_per_particle"] = scatter_sec / n
    if "global_scatter_sectors_per_request" not in df:
        df["global_scatter_sectors_per_request"] = scatter_sec / scatter_req

    for metric, column in [
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sol_sm_pct"),
        (
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sol_compute_memory_pct",
        ),
        (
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "sol_dram_pct",
        ),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "active_warps_pct"),
        ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue_active_pct"),
        ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "sol_l2_pct"),
        (
            "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
            "sol_l1_pct",
        ),
        ("lts__t_sector_hit_rate.pct", "l2_hit_rate_pct"),
        ("l1tex__t_sector_hit_rate.pct", "l1_hit_rate_pct"),
        (
            "lts__t_sectors_op_red.sum.pct_of_peak_sustained_elapsed",
            "l2_red_throughput_pct",
        ),
        (
            "lts__d_atomic_input_cycles_active.sum.pct_of_peak_sustained_elapsed",
            "l2_atomic_unit_pct",
        ),
        (
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red.ratio",
            "scatter_sectors_per_request",
        ),
        (
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            "load_sectors_per_request",
        ),
        ("launch__registers_per_thread", "regs_per_thread"),
        ("launch__shared_mem_per_block", "smem_bytes_per_block"),
    ]:
        if column not in df:
            fallbacks = {
                "sol_dram_pct": [
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                ],
                "sol_l1_pct": [
                    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
                    "l1tex__throughput.avg.pct_of_peak_sustained_active",
                ],
                "smem_bytes_per_block": [
                    "launch__shared_mem_per_block",
                    "launch__shared_mem_per_block_static",
                ],
            }
            df[column] = _first_col(df, fallbacks.get(column, [metric]))

    if "sol_max_pct" not in df:
        df["sol_max_pct"] = df[
            ["sol_sm_pct", "sol_compute_memory_pct", "sol_dram_pct"]
        ].max(axis=1)

    if "theoretical_occ_pct" not in df:
        df["theoretical_occ_pct"] = _col(df, "sm__maximum_warps_per_active_cycle_pct")

    if "occ_limiter_code" not in df:
        limits = pd.DataFrame(
            {
                "registers": _col(df, "launch__occupancy_limit_registers"),
                "shared_mem": _col(df, "launch__occupancy_limit_shared_mem"),
                "warps": _col(df, "launch__occupancy_limit_warps"),
                "blocks": _col(df, "launch__occupancy_limit_blocks"),
            }
        )
        max_warps = _col(df, "sm__maximum_warps_avg_per_active_cycle", 64.0)
        binder = limits.idxmin(axis=1)
        if not _has_any_value(df["theoretical_occ_pct"]):
            df["theoretical_occ_pct"] = 100.0 * limits.min(axis=1) / max_warps
        df["occ_limiter_code"] = binder.map(
            {"registers": 0, "shared_mem": 1, "warps": 2, "blocks": 3}
        )

    stall_total = pd.Series(0.0, index=df.index)
    for reason, _color in STALLS:
        metric = f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio"
        col = f"stallr_{reason}"
        if col not in df:
            df[col] = _col(df, metric)
        stall_total = stall_total + _col(df, col)
    if "stallr_total" not in df:
        df["stallr_total"] = stall_total
    return df.replace([float("inf"), -float("inf")], pd.NA)


def load_dataframe(path):
    """Read ProfileResults.to_dataframe() outputs, with old CSV fallback."""
    return _derive_analysis_columns(_wide_metric_rows(_read_dataframe(path)))


def _set_style(style=None):
    style = dict(style or {})
    theme = style.pop("theme", style.pop("style", "darkgrid"))
    context = style.pop("context", "paper")
    sns.set_theme(style=theme, context=context, **style)


def _gpu_label(df, results_path=None):
    for column in ("gpu_kind", "gpu_type"):
        if column not in df:
            continue
        values = df[column].dropna().astype(str)
        values = values[values != ""]
        if not values.empty:
            return values.iloc[0]
    if results_path is not None:
        parts = Path(results_path).parts
        if "nsight" in parts:
            index = parts.index("nsight") + 1
            if index < len(parts):
                return parts[index]
    return "gpu"


def _set_figure_title(fig, ax, title, gpu_label=None, *, subtitle=None):
    fig.suptitle(title, fontweight="bold")
    subtitle_lines = [line for line in (gpu_label, subtitle) if line]
    if subtitle_lines:
        ax.set_title(
            "\n".join(str(line) for line in subtitle_lines),
            style="italic",
            color="0.35",
            fontsize=9,
        )


def _set_multi_panel_title(fig, title, gpu_label=None):
    fig.suptitle(title, fontweight="bold")
    if gpu_label:
        fig.text(
            0.5,
            0.91,
            str(gpu_label),
            ha="center",
            va="top",
            style="italic",
            color="0.35",
            fontsize=9,
        )


def _resolve_scale_axis(df, scale_axis="auto"):
    if scale_axis not in (None, "auto"):
        labels = {
            "n_particles": "particle count",
            "num_grids": "grid size G",
            "mps_thread_percent": "CUDA MPS active-thread %",
        }
        return scale_axis, labels.get(scale_axis, scale_axis)

    n_var = "n_particles" in df.columns and df["n_particles"].nunique() > 1
    g_var = "num_grids" in df.columns and df["num_grids"].nunique() > 1
    mps_var = (
        "mps_thread_percent" in df.columns
        and df["mps_thread_percent"].nunique() > 1
    )
    if mps_var:
        return "mps_thread_percent", "CUDA MPS active-thread %"
    if n_var and g_var:
        return "num_grids", "grid size G (weak scaling, const ppc)"
    if n_var:
        return "n_particles", "particle count (fixed grid - load sweep)"
    if g_var:
        return "num_grids", "grid size G (density sweep)"
    return None, None


def roofline_long_dataframe(df, scale_col=None):
    """Wide roofline columns -> tidy rows for seaborn semantic plotting."""
    keep = []
    for col in ("backend", "n_particles", "num_grids", "gflops_per_s", scale_col):
        if col and col in df.columns and col not in keep:
            keep.append(col)
    rows = []
    for level, _marker in MEMORY_LEVELS:
        ai_col = f"ai_{level}_flop_per_byte"
        if ai_col not in df.columns:
            continue
        sub = df[keep + [ai_col]].copy()
        sub["memory_level"] = level
        sub["ai_flop_per_byte"] = sub[ai_col]
        rows.append(sub.drop(columns=[ai_col]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _has_positive(df, columns):
    for column in columns:
        if column in df and (pd.to_numeric(df[column], errors="coerce") > 0).any():
            return True
    return False


def _has_roofline_data(df):
    return _has_positive(
        df,
        [
            "gflops_per_s",
            "peak_compute_gflops",
            "ai_l1_flop_per_byte",
            "ai_l2_flop_per_byte",
            "ai_hbm_flop_per_byte",
        ],
    ) and _has_positive(df, ["ai_hbm_flop_per_byte"])


def _has_atomics_data(df):
    return _has_positive(
        df,
        [
            "global_scatter_requests_per_particle",
            "global_scatter_sectors_per_particle",
            "stallr_lg_throttle",
            "stallr_mio_throttle",
        ],
    )


def _has_occupancy_data(df):
    return _has_positive(df, ["theoretical_occ_pct", "active_warps_pct"])


def _has_scheduler_data(df):
    return _has_positive(
        df,
        ["issue_active_pct", *(f"stallr_{reason}" for reason, _color in STALLS)],
    )


def table_from_dataframe(df):
    """Wide DataFrame -> {backend: {metric: value}}, dropping NaN cells.

    Scaling sweeps use the largest row per backend as the representative point."""
    sort_keys = [
        c for c in ("n_particles", "num_grids", "mps_thread_percent") if c in df.columns
    ]
    if sort_keys:
        df = df.copy()  # defragment before groupby
        df = df.sort_values(sort_keys).groupby("backend", as_index=False).last()
    table = {}
    for _, row in df.iterrows():
        backend = row.get("backend")
        if backend is None or (isinstance(backend, float) and pd.isna(backend)):
            continue
        table[str(backend)] = {k: v for k, v in row.items() if pd.notna(v)}
    return table


def _ordered(table):
    return [b for b in BACKEND_ORDER if b in table] or list(table)


def _first(table, backends, key):
    for b in backends:
        v = table[b].get(key)
        if v is not None:
            return v
    return None


def _annotate_roofs(ax, ai, peak_c, roof_specs):
    ax.figure.canvas.draw()
    for lvl, bw, color in roof_specs:
        if not bw or not peak_c:
            continue
        x0 = ai[0]
        x1 = min(peak_c / bw, ai[-1])
        x = math.sqrt(x0 * x1)
        y = min(bw * x, peak_c)
        p0 = ax.transData.transform((x, y))
        p1 = ax.transData.transform((x * 1.3, min(bw * x * 1.3, peak_c)))
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        offset_px = 8.0
        p_text = (p0[0] - dy / norm * offset_px, p0[1] + dx / norm * offset_px)
        x_text, y_text = ax.transData.inverted().transform(p_text)
        ax.text(
            x_text,
            y_text,
            f"{lvl.upper()} {bw / 1000:.1f} TB/s",
            fontsize=7,
            color=color,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
        )


def plot_roofline(table, out, gpu_label=None):
    import numpy as np

    backends = _ordered(table)
    peak_c = _first(table, backends, "peak_compute_gflops")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ai = np.logspace(-2, 3, 300)
    roof_specs = []
    for lvl, _marker in MEMORY_LEVELS:
        bw = _first(table, backends, f"peak_{lvl}_gbps")
        if bw and peak_c:
            color = MEMORY_COLOR[lvl]
            ax.plot(
                ai,
                np.minimum(bw * ai, peak_c),
                color=color,
                lw=1.0,
                alpha=0.35,
                label="_nolegend_",
            )
            roof_specs.append((lvl, bw, color))
    if peak_c:
        ax.axhline(peak_c, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.text(
            ai[-1], peak_c * 1.05, f"fp32 compute {peak_c / 1000:.1f} TFLOP/s",
            ha="right", va="bottom", fontsize=8, color="0.25",
        )
    points = []
    for b in backends:
        g = table[b].get("gflops_per_s")
        for order, (lvl, _mark) in enumerate(MEMORY_LEVELS):
            x = table[b].get(f"ai_{lvl}_flop_per_byte")
            if x and g:
                points.append(
                    {
                        "backend": b,
                        "memory_level": lvl,
                        "memory_order": order,
                        "ai_flop_per_byte": x,
                        "gflops_per_s": g,
                    }
                )
    if points:
        point_df = pd.DataFrame(points).sort_values(
            ["backend", "ai_flop_per_byte", "memory_order"]
        )
        sns.lineplot(
            data=point_df,
            x="ai_flop_per_byte",
            y="gflops_per_s",
            hue="memory_level",
            style="backend",
            palette=MEMORY_COLOR,
            markers=BACKEND_MARKER,
            dashes=False,
            estimator=None,
            errorbar=None,
            sort=False,
            markeredgecolor="white",
            markeredgewidth=0.4,
            ax=ax,
            zorder=3,
        )
        sns.move_legend(
            ax,
            "best",
            frameon=True,
            title=None,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    _set_figure_title(fig, ax, "Hierarchical fp32 roofline — P2G scatter", gpu_label)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _annotate_roofs(ax, ai, peak_c, roof_specs)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_roofline_scaling(
    df, out, scale_col="n_particles", scale_label=None, gpu_label=None
):
    """fp32 roofline trajectory over the selected scaling axis."""
    import numpy as np

    # Median over noisy rows; ceilings are constant.
    peak_c = float(df["peak_compute_gflops"].median())
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ai = np.logspace(-2, 3, 300)
    roof_specs = []
    for lvl, _marker in MEMORY_LEVELS:
        col = f"peak_{lvl}_gbps"
        if col not in df:
            continue
        bw = float(df[col].median())
        color = MEMORY_COLOR[lvl]
        ax.plot(
            ai,
            np.minimum(bw * ai, peak_c),
            color=color,
            lw=1.0,
            alpha=0.35,
            label="_nolegend_",
        )
        roof_specs.append((lvl, bw, color))
    ax.axhline(peak_c, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.text(ai[-1], peak_c * 1.05, f"fp32 compute {peak_c / 1000:.1f} TFLOP/s",
            ha="right", va="bottom", fontsize=8, color="0.25")

    roof_long = roofline_long_dataframe(df, scale_col).sort_values(
        ["backend", "memory_level", scale_col]
    )
    if not roof_long.empty:
        sns.lineplot(
            data=roof_long,
            x="ai_flop_per_byte",
            y="gflops_per_s",
            hue="memory_level",
            style="backend",
            palette=MEMORY_COLOR,
            markers=BACKEND_MARKER,
            dashes=False,
            estimator=None,
            errorbar=None,
            sort=False,
            ax=ax,
        )

    if not roof_long.empty:
        sns.move_legend(
            ax,
            "best",
            frameon=True,
            title=None,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    _set_figure_title(
        fig,
        ax,
        "Hierarchical fp32 roofline trajectory — P2G scatter",
        gpu_label,
    )
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _annotate_roofs(ax, ai, peak_c, roof_specs)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_atomics(table, out, gpu_label=None):
    backends = _ordered(table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _set_multi_panel_title(fig, "Atomic-scatter analysis — P2G scatter", gpu_label)
    scatter = pd.DataFrame(
        {
            "backend": backends,
            "global scatter requests / particle": [
                table[b].get("global_scatter_requests_per_particle", 0)
                for b in backends
            ],
        }
    )
    sns.barplot(
        data=scatter,
        x="backend",
        y="global scatter requests / particle",
        hue="backend",
        palette=COLOR,
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(108.0, ls="--", color="0.4", lw=0.8)
    axes[0].text(0, 108.0, "108 (uncoalesced)", fontsize=7, va="bottom", color="0.3")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_title("Atomic-scatter volume", fontsize=10)

    stalls = pd.DataFrame(
        [
            {"backend": b, "reason": "lg_throttle (global)", "value": table[b].get("stallr_lg_throttle", 0)}
            for b in backends
        ]
        + [
            {"backend": b, "reason": "mio_throttle (shared)", "value": table[b].get("stallr_mio_throttle", 0)}
            for b in backends
        ]
    )
    sns.barplot(
        data=stalls,
        x="backend",
        y="value",
        hue="reason",
        palette=["#ff7f0e", "#17becf"],
        ax=axes[1],
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("warps stalled / issue-active cycle")
    axes[1].set_title("Contention fingerprint: global -> shared", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_occupancy(table, out, gpu_label=None):
    backends = _ordered(table)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    occ = pd.DataFrame(
        [
            {"backend": b, "occupancy": "theoretical", "pct": table[b].get("theoretical_occ_pct", 0)}
            for b in backends
        ]
        + [
            {"backend": b, "occupancy": "achieved", "pct": table[b].get("active_warps_pct", 0)}
            for b in backends
        ]
    )
    sns.barplot(
        data=occ,
        x="backend",
        y="pct",
        hue="occupancy",
        palette={"theoretical": "0.8", "achieved": "#2171b5"},
        ax=ax,
    )
    for i, b in enumerate(backends):
        code = table[b].get("occ_limiter_code")
        regs = table[b].get("regs_per_thread")
        if code is not None:
            lbl = LIMITER_NAME.get(int(code), "?")
            note = f"[{lbl}]" + (f"\nR={int(regs)}" if regs else "")
            top = max(
                table[b].get("theoretical_occ_pct", 0),
                table[b].get("active_warps_pct", 0),
            )
            ax.annotate(
                note,
                (i, top),
                fontsize=7,
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("occupancy (% of 64 warps/SM)")
    ax.set_ylim(0, 105)
    _set_figure_title(
        fig,
        ax,
        "Occupancy — P2G scatter",
        gpu_label,
        subtitle="achieved vs theoretical; limiter annotated",
    )
    ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=140)
    plt.close(fig)


def render_nsight_figures(
    results_path,
    out_dir,
    *,
    plots=DEFAULT_PLOTS,
    scale_axis="auto",
    style=None,
):
    """Load a sweep parquet and render the configured Nsight analysis figures."""
    _set_style(style)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(results_path)
    df = load_dataframe(results_path)
    gpu_label = _gpu_label(df, results_path)
    table = table_from_dataframe(df)
    requested = set(plots or DEFAULT_PLOTS)
    written = []

    if "roofline" in requested:
        if _has_roofline_data(df):
            path = out_dir / "roofline.png"
            plot_roofline(table, path, gpu_label=gpu_label)
            written.append(path)
        else:
            logger.warning("Skipping roofline.png; required roofline metrics are absent.")
    if "atomics" in requested:
        if _has_atomics_data(df):
            path = out_dir / "atomics.png"
            plot_atomics(table, path, gpu_label=gpu_label)
            written.append(path)
        else:
            logger.warning("Skipping atomics.png; required atomics metrics are absent.")
    if "occupancy" in requested:
        if _has_occupancy_data(df):
            path = out_dir / "occupancy.png"
            plot_occupancy(table, path, gpu_label=gpu_label)
            written.append(path)
        else:
            logger.warning("Skipping occupancy.png; required occupancy metrics are absent.")
    if "scheduler" in requested:
        if _has_scheduler_data(df):
            path = out_dir / "scheduler.png"
            plot_scheduler(table, path, gpu_label=gpu_label)
            written.append(path)
        else:
            logger.warning("Skipping scheduler.png; required scheduler metrics are absent.")
    if "roofline_scaling" in requested:
        scale_col, scale_label = _resolve_scale_axis(df, scale_axis)
        if scale_col is not None and _has_roofline_data(df):
            path = out_dir / "roofline_scaling.png"
            plot_roofline_scaling(
                df, path, scale_col, scale_label, gpu_label=gpu_label
            )
            written.append(path)
        elif scale_col is not None:
            logger.warning(
                "Skipping roofline_scaling.png; required roofline metrics are absent."
            )
    return written


def plot_scheduler(table, out, gpu_label=None):
    import numpy as np

    backends = _ordered(table)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(backends))
    left = np.zeros(len(backends))
    for reason, color in STALLS:
        vals = np.array([table[b].get(f"stallr_{reason}", 0.0) for b in backends])
        ax.barh(y, vals, left=left, color=color, label=reason)
        left += vals
    for i, b in enumerate(backends):
        issue = table[b].get("issue_active_pct")
        if issue is not None:
            ax.text(left[i], i, f"  issue {issue:.0f}%", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(backends, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("warps stalled / issue-active cycle (sum of reasons)")
    _set_figure_title(
        fig, ax, "Warp-issue-stall breakdown — P2G scatter", gpu_label
    )
    ax.legend(fontsize=6, ncol=4, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results_path",
        type=Path,
        help="Sweep results.parquet from a profile_nsight.py Hydra-multirun sweep.",
    )
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("nsight_figs"))
    args = ap.parse_args()
    written = render_nsight_figures(args.results_path, args.out_dir)
    names = "/".join(path.stem for path in written)
    print(f"wrote {names} .png to {args.out_dir}")


if __name__ == "__main__":
    main()
