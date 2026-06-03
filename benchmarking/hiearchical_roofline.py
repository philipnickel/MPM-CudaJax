"""Collect and plot FP32 Roofline points with Nsight Python."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import nsight
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from nsight.visualization import visualize
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from profile_nsight import (  # noqa: E402
    _disable_editable_pth_for_nsight,
    _prepare_nsight_child_python,
    _run_nsight_profile,
)

WORK_METRICS = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
]
DRAM_TRAFFIC_METRIC = "dram__bytes.sum"
HIERARCHICAL_TRAFFIC_METRICS = [
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    DRAM_TRAFFIC_METRIC,
]
ORDINARY_PEAK_METRICS = [
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
]
HIERARCHICAL_PEAK_METRICS = [
    "lts__lts2xbar_cycles_active.sum.peak_sustained",
    "lts__cycles_elapsed.avg.per_second",
    "l1tex__cycles_elapsed.avg.per_second",
]
SUM_METRICS = {
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
}
CUDA_KERNELS = [
    "cuda_v1_inline",
    "cuda_v2_inline",
    "cuda_v3_inline",
    "cuda_v4_inline",
]
ROOFLINE_KERNELS = [
    "jax_v1",
    "jax_v1_5",
    *CUDA_KERNELS,
    "warp_v3_supercell_tile",
]
DEFAULT_KERNELS = [
    "jax_v1",
    "jax_v1_5",
    "cuda_v1_inline",
    "cuda_v2_inline",
    "cuda_v3_inline",
    "cuda_v4_inline",
]
KERNEL_MARKERS = {
    "jax_v1": "*",
    "jax_v1_5": "X",
    "cuda_v1_inline": "o",
    "cuda_v2_inline": "s",
    "cuda_v3_inline": "^",
    "cuda_v4_inline": "D",
    "warp_v3_supercell_tile": "P",
}
SUPPORTED_STAGES = ("p2g", "p2g_kernel")
SUPPORTED_ROOFLINES = ("ordinary", "hierarchical")


def _safe_rate(numerator: float, denominator: float):
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _metric_value(metric_values, metrics: list[str], metric: str):
    return float(metric_values[metrics.index(metric)])


def _detect_compute_cap():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return 8.0
    first = out.strip().splitlines()[0]
    return float(first)


def _l1_peak_metric(compute_cap: float):
    if compute_cap >= 9.0:
        return "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained"
    if compute_cap >= 7.5:
        return "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained"
    return "l1tex__lsu_writeback_active.sum.peak_sustained"


def _metrics_for_roofline(roofline: str, l1_peak_metric: str):
    metrics = [
        *WORK_METRICS,
        DRAM_TRAFFIC_METRIC,
        *ORDINARY_PEAK_METRICS,
    ]
    if roofline == "hierarchical":
        metrics = [
            *WORK_METRICS,
            *HIERARCHICAL_TRAFFIC_METRICS,
            *ORDINARY_PEAK_METRICS,
            *HIERARCHICAL_PEAK_METRICS,
            l1_peak_metric,
        ]
    return list(dict.fromkeys(metrics))


def _kernel_label(kernel_name: str):
    if kernel_name == "jax_v1_5":
        return "jax_v1_5"
    if kernel_name == "jax_v1":
        return "jax_v1"
    if kernel_name == "warp_v3_supercell_tile":
        return "warp_v3"
    return kernel_name.removeprefix("cuda_").removesuffix("_inline")


def _stage_label(stage_name: str):
    labels = {
        "p2g": "P2G Variant",
        "p2g_kernel": "P2G Kernel",
    }
    return labels.get(stage_name, stage_name.upper())


def _roofline_label(roofline: str):
    labels = {
        "ordinary": "Ordinary",
        "hierarchical": "Hierarchical",
    }
    return labels.get(roofline, roofline.title())


def _roofline_metric(metrics: list[str], l1_peak_metric: str):
    def derive_roofline(*args):
        metric_values = args[: len(metrics)]
        _stage_name, _kernel_name, n_particles, num_grids = args[len(metrics):]
        cycles = _metric_value(metric_values, metrics, "sm__cycles_elapsed.avg")
        cycles_per_second = _metric_value(
            metric_values,
            metrics,
            "sm__cycles_elapsed.avg.per_second",
        )
        seconds = _safe_rate(cycles, cycles_per_second)
        fadd = _metric_value(
            metric_values,
            metrics,
            "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        )
        fmul = _metric_value(
            metric_values,
            metrics,
            "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        )
        ffma = _metric_value(
            metric_values,
            metrics,
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        )
        dram_bytes = _metric_value(metric_values, metrics, DRAM_TRAFFIC_METRIC)
        fp32_flops = fadd + fmul + 2.0 * ffma

        sm_ffma_peak = _metric_value(
            metric_values,
            metrics,
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        )
        dram_peak = _metric_value(metric_values, metrics, "dram__bytes.sum.peak_sustained")
        dram_hz = _metric_value(metric_values, metrics, "dram__cycles_elapsed.avg.per_second")
        derived = {
            "time_ms": seconds * 1e3,
            "p2g_mparticles_per_s": _safe_rate(float(n_particles), seconds) / 1e6,
            "fp32_flops": fp32_flops,
            "fp32_gflops": _safe_rate(fp32_flops, seconds) / 1e9,
            "ai_dram_flop_per_byte": _safe_rate(fp32_flops, dram_bytes),
            "dram_bytes": dram_bytes,
            "fp32_peak_gflops": (sm_ffma_peak * 2.0 * cycles_per_second) / 1e9,
            "dram_peak_gbytes_per_s": (dram_peak * dram_hz) / 1e9,
            "n_particles": int(n_particles),
            "num_grids": int(num_grids),
        }
        if "l1tex__t_bytes.sum" in metrics:
            l1_bytes = _metric_value(metric_values, metrics, "l1tex__t_bytes.sum")
            l2_bytes = _metric_value(metric_values, metrics, "lts__t_bytes.sum")
            l2_peak_cycles = _metric_value(
                metric_values,
                metrics,
                "lts__lts2xbar_cycles_active.sum.peak_sustained",
            )
            l2_hz = _metric_value(metric_values, metrics, "lts__cycles_elapsed.avg.per_second")
            l1_peak_cycles = _metric_value(metric_values, metrics, l1_peak_metric)
            l1_hz = _metric_value(metric_values, metrics, "l1tex__cycles_elapsed.avg.per_second")
            derived.update(
                {
                    "ai_l1_flop_per_byte": _safe_rate(fp32_flops, l1_bytes),
                    "ai_l2_flop_per_byte": _safe_rate(fp32_flops, l2_bytes),
                    "l1_bytes": l1_bytes,
                    "l2_bytes": l2_bytes,
                    "l1_peak_gbytes_per_s": (l1_peak_cycles * 128.0 * l1_hz) / 1e9,
                    "l2_peak_gbytes_per_s": (l2_peak_cycles * 32.0 * l2_hz) / 1e9,
                }
            )

        return derived

    return derive_roofline


def _combine_roofline_metrics(metrics: list[str]):
    cycles_idx = metrics.index("sm__cycles_elapsed.avg")
    sm_hz_idx = metrics.index("sm__cycles_elapsed.avg.per_second")

    def combine(lhs, rhs):
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        out = lhs.copy()

        lhs_seconds = _safe_rate(lhs[cycles_idx], lhs[sm_hz_idx])
        rhs_seconds = _safe_rate(rhs[cycles_idx], rhs[sm_hz_idx])
        out[cycles_idx] = (lhs_seconds + rhs_seconds) * lhs[sm_hz_idx]
        out[sm_hz_idx] = lhs[sm_hz_idx]

        for idx, metric in enumerate(metrics):
            if idx in {cycles_idx, sm_hz_idx}:
                continue
            if metric in SUM_METRICS:
                out[idx] = lhs[idx] + rhs[idx]
            else:
                out[idx] = lhs[idx]
        return out

    return combine


def _make_cfg(args, kernel_name: str):
    return OmegaConf.create(
        {
            "benchmark": True,
            "kernel": {
                "name": kernel_name,
                "loop_kind": args.loop_kind,
                "cuda_graph": args.cuda_graph,
            },
            "material": {
                "elasticity": {
                    "name": "StVKElasticityJacobi",
                    "E": 2e6,
                    "nu": 0.4,
                },
                "plasticity": {
                    "name": "DruckerPragerPlasticityJacobi",
                    "E": 2e6,
                    "nu": 0.4,
                    "friction_angle": 25.0,
                    "cohesion": 0.0,
                },
                "color": "orange",
            },
            "sim": {
                "num_frames": 0,
                "steps_per_frame": 1,
                "n_particles": args.n_particles,
                "initial_velocity": [0.0, 0.0, 0.0],
                "num_grids": args.num_grids,
                "dt": 5.0e-5,
                "gravity": [0.0, 0.0, -9.8],
                "rho": 1000.0,
                "clip_bound": 0.5,
                "damping": 1.0,
                "center": [0.5, 0.5, 0.5],
                "size": [0.8, 0.8, 0.8],
                "boundary_conditions": [
                    {
                        "type": "surface_collider",
                        "point": [1.0, 1.0, 0.02],
                        "normal": [0.0, 0.0, 1.0],
                        "surface": "sticky",
                        "start_time": 0.0,
                        "end_time": 1e3,
                    }
                ],
            },
        }
    )


def _build_stage_runner(cfg, nsight_mod, stage_name: str):
    if stage_name not in SUPPORTED_STAGES:
        supported = ", ".join(SUPPORTED_STAGES)
        raise RuntimeError(f"Unsupported stage={stage_name!r}. Supported stages: {supported}.")

    import jax
    import jax.numpy as jnp

    from mpm_jax.blocks.init import get_particles
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.registry import KERNELS
    from mpm_jax.types import MPMState, make_params

    kernel_name = str(cfg.kernel.name)
    if kernel_name not in KERNELS:
        raise RuntimeError(f"Unsupported kernel={kernel_name!r}.")
    sim = cfg.sim
    mat = cfg.material
    n = int(sim.n_particles)
    params = make_params(
        n_particles=n,
        num_grids=int(sim.num_grids),
        dt=float(sim.dt),
        gravity=list(sim.gravity),
        rho=float(sim.rho),
        clip_bound=float(sim.clip_bound),
        damping=float(sim.damping),
        center=list(sim.center),
        size=list(sim.size),
    )
    particles = jnp.array(
        get_particles(n, center=list(sim.center), size=list(sim.size)),
        dtype=jnp.float32,
    )
    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity), dtype=jnp.float32), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    spec = KERNELS[kernel_name]
    frame_opts = dict(spec.defaults)
    for key in ("loop_kind", "cuda_graph", "graph_mode"):
        if key in cfg.kernel:
            frame_opts[key] = cfg.kernel[key]
    backend = spec.backend_factory(num_grids=params.num_grids, **frame_opts)
    elasticity_fn = get_constitutive(mat.elasticity)

    @jax.jit
    def jit_common_stress(state):
        return elasticity_fn(state.F)

    stress = jit_common_stress(state)
    jax.block_until_ready(stress)

    if stage_name == "p2g_kernel":
        @jax.jit
        def jit_prepare(state, stress):
            return backend.prepare(params, state, stress)

        prepared = jit_prepare(state, stress)
        jax.block_until_ready(prepared)

        @jax.jit
        def jit_stage(prepared):
            return backend.p2g(params, prepared)

        warmup = jit_stage(prepared)
        jax.block_until_ready(warmup)

        def run_stage_once():
            with nsight_mod.annotate(stage_name):
                out = jit_stage(prepared)
                jax.block_until_ready(out)

        return run_stage_once

    @jax.jit
    def jit_stage(state, stress):
        prepared = backend.prepare(params, state, stress)
        return backend.p2g(params, prepared)

    warmup = jit_stage(state, stress)
    jax.block_until_ready(warmup)

    def run_stage_once():
        with nsight_mod.annotate(stage_name):
            out = jit_stage(state, stress)
            jax.block_until_ready(out)

    return run_stage_once


def _value_from_df(
    df,
    metric: str,
    kernel_name: str | None = None,
    n_particles: int | None = None,
    num_grids: int | None = None,
):
    rows = df[df["Metric"] == metric]
    if kernel_name is not None:
        if "kernel_name" not in rows.columns:
            raise RuntimeError("Nsight results did not include the kernel_name config column.")
        rows = rows[rows["kernel_name"] == kernel_name]
    if n_particles is not None:
        if "n_particles" not in rows.columns:
            raise RuntimeError("Nsight results did not include the n_particles config column.")
        rows = rows[rows["n_particles"].astype(int) == int(n_particles)]
    if num_grids is not None:
        if "num_grids" not in rows.columns:
            raise RuntimeError("Nsight results did not include the num_grids config column.")
        rows = rows[rows["num_grids"].astype(int) == int(num_grids)]
    if rows.empty:
        suffix_parts = []
        if kernel_name is not None:
            suffix_parts.append(f"kernel {kernel_name!r}")
        if n_particles is not None:
            suffix_parts.append(f"n_particles={n_particles}")
        if num_grids is not None:
            suffix_parts.append(f"num_grids={num_grids}")
        suffix = "" if not suffix_parts else " for " + ", ".join(suffix_parts)
        raise RuntimeError(f"Nsight results did not include metric {metric!r}{suffix}.")
    value_column = "Value" if "Value" in rows else "AvgValue"
    return float(rows[value_column].iloc[0])


def _df_kernel_names(df):
    if "kernel_name" not in df:
        return [""]
    return list(dict.fromkeys(str(value) for value in df["kernel_name"]))


def _df_particle_counts(df):
    if "n_particles" not in df:
        return []
    return list(dict.fromkeys(int(value) for value in df["n_particles"]))


def _df_particle_counts_for_kernel(df, kernel_name: str):
    rows = df
    if "kernel_name" in rows.columns:
        rows = rows[rows["kernel_name"] == kernel_name]
    return list(dict.fromkeys(int(value) for value in rows["n_particles"]))


def _count_label(n_particles: int):
    if n_particles % 1_000_000 == 0:
        return f"{n_particles // 1_000_000}M"
    if n_particles % 1_000 == 0:
        return f"{n_particles // 1_000}k"
    return f"{n_particles:,}"


def _roofline_points(df, kernels: list[str], levels: list[tuple[str, str, str, str]]):
    rows = []
    for kernel_name in kernels:
        for n_particles in _df_particle_counts_for_kernel(df, kernel_name):
            performance = _value_from_df(
                df, "fp32_gflops", kernel_name, n_particles=n_particles)
            for label, ai_metric, _bw_metric, _color in levels:
                rows.append(
                    {
                        "kernel": _kernel_label(kernel_name),
                        "kernel_name": kernel_name,
                        "memory_level": label,
                        "n_particles": n_particles,
                        "particle_count": _count_label(n_particles),
                        "arithmetic_intensity": _value_from_df(
                            df, ai_metric, kernel_name, n_particles=n_particles),
                        "performance_gflops": performance,
                    }
                )
    return pd.DataFrame(rows)


def _roofline_levels(roofline: str):
    if roofline == "ordinary":
        return [
            ("HBM", "ai_dram_flop_per_byte", "dram_peak_gbytes_per_s", "#59a14f"),
        ]
    return [
        ("L1", "ai_l1_flop_per_byte", "l1_peak_gbytes_per_s", "#2878b5"),
        ("L2", "ai_l2_flop_per_byte", "l2_peak_gbytes_per_s", "#f28e2b"),
        ("HBM", "ai_dram_flop_per_byte", "dram_peak_gbytes_per_s", "#59a14f"),
    ]


def _draw_roofline(fig, df, title: str, kernels: list[str], stage_name: str, roofline: str):
    fig.clear()
    fig.set_size_inches(9.5, 6.2)
    fig.set_dpi(180)
    sns.set_theme(style="whitegrid")
    reference_kernel = kernels[0] if kernels else None
    fp32_peak = _value_from_df(df, "fp32_peak_gflops", reference_kernel)
    levels = _roofline_levels(roofline)
    particle_counts = _df_particle_counts(df)
    ais = [
        _value_from_df(df, ai_metric, kernel_name, n_particles=n_particles)
        for kernel_name in kernels
        for n_particles in _df_particle_counts_for_kernel(df, kernel_name)
        for _, ai_metric, _, _ in levels
    ]
    valid_ais = [value for value in ais if math.isfinite(value) and value > 0]
    if not valid_ais:
        raise RuntimeError("Roofline arithmetic intensities were not finite.")

    max_bw = max(_value_from_df(df, bw_metric, reference_kernel) for _, _, bw_metric, _ in levels)
    min_ai = min(valid_ais + [fp32_peak / max_bw]) / 4.0
    max_ai = max(
        valid_ais
        + [fp32_peak / _value_from_df(df, bw_metric, reference_kernel) for _, _, bw_metric, _ in levels]
    ) * 4.0
    x = [
        10
        ** (math.log10(min_ai) + i * (math.log10(max_ai) - math.log10(min_ai)) / 255)
        for i in range(256)
    ]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    palette = {label: color for label, _ai, _bw, color in levels}
    ax.axhline(fp32_peak, color="#2f2f2f", linewidth=1.6, linestyle="--", label="FP32 peak")
    peak_label_ai = 10 ** (
        math.log10(min_ai)
        + 0.08 * (math.log10(max_ai) - math.log10(min_ai))
    )
    ax.text(
        peak_label_ai,
        fp32_peak * 0.96,
        f"FP32 peak {fp32_peak / 1000:.2f} TFLOP/s",
        ha="left",
        va="top",
        fontsize=9,
        color="#2f2f2f",
    )

    for label, ai_metric, bw_metric, color in levels:
        bandwidth = _value_from_df(df, bw_metric, reference_kernel)
        y = [min(fp32_peak, bandwidth * xi) for xi in x]
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{label} roofline")
        roofline_knee_ai = fp32_peak / bandwidth
        visible_slope_min_ai = min_ai
        visible_slope_max_ai = min(max_ai, roofline_knee_ai)
        roof_ai = 10 ** (
            0.5
            * (
                math.log10(visible_slope_min_ai)
                + math.log10(visible_slope_max_ai)
            )
        )
        roof_perf = bandwidth * roof_ai
        ax.text(
            roof_ai,
            roof_perf,
            f"{label} {bandwidth / 1000:.2f} TB/s",
            color=color,
            fontsize=9,
            rotation=35,
            rotation_mode="anchor",
            ha="left",
            va="bottom",
        )

    points = _roofline_points(df, kernels, levels)
    if len(particle_counts) > 1:
        for (_kernel, memory_level), group in points.groupby(["kernel", "memory_level"]):
            group = group.sort_values("n_particles")
            ax.plot(
                group["arithmetic_intensity"],
                group["performance_gflops"],
                color=palette[memory_level],
                linewidth=1.3,
                alpha=0.45,
                zorder=3,
            )
    sns.scatterplot(
        data=points,
        x="arithmetic_intensity",
        y="performance_gflops",
        hue="memory_level",
        style="kernel",
        palette=palette,
        markers={_kernel_label(k): KERNEL_MARKERS.get(k, "o") for k in kernels},
        edgecolor="black",
        s=80,
        legend=False,
        ax=ax,
        zorder=4,
    )

    num_grids = int(_value_from_df(df, "num_grids", reference_kernel))
    stage_label = _stage_label(stage_name)
    roofline_label = _roofline_label(roofline)
    if title:
        ax.set_title(title)
    elif len(kernels) == 1:
        particle_label = (
            f"{particle_counts[0]:,} particles"
            if len(particle_counts) == 1
            else f"{_count_label(min(particle_counts))}-{_count_label(max(particle_counts))} particles"
        )
        ax.set_title(
            f"{roofline_label} FP32 Roofline: {stage_label}, {kernels[0]} "
            f"({particle_label}, {num_grids}^3 grid)"
        )
    else:
        particle_label = (
            f"{particle_counts[0]:,} particles"
            if len(particle_counts) == 1
            else f"{_count_label(min(particle_counts))}-{_count_label(max(particle_counts))} particles"
        )
        ax.set_title(
            f"{roofline_label} FP32 Roofline {stage_label} Sweep "
            f"({particle_label}, {num_grids}^3 grid)"
        )
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.grid(True, which="both", alpha=0.25)
    memory_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor=color,
            markeredgecolor="black",
            linestyle="None",
            markersize=7,
            label=label,
        )
        for label, _, _, color in levels
    ]
    kernel_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=KERNEL_MARKERS.get(kernel_name, "o"),
            linestyle="None",
            markersize=7,
            label=_kernel_label(kernel_name),
        )
        for kernel_name in kernels
    ]
    memory_legend = ax.legend(
        handles=memory_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        title="Memory level",
        frameon=False,
    )
    ax.add_artist(memory_legend)
    ax.legend(
        handles=kernel_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.72),
        borderaxespad=0,
        title="Kernel",
        frameon=False,
    )


def _plot_roofline(df, output: Path, title: str, kernels: list[str], stage_name: str, roofline: str):
    output.parent.mkdir(parents=True, exist_ok=True)

    def customize_roofline(fig):
        _draw_roofline(fig, df, title, kernels, stage_name, roofline)

    visualize(
        df,
        metric="fp32_gflops",
        row_panels=None,
        col_panels=None,
        x_keys=["n_particles"],
        title=title or f"{_roofline_label(roofline)} FP32 Roofline {_stage_label(stage_name)}",
        filename=str(output),
        ylabel="Performance [GFLOP/s]",
        annotate_points=False,
        show_avg=False,
        show_geomean=False,
        plot_width=9.5,
        plot_height=6.2,
        plot_callback=customize_roofline,
    )


def _write_summary(df, output: Path):
    value_column = "Value" if "Value" in df else "AvgValue"
    particle_counts = _df_particle_counts(df)
    if "kernel_name" in df and len(particle_counts) > 1:
        rows: dict[str, dict[str, dict[str, float]]] = {}
        for _, row in df.iterrows():
            kernel_name = str(row["kernel_name"])
            particle_count = str(int(row["n_particles"]))
            rows.setdefault(kernel_name, {}).setdefault(particle_count, {})[
                str(row["Metric"])
            ] = row[value_column]
    elif "kernel_name" in df:
        rows: dict[str, dict[str, float]] = {}
        for _, row in df.iterrows():
            kernel_name = str(row["kernel_name"])
            rows.setdefault(kernel_name, {})[str(row["Metric"])] = row[value_column]
    else:
        rows = {str(row["Metric"]): row[value_column] for _, row in df.iterrows()}
    output.write_text(json.dumps(rows, indent=2))


def _parse_kernels(args):
    if args.kernel is not None and args.kernels is not None:
        raise RuntimeError("Use either --kernel for one backend or --kernels for a sweep, not both.")
    raw_kernels = args.kernels
    if args.kernel is not None:
        raw_kernels = [args.kernel]
    if raw_kernels is None:
        raw_kernels = list(DEFAULT_KERNELS)

    kernels = []
    for item in raw_kernels:
        for kernel_name in item.split(","):
            kernel_name = kernel_name.strip()
            if not kernel_name:
                continue
            if kernel_name == "all_cuda":
                kernels.extend(CUDA_KERNELS)
            elif kernel_name == "all":
                kernels.extend(ROOFLINE_KERNELS)
            else:
                kernels.append(kernel_name)

    kernels = list(dict.fromkeys(kernels))
    unsupported = [kernel_name for kernel_name in kernels if kernel_name not in ROOFLINE_KERNELS]
    if unsupported:
        raise RuntimeError(
            "This script profiles solver-stage rooflines. "
            f"Unsupported kernel(s): {', '.join(unsupported)}. "
            f"Supported: {', '.join(ROOFLINE_KERNELS)}"
        )
    supercell_kernels = {"cuda_v4_inline", "warp_v3_supercell_tile"}.intersection(kernels)
    if supercell_kernels and args.num_grids % 2 != 0:
        raise RuntimeError(
            f"{', '.join(sorted(supercell_kernels))} requires --num-grids divisible by 2. "
            f"Got --num-grids {args.num_grids}."
        )
    return kernels


def _parse_particle_counts(args):
    raw_counts = args.particle_counts
    if raw_counts is None:
        return [int(args.n_particles)]
    counts = []
    for item in raw_counts:
        for value in str(item).split(","):
            value = value.strip()
            if not value:
                continue
            count = int(value)
            if count <= 0:
                raise RuntimeError(f"Particle counts must be positive. Got {count}.")
            counts.append(count)
    counts = list(dict.fromkeys(counts))
    if not counts:
        raise RuntimeError("--particle-counts did not contain any valid counts.")
    return counts


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=None, help="Profile one solver backend.")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        help=(
            "Profile several solver backends. Accepts space-separated values, "
            "comma-separated values, all_cuda, or all. Defaults to jax_v1 "
            "jax_v1_5 cuda_v1_inline cuda_v2_inline cuda_v3_inline cuda_v4_inline."
        ),
    )
    parser.add_argument(
        "--roofline",
        default="ordinary",
        choices=SUPPORTED_ROOFLINES,
        help="ordinary uses FP32 work and HBM traffic; hierarchical also collects L1/L2.",
    )
    parser.add_argument("--n-particles", type=int, default=1_000_000)
    parser.add_argument(
        "--particle-counts",
        nargs="+",
        default=None,
        help=(
            "Profile a particle-count trajectory. Accepts space-separated or "
            "comma-separated counts. Defaults to --n-particles."
        ),
    )
    parser.add_argument("--num-grids", type=int, default=124)
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=10,
        help=(
            "Accepted for compatibility with benchmark commands. This stage profiler "
            "runs one annotated stage invocation, not a full frame loop."
        ),
    )
    parser.add_argument(
        "--stage",
        default="p2g_kernel",
        choices=SUPPORTED_STAGES,
        help=(
            "Annotated solver stage to profile. p2g includes backend.prepare plus "
            "backend.p2g; p2g_kernel excludes backend.prepare while including "
            "the backend P2G call's required grid zeroing."
        ),
    )
    parser.add_argument("--loop-kind", default="fori", choices=["fori", "python"])
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--replay-mode",
        default="kernel",
        choices=["range", "kernel"],
        help=(
            "Nsight Compute replay mode. kernel is required for the SASS FP32 counters "
            "used by this roofline; range is only useful if the metric list is changed "
            "to range-compatible counters."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/roofline/roofline.png"),
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--output-csv", action="store_true")
    parser.add_argument("--ignore-kernel", action="append", default=[])
    return parser.parse_args()


def _env_ignore_kernels():
    raw = os.environ.get("ROOFLINE_IGNORE_KERNELS_JSON")
    if not raw:
        return []
    values = json.loads(raw)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RuntimeError("ROOFLINE_IGNORE_KERNELS_JSON must be a JSON string list.")
    return values


def main():
    args = _parse_args()
    args.ignore_kernel = [*args.ignore_kernel, *_env_ignore_kernels()]
    kernels = _parse_kernels(args)
    particle_counts = _parse_particle_counts(args)
    run_dir = args.output.resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    compute_cap = _detect_compute_cap()
    l1_peak_metric = _l1_peak_metric(compute_cap)
    metrics = _metrics_for_roofline(args.roofline, l1_peak_metric)

    _prepare_nsight_child_python(run_dir)

    dataframes = []

    def profiled_variant(stage_name, kernel_name, n_particles, num_grids):
        cfg = _make_cfg(args, str(kernel_name))
        cfg.sim.n_particles = int(n_particles)
        cfg.sim.num_grids = int(num_grids)
        _build_stage_runner(cfg, nsight, str(stage_name))()

    configs = [
        (
            args.stage,
            kernel_name,
            n_particles,
            args.num_grids,
        )
        for kernel_name in kernels
        for n_particles in particle_counts
    ]

    with _disable_editable_pth_for_nsight():
        profiled_kernel = nsight.analyze.kernel(
            configs=configs,
            runs=args.runs,
            metrics=metrics,
            derive_metric=_roofline_metric(metrics, l1_peak_metric),
            replay_mode=args.replay_mode,
            combine_kernel_metrics=(
                _combine_roofline_metrics(metrics)
                if args.replay_mode == "kernel"
                else None
            ),
            ignore_kernel_list=args.ignore_kernel,
            output="progress",
            output_csv=args.output_csv,
            output_prefix=str(
                run_dir
                / f"{args.roofline}_roofline_{args.stage}_{args.replay_mode}_"
            ),
        )(profiled_variant)
        results = _run_nsight_profile(profiled_kernel)
        if results is None:
            if os.environ.get("NSPY_NCU_PROFILE"):
                return
            raise RuntimeError("Nsight Python did not return profiling results.")
        dataframes.append(results.to_dataframe())

    df = pd.concat(dataframes, ignore_index=True)
    _plot_roofline(
        df,
        args.output.resolve(),
        args.title,
        _df_kernel_names(df),
        args.stage,
        args.roofline,
    )
    _write_summary(df, args.output.with_suffix(".json").resolve())
    print(df)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
