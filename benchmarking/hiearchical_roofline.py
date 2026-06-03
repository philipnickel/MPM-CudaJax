"""Collect and plot one hierarchical FP32 Roofline point with Nsight Python."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import nsight
from nsight.visualization import visualize
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_nsight import (  # noqa: E402
    _disable_editable_pth_for_nsight,
    _prepare_nsight_child_python,
    _run_nsight_profile,
)

APP_METRICS = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
]
PEAK_METRICS = [
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "lts__lts2xbar_cycles_active.sum.peak_sustained",
    "lts__cycles_elapsed.avg.per_second",
    "l1tex__cycles_elapsed.avg.per_second",
]


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


def _roofline_metric(metrics: list[str], l1_peak_metric: str):
    def derive_roofline(*args):
        metric_values = args[: len(metrics)]
        _kernel_name, n_particles, num_grids, _steps_per_frame = args[len(metrics):]
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
        l1_bytes = _metric_value(metric_values, metrics, "l1tex__t_bytes.sum")
        l2_bytes = _metric_value(metric_values, metrics, "lts__t_bytes.sum")
        dram_bytes = _metric_value(metric_values, metrics, "dram__bytes.sum")
        fp32_flops = fadd + fmul + 2.0 * ffma

        sm_ffma_peak = _metric_value(
            metric_values,
            metrics,
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        )
        dram_peak = _metric_value(metric_values, metrics, "dram__bytes.sum.peak_sustained")
        dram_hz = _metric_value(metric_values, metrics, "dram__cycles_elapsed.avg.per_second")
        l2_peak_cycles = _metric_value(
            metric_values,
            metrics,
            "lts__lts2xbar_cycles_active.sum.peak_sustained",
        )
        l2_hz = _metric_value(metric_values, metrics, "lts__cycles_elapsed.avg.per_second")
        l1_peak_cycles = _metric_value(metric_values, metrics, l1_peak_metric)
        l1_hz = _metric_value(metric_values, metrics, "l1tex__cycles_elapsed.avg.per_second")

        return {
            "time_ms": seconds * 1e3,
            "p2g_mparticles_per_s": _safe_rate(float(n_particles), seconds) / 1e6,
            "fp32_flops": fp32_flops,
            "fp32_gflops": _safe_rate(fp32_flops, seconds) / 1e9,
            "ai_l1_flop_per_byte": _safe_rate(fp32_flops, l1_bytes),
            "ai_l2_flop_per_byte": _safe_rate(fp32_flops, l2_bytes),
            "ai_dram_flop_per_byte": _safe_rate(fp32_flops, dram_bytes),
            "l1_bytes": l1_bytes,
            "l2_bytes": l2_bytes,
            "dram_bytes": dram_bytes,
            "fp32_peak_gflops": (sm_ffma_peak * 2.0 * cycles_per_second) / 1e9,
            "l1_peak_gbytes_per_s": (l1_peak_cycles * 128.0 * l1_hz) / 1e9,
            "l2_peak_gbytes_per_s": (l2_peak_cycles * 32.0 * l2_hz) / 1e9,
            "dram_peak_gbytes_per_s": (dram_peak * dram_hz) / 1e9,
            "n_particles": int(n_particles),
            "num_grids": int(num_grids),
        }

    return derive_roofline


def _make_cfg(args):
    return OmegaConf.create(
        {
            "benchmark": True,
            "kernel": {
                "name": args.kernel,
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
                "num_frames": 1,
                "steps_per_frame": args.steps_per_frame,
                "n_particles": args.n_particles,
                "initial_velocity": [0.0, 0.0, -0.5],
                "num_grids": args.num_grids,
                "dt": 3e-4,
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


def _build_single_p2g_runner(cfg, nsight_mod):
    import jax
    import jax.numpy as jnp

    from mpm_jax.blocks.init import get_particles
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.registry import KERNELS
    from mpm_jax.types import MPMState, make_params

    kernel_name = str(cfg.kernel.name)
    if kernel_name not in KERNELS:
        raise RuntimeError(f"Unsupported kernel={kernel_name!r}.")
    if not kernel_name.startswith("cuda_"):
        raise RuntimeError(
            "The standalone hierarchical Roofline script profiles one CUDA P2G kernel. "
            f"Got kernel={kernel_name!r}."
        )

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
    def prepare_once(state):
        stress = elasticity_fn(state.F)
        return backend.prepare(params, state, stress)

    prepared = prepare_once(state)
    jax.block_until_ready(prepared)

    @jax.jit
    def jit_p2g(prepared):
        return backend.p2g(params, prepared)

    warmup = jit_p2g(prepared)
    jax.block_until_ready(warmup)
    annotation_name = f"{kernel_name}_p2g_kernel"

    def run_p2g_once():
        with nsight_mod.annotate(annotation_name):
            out = jit_p2g(prepared)
            jax.block_until_ready(out)

    return run_p2g_once


def _value_from_df(df, metric: str):
    rows = df[df["Metric"] == metric]
    if rows.empty:
        raise RuntimeError(f"Nsight results did not include metric {metric!r}.")
    value_column = "Value" if "Value" in rows else "AvgValue"
    return float(rows[value_column].iloc[0])


def _draw_roofline(fig, df, title: str, kernel: str):
    fig.clear()
    fig.set_size_inches(9.5, 6.2)
    fig.set_dpi(180)
    fp32_peak = _value_from_df(df, "fp32_peak_gflops")
    levels = [
        ("L1", "ai_l1_flop_per_byte", "l1_peak_gbytes_per_s", "#2878b5"),
        ("L2", "ai_l2_flop_per_byte", "l2_peak_gbytes_per_s", "#f28e2b"),
        ("DRAM", "ai_dram_flop_per_byte", "dram_peak_gbytes_per_s", "#59a14f"),
    ]
    ais = [_value_from_df(df, ai_metric) for _, ai_metric, _, _ in levels]
    valid_ais = [value for value in ais if math.isfinite(value) and value > 0]
    if not valid_ais:
        raise RuntimeError("Roofline arithmetic intensities were not finite.")

    max_bw = max(_value_from_df(df, bw_metric) for _, _, bw_metric, _ in levels)
    min_ai = min(valid_ais + [fp32_peak / max_bw]) / 4.0
    max_ai = max(valid_ais + [fp32_peak / _value_from_df(df, bw) for _, _, bw, _ in levels]) * 4.0
    x = [10 ** (math.log10(min_ai) + i * (math.log10(max_ai) - math.log10(min_ai)) / 255) for i in range(256)]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
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

    achieved_gflops = _value_from_df(df, "fp32_gflops")
    for label, ai_metric, bw_metric, color in levels:
        bandwidth = _value_from_df(df, bw_metric)
        y = [min(fp32_peak, bandwidth * xi) for xi in x]
        ax.plot(x, y, color=color, linewidth=2.0, label=f"{label} roofline")
        ai = _value_from_df(df, ai_metric)
        ax.scatter(ai, achieved_gflops, color=color, s=80, edgecolor="black", zorder=4)
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

    n_particles = int(_value_from_df(df, "n_particles"))
    num_grids = int(_value_from_df(df, "num_grids"))
    ax.set_title(title or f"Hierarchical FP32 Roofline: {kernel} ({n_particles:,} particles, {num_grids}^3 grid)")
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False)


def _plot_roofline(df, output: Path, title: str, kernel: str):
    output.parent.mkdir(parents=True, exist_ok=True)

    def customize_roofline(fig):
        _draw_roofline(fig, df, title, kernel)

    visualize(
        df,
        metric="fp32_gflops",
        row_panels=None,
        col_panels=None,
        x_keys=["n_particles"],
        title=title
        or f"Hierarchical FP32 Roofline: {kernel}",
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
    rows = {str(row["Metric"]): row[value_column] for _, row in df.iterrows()}
    output.write_text(json.dumps(rows, indent=2))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="cuda_v3_inline")
    parser.add_argument("--n-particles", type=int, default=8_000_000)
    parser.add_argument("--num-grids", type=int, default=125)
    parser.add_argument("--steps-per-frame", type=int, default=1)
    parser.add_argument("--loop-kind", default="fori", choices=["fori", "python"])
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/roofline/hierarchical_roofline.png"),
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--output-csv", action="store_true")
    parser.add_argument("--ignore-kernel", action="append", default=["zero_kernel"])
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = args.output.resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    compute_cap = _detect_compute_cap()
    l1_peak_metric = _l1_peak_metric(compute_cap)
    metrics = [*APP_METRICS, *PEAK_METRICS, l1_peak_metric]

    _prepare_nsight_child_python(run_dir)

    def profiled_variant(kernel_name, n_particles, num_grids, steps_per_frame):
        cfg = _make_cfg(args)
        cfg.kernel.name = str(kernel_name)
        cfg.sim.n_particles = int(n_particles)
        cfg.sim.num_grids = int(num_grids)
        cfg.sim.steps_per_frame = int(steps_per_frame)
        _build_single_p2g_runner(cfg, nsight)()

    profiled_variant = nsight.analyze.kernel(
        configs=[(args.kernel, args.n_particles, args.num_grids, args.steps_per_frame)],
        runs=args.runs,
        metrics=metrics,
        derive_metric=_roofline_metric(metrics, l1_peak_metric),
        replay_mode="kernel",
        ignore_kernel_list=args.ignore_kernel,
        output="progress",
        output_csv=args.output_csv,
        output_prefix=str(run_dir / "hierarchical_roofline_"),
    )(profiled_variant)

    with _disable_editable_pth_for_nsight():
        results = _run_nsight_profile(profiled_variant)
    df = results.to_dataframe()
    _plot_roofline(df, args.output.resolve(), args.title, args.kernel)
    _write_summary(df, args.output.with_suffix(".json").resolve())
    print(df)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
