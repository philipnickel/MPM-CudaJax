"""Hydra-driven Nsight Compute (NCU) profiler for the current JAX-loop MPM backends.

One Hydra job profiles **one** backend variant (one config). Sweep across
backends / particle counts with **Hydra multirun**, exactly like ``simulate.py``::

    pixi run python profile_nsight.py -cn nsight_profile -m \
        backend=cuda_v1,cuda_v2,cuda_v3,cuda_v4,cutile_v1,cutile_v3 \
        nsight.target=scatter

Each job writes its NCU-derived metrics as one wide row to a per-run
``nsight_metrics.{json,csv}`` and appends the same row to a sweep-level
``results.csv`` (under the multirun sweep dir). ``postprocessing/nsight_plots.py``
loads that ``results.csv`` into pandas and renders the analysis figures.

Execution model note: nsight-python re-execs this script under ``ncu`` with the
same argv; the child selects the matching config by signature. Under Hydra
multirun the child re-runs the whole sweep, but non-matching jobs return ``None``
(no NCU launch, no writes) and the matching job ``os._exit(0)``s inside NCU --
so only the parent process ever writes results. The solver is built lazily inside
the profiled callable so non-matching child jobs stay cheap.
"""

from __future__ import annotations

# ruff: noqa: E402 -- XLA_FLAGS must be set before importing jax (via mpm_jax).

import csv
import importlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

# NCU's range capture cannot profile cuGraphLaunch, so disable XLA command
# buffers (CUDA graphs) before jax initializes (it does so during the mpm_jax
# imports below). XLA reads XLA_FLAGS at backend init; the modified env is also
# inherited by the NCU child process that re-runs this module.
_xla_flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = (
    re.sub(
        r"--xla_gpu_enable_command_buffer=\S*",
        "--xla_gpu_enable_command_buffer=",
        _xla_flags,
    )
    if "--xla_gpu_enable_command_buffer" in _xla_flags
    else f"{_xla_flags} --xla_gpu_enable_command_buffer=".strip()
)
# JAX pre-allocates ~75% of GPU memory by default, which collides with NCU's own
# device memory while profiling (CUDA_ERROR_OUT_OF_MEMORY). On-demand allocation
# keeps the footprint small -- the single-substep targets need little memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mpm_jax.p2g.backends as backend_configs
from mpm_jax.profiling import block_until_ready, build_profile_target
from mpm_jax.solver import MPMSolver

logger = logging.getLogger(__name__)


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - optional profiling package
        return None


nsight = _optional_module("nsight")

# Only these top-level nsight.* keys are recognized; anything else is a typo.
_SCRIPT_NSIGHT_KEYS = {"target", "write_json", "analyze"}
# nsight.analyze.configs is derived from the Hydra config, never user-supplied.
_UNSUPPORTED_ANALYZE_CONFIG_KEYS = {"configs"}


# ---------------------------------------------------------------------------
# GPU-kind tag + Hydra resolver (mirrors simulate.py so outputs land under the
# same gpu-kind partitioning).
# ---------------------------------------------------------------------------
def _slugify(value):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def _current_gpu_kind():
    override = os.environ.get("MPM_GPU_KIND")
    if override:
        return _slugify(override)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return _slugify(result.stdout.splitlines()[0])
    except Exception:
        return "unknown"


OmegaConf.register_new_resolver("gpu_kind", _current_gpu_kind, replace=True)


_SPEED_OF_LIGHT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
]
# Hierarchical single-precision (fp32) roofline. Achieved FLOP counts + per-level
# traffic + the PEAK ceiling rollups NCU derives from the running chip (so the
# L1/L2/HBM + compute ceilings are architecture-correct on A100/H100/GH200 with
# no datasheet hardcoding). Mirrors NCU's
# SpeedOfLight_HierarchicalSingleRooflineChart section: per-level bytes use the
# same counter for the achieved point and the peak ceiling (L1 = lsu_writeback
# cycles x 128 B; L2 = lts2xbar cycles x 32 B; HBM = dram bytes), peak rate =
# <counter>.peak_sustained x <level clock>.per_second. Derived in _roofline_metric.
_ROOFLINE_METRICS = [
    "gpu__time_duration.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "sm__cycles_elapsed.avg.per_second",
    "dram__bytes.sum",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "lts__lts2xbar_cycles_active.sum",
    "lts__lts2xbar_cycles_active.sum.peak_sustained",
    "lts__cycles_elapsed.avg.per_second",
    "l1tex__lsu_writeback_active_mem_lg.sum",
    "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained",
    "l1tex__cycles_elapsed.avg.per_second",
]
# P2G scatter writes through the atomic/reduction path; fire-and-forget atomicAdd
# lowers to hardware RED (op_red), so collect op_red + op_atom + op_st and sum
# (cuTile's atomic_store_add may land on op_st). Per-particle normalization in
# _diagnostic_metric. lg_throttle (global) vs mio_throttle (shared) is the
# on-chip-reduction contention fingerprint.
_ATOMIC_METRICS = [
    "gpu__time_duration.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red.ratio",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_red.sum.pct_of_peak_sustained_elapsed",
    "lts__d_atomic_input_cycles_active.sum.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
]
_MEMORY_LOCALITY_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red.ratio",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__t_sector_hit_rate.pct",
]
# Occupancy + its binding limiter (smallest launch__occupancy_limit_*) +
# context. These launch__* metrics collect at runtime though they are absent
# from `ncu --query-metrics`. sm__maximum_warps_avg_per_active_cycle == 64 on sm_80.
_OCCUPANCY_METRICS = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_avg_per_active_cycle",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
]
# Warp-issue-stall breakdown for a stacked bar: the per_issue_active.ratio family
# shares a denominator so the reasons sum to total resident-stalled warps/issue.
_STALL_REASONS = [
    "long_scoreboard",
    "lg_throttle",
    "short_scoreboard",
    "mio_throttle",
    "barrier",
    "math_pipe_throttle",
    "wait",
    "not_selected",
]
_SCHEDULER_METRICS = [
    "gpu__time_duration.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    *[
        f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio"
        for reason in _STALL_REASONS
    ],
]
def _dedupe(seq):
    return list(dict.fromkeys(seq))


# Everything in one NCU pass-set: roofline + SOL + atomics + locality + occupancy
# + scheduler, so a single sweep yields every column the analysis plots need.
_FULL_METRICS = _dedupe(
    [
        *_ROOFLINE_METRICS,
        *_SPEED_OF_LIGHT_METRICS,
        *_ATOMIC_METRICS,
        *_MEMORY_LOCALITY_METRICS,
        *_OCCUPANCY_METRICS,
        *_SCHEDULER_METRICS,
    ]
)
_METRIC_PRESETS = {
    "time": ["gpu__time_duration.sum"],
    "speed_of_light": _SPEED_OF_LIGHT_METRICS,
    "sol": _SPEED_OF_LIGHT_METRICS,
    "roofline": _ROOFLINE_METRICS,
    "full": _FULL_METRICS,
    # Nsight Compute metric names validated on A100 (sm_80) / NCU 2026.2.
    # Re-check with `ncu --query-metrics` when moving GPUs or Nsight versions,
    # especially the scheduler/L1TEX/LTS submetrics and launch__* occupancy.
    "memory_locality": _MEMORY_LOCALITY_METRICS,
    "atomics": _ATOMIC_METRICS,
    "atomic": _ATOMIC_METRICS,
    "occupancy": _OCCUPANCY_METRICS,
    "scheduler": _SCHEDULER_METRICS,
}


def _normalize_backend_choice(value) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Backend choices must be Hydra config-group names.")
    choice = value.strip()
    supported = backend_configs.backend_choices()
    if choice not in supported:
        raise RuntimeError(
            f"Unsupported backend choice {value!r}. "
            "Use Hydra backend choices: " + ", ".join(supported)
        )
    return choice


def _backend_choice_from_backend_cfg(backend_cfg) -> str:
    target = backend_cfg.get("_target_", None)
    for choice in backend_configs.backend_choices():
        if target == backend_configs.backend_config(choice).get("_target_", None):
            return choice
    raise RuntimeError(
        "Could not infer backend choice from cfg.backend. "
        "Use one of: " + ", ".join(backend_configs.backend_choices())
    )


def _backend_choice_from_cfg(cfg: DictConfig):
    try:
        return _normalize_backend_choice(HydraConfig.get().runtime.choices["backend"])
    except (ValueError, KeyError):
        pass
    return _backend_choice_from_backend_cfg(cfg.get("backend", {}))


def _profile_config(cfg: DictConfig, backend_choice: str):
    """The single (backend, n, g, steps) config tuple this job profiles.

    nsight-python records these as DataFrame columns and forwards n_particles to
    the derive-metric callbacks (per-particle normalization)."""
    return (
        backend_choice,
        int(cfg.sim.n_particles),
        int(cfg.sim.num_grids),
        int(cfg.sim.steps_per_frame),
    )


def _require_nsight():
    if nsight is None:
        raise RuntimeError(
            "nsight-python is not installed. Run `pixi install` in the GPU environment."
        )
    return nsight


def _profile_runner(cfg: DictConfig, nsight):
    """Return a zero-arg callable that runs the warmed profile target once.

    The solver + target are built lazily on first call and cached: nsight only
    invokes this for the config whose signature matches the NCU child, so
    non-matching Hydra-multirun jobs in the child never pay the JAX warmup."""
    target_name = str(cfg.nsight.get("target", "p2g"))
    cache = {}

    def run_once():
        target = cache.get("target")
        if target is None:
            solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
            target = build_profile_target(solver, target_name)
            cache["target"] = target
        with nsight.annotate(target.annotation):
            block_until_ready(target.run())

    return run_once


# ---------------------------------------------------------------------------
# Derived metrics (achieved roofline + per-particle/atomics/occupancy/scheduler
# diagnostics). nsight passes (metric_values..., config_values...) to each.
# ---------------------------------------------------------------------------
def _value_for_metric(metric_values, metrics: list[str], metric: str):
    if metric not in metrics:
        raise RuntimeError(
            f"Configured derive_metric requires metric {metric!r}. "
            f"Configured metrics: {metrics}"
        )
    return float(metric_values[metrics.index(metric)])


def _optional_value_for_metric(metric_values, metrics: list[str], metric: str):
    if metric not in metrics:
        return None
    return float(metric_values[metrics.index(metric)])


def _n_particles_from_config(config_values):
    if not config_values:
        raise RuntimeError("Expected config values to include n_particles.")
    if isinstance(config_values[0], str):
        if len(config_values) < 2:
            raise RuntimeError(
                "Expected legacy config values to include kernel_name and n_particles."
            )
        return int(config_values[1])
    return int(config_values[0])


def _p2g_throughput_metric(metrics: list[str]):
    def derive_p2g_throughput(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics) :]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
        }

    return derive_p2g_throughput


def _speed_of_light_metric(metrics: list[str]):
    def derive_speed_of_light(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics) :]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        sm_pct = _value_for_metric(
            metric_values,
            metrics,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        compute_memory_pct = _value_for_metric(
            metric_values,
            metrics,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        )
        dram_pct = _value_for_metric(
            metric_values,
            metrics,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
            "sol_sm_pct": sm_pct,
            "sol_compute_memory_pct": compute_memory_pct,
            "sol_dram_pct": dram_pct,
            "sol_max_pct": max(sm_pct, compute_memory_pct, dram_pct),
        }

    return derive_speed_of_light


def _diagnostic_metric(metrics: list[str]):
    def derive_diagnostics(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics) :]
        n_particles = _n_particles_from_config(config_values)
        out = {}

        time_ns = _optional_value_for_metric(
            metric_values, metrics, "gpu__time_duration.sum"
        )
        if time_ns is not None:
            seconds = time_ns / 1e9
            out["time_ms"] = time_ns / 1e6
            out["p2g_mparticles_per_s"] = (n_particles / seconds) / 1e6

        for metric, column in [
            ("dram__bytes.sum", "dram_bytes_per_particle"),
            ("dram__bytes_read.sum", "dram_read_bytes_per_particle"),
            ("dram__bytes_write.sum", "dram_write_bytes_per_particle"),
        ]:
            value = _optional_value_for_metric(metric_values, metrics, metric)
            if value is not None:
                out[column] = value / n_particles

        for op in ("ld", "st", "atom", "red"):
            request_metric = f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum"
            sector_metric = f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum"
            requests = _optional_value_for_metric(
                metric_values, metrics, request_metric
            )
            sectors = _optional_value_for_metric(metric_values, metrics, sector_metric)
            if requests is not None:
                out[f"global_{op}_requests_per_particle"] = requests / n_particles
            if sectors is not None:
                out[f"global_{op}_sectors_per_particle"] = sectors / n_particles
            if requests not in (None, 0.0) and sectors is not None:
                out[f"global_{op}_sectors_per_request"] = sectors / requests

        # Apples-to-apples scatter-write aggregate over the three global write-op
        # classes: fire-and-forget atomicAdd -> op_red, cuTile store-add -> op_st.
        scatter_req = sum(
            _optional_value_for_metric(
                metric_values,
                metrics,
                f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum",
            )
            or 0.0
            for op in ("red", "atom", "st")
        )
        scatter_sec = sum(
            _optional_value_for_metric(
                metric_values,
                metrics,
                f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum",
            )
            or 0.0
            for op in ("red", "atom", "st")
        )
        if scatter_req:
            out["global_scatter_requests_per_particle"] = scatter_req / n_particles
            out["global_scatter_sectors_per_particle"] = scatter_sec / n_particles
            out["global_scatter_sectors_per_request"] = scatter_sec / scatter_req

        atom_requests = _optional_value_for_metric(
            metric_values, metrics, "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"
        )
        red_requests = _optional_value_for_metric(
            metric_values, metrics, "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"
        )
        if atom_requests is not None or red_requests is not None:
            out["global_atomic_or_reduction_requests_per_particle"] = (
                (atom_requests or 0.0) + (red_requests or 0.0)
            ) / n_particles
            out["expected_p2g_float_atomic_ops_per_particle"] = 108.0

        for metric, column in [
            (
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sol_sm_pct",
            ),
            (
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                "sol_compute_memory_pct",
            ),
            (
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "sol_dram_pct",
            ),
            (
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "active_warps_pct",
            ),
            (
                "smsp__warps_eligible.avg.per_cycle_active",
                "eligible_warps_per_cycle",
            ),
            (
                "smsp__issue_active.avg.pct_of_peak_sustained_active",
                "issue_active_pct",
            ),
            ("lts__throughput.avg.pct_of_peak_sustained_elapsed", "sol_l2_pct"),
            ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "sol_l1_pct"),
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
            ("launch__shared_mem_per_block_static", "smem_bytes_per_block"),
        ]:
            value = _optional_value_for_metric(metric_values, metrics, metric)
            if value is not None:
                out[column] = value

        # Occupancy limiter = the smallest launch__occupancy_limit_* (warps/SM);
        # theoretical occupancy = that / the HW max warps (64 on sm_80).
        max_warps = (
            _optional_value_for_metric(
                metric_values, metrics, "sm__maximum_warps_avg_per_active_cycle"
            )
            or 64.0
        )
        occ_limits = {
            name: _optional_value_for_metric(
                metric_values, metrics, f"launch__occupancy_limit_{name}"
            )
            for name in ("registers", "shared_mem", "warps", "blocks")
        }
        present = {k: v for k, v in occ_limits.items() if v is not None}
        if present and max_warps:
            binder = min(present, key=present.get)
            out["theoretical_occ_pct"] = 100.0 * present[binder] / max_warps
            out["occ_limiter_code"] = {
                "registers": 0,
                "shared_mem": 1,
                "warps": 2,
                "blocks": 3,
            }[binder]

        # Warp-issue-stall breakdown. New per_issue_active.ratio family (stackable;
        # reasons sum to stallr_total) plus the legacy pct family for back-compat.
        stallr_total = 0.0
        for metric in metrics:
            new_prefix = "smsp__average_warps_issue_stalled_"
            new_suffix = "_per_issue_active.ratio"
            old_prefix = "smsp__warps_issue_stalled_"
            old_suffix = ".avg.pct_of_peak_sustained_elapsed"
            value = _optional_value_for_metric(metric_values, metrics, metric)
            if value is None:
                continue
            if metric.startswith(new_prefix) and metric.endswith(new_suffix):
                reason = metric[len(new_prefix) : -len(new_suffix)]
                out[f"stallr_{reason}"] = value
                stallr_total += value
            elif metric.startswith(old_prefix) and metric.endswith(old_suffix):
                reason = metric.removeprefix(old_prefix).removesuffix(old_suffix)
                out[f"stall_{reason}_pct"] = value
        if stallr_total:
            out["stallr_total"] = stallr_total

        sol_values = [
            out[key]
            for key in ("sol_sm_pct", "sol_compute_memory_pct", "sol_dram_pct")
            if key in out
        ]
        if sol_values:
            out["sol_max_pct"] = max(sol_values)
        return out

    return derive_diagnostics


def _roofline_metric(metrics: list[str]):
    """Hierarchical fp32 roofline: achieved (GFLOP/s, per-level AI) + the per-chip
    peak ceilings NCU derives, so the result carries everything the roofline plot
    needs without any datasheet constants."""

    def derive_roofline(*args):
        metric_values = args[: len(metrics)]

        def g(name):
            return _optional_value_for_metric(metric_values, metrics, name)

        out = {}
        fadd = g("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") or 0.0
        fmul = g("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") or 0.0
        ffma = g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") or 0.0
        flops = fadd + fmul + 2.0 * ffma
        out["fp32_flops"] = flops

        time_ns = g("gpu__time_duration.sum")
        if time_ns:
            out["gflops_per_s"] = flops / (time_ns * 1e-9) / 1e9

        # Achieved bytes per memory level (same counters NCU's roofline uses:
        # L1 lsu-writeback cycles x 128 B, L2 lts2xbar cycles x 32 B, HBM dram).
        l1_active = g("l1tex__lsu_writeback_active_mem_lg.sum")
        l2_active = g("lts__lts2xbar_cycles_active.sum")
        bytes_by_level = {
            "l1": l1_active * 128.0 if l1_active is not None else None,
            "l2": l2_active * 32.0 if l2_active is not None else None,
            "hbm": g("dram__bytes.sum"),
        }
        for level, nbytes in bytes_by_level.items():
            if nbytes:
                out[f"bytes_{level}"] = nbytes
                out[f"ai_{level}_flop_per_byte"] = flops / nbytes

        # Peak ceilings from NCU's per-chip .peak_sustained rollups (units/cycle)
        # x the level clock (.per_second). Architecture-correct, no hardcoding.
        ffma_peak = g("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")
        sm_hz = g("sm__cycles_elapsed.avg.per_second")
        if ffma_peak is not None and sm_hz is not None:
            out["peak_compute_gflops"] = ffma_peak * 2.0 * sm_hz / 1e9
        for level, peak_name, clock_name, width in (
            (
                "l1",
                "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained",
                "l1tex__cycles_elapsed.avg.per_second",
                128.0,
            ),
            (
                "l2",
                "lts__lts2xbar_cycles_active.sum.peak_sustained",
                "lts__cycles_elapsed.avg.per_second",
                32.0,
            ),
            ("hbm", "dram__bytes.sum.peak_sustained", "dram__cycles_elapsed.avg.per_second", 1.0),
        ):
            peak = g(peak_name)
            clock = g(clock_name)
            if peak is not None and clock is not None:
                out[f"peak_{level}_gbps"] = peak * width * clock / 1e9
        return out

    return derive_roofline


def _full_metric(metrics: list[str]):
    """Roofline + per-particle/atomics/occupancy/scheduler diagnostics together."""
    roofline = _roofline_metric(metrics)
    diagnostics = _diagnostic_metric(metrics)

    def derive_full(*args):
        out = {}
        out.update(roofline(*args))
        out.update(diagnostics(*args))
        return out

    return derive_full


def _derive_metric(name, metrics: list[str]):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError(
            "nsight.analyze.derive_metric must be null or a supported preset name."
        )
    if name in {"throughput", "p2g_throughput"}:
        if "gpu__time_duration.sum" not in metrics:
            raise RuntimeError(
                "derive_metric='throughput' requires "
                "nsight.analyze.metrics=[gpu__time_duration.sum, ...]."
            )
        return _p2g_throughput_metric(metrics)
    if name in {"speed_of_light", "sol"}:
        missing = [
            metric for metric in _SPEED_OF_LIGHT_METRICS if metric not in metrics
        ]
        if missing:
            raise RuntimeError(
                "derive_metric='speed_of_light' requires these nsight.analyze.metrics: "
                + ", ".join(missing)
            )
        return _speed_of_light_metric(metrics)
    if name == "roofline":
        missing = [m for m in _ROOFLINE_METRICS if m not in metrics]
        if missing:
            raise RuntimeError(
                "derive_metric='roofline' requires these nsight.analyze.metrics: "
                + ", ".join(missing)
            )
        return _roofline_metric(metrics)
    if name == "full":
        if "gpu__time_duration.sum" not in metrics:
            raise RuntimeError(
                "derive_metric='full' requires nsight.analyze.metrics with "
                "gpu__time_duration.sum (use metric_preset=full)."
            )
        return _full_metric(metrics)
    if name in {
        "diagnostics",
        "p2g_diagnostics",
        "memory_locality",
        "atomics",
        "occupancy",
        "scheduler",
    }:
        if "gpu__time_duration.sum" not in metrics:
            raise RuntimeError(
                f"derive_metric={name!r} requires "
                "nsight.analyze.metrics=[gpu__time_duration.sum, ...]."
            )
        return _diagnostic_metric(metrics)
    raise RuntimeError(
        f"Unsupported nsight.analyze.derive_metric={name!r}; "
        "supported presets: throughput, speed_of_light, diagnostics"
    )


def _normalize_metric_preset_names(*values):
    names = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            names.append(value)
            continue
        if isinstance(value, ListConfig | list | tuple):
            names.extend(str(item) for item in value)
            continue
        raise RuntimeError(
            "nsight.analyze.metric_preset(s) must be a string or list of strings."
        )
    return names


def _dedupe_metrics(metrics):
    deduped = []
    for metric in metrics:
        metric = str(metric)
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def _expand_metric_presets(kwargs):
    preset_names = _normalize_metric_preset_names(
        kwargs.pop("metric_preset", None),
        kwargs.pop("metric_presets", None),
    )
    explicit_metrics = list(kwargs.get("metrics") or [])
    if not preset_names:
        kwargs["metrics"] = _dedupe_metrics(
            explicit_metrics or ["gpu__time_duration.sum"]
        )
        return

    metrics = []
    for preset_name in preset_names:
        if preset_name not in _METRIC_PRESETS:
            supported = ", ".join(sorted(_METRIC_PRESETS))
            raise RuntimeError(
                f"Unsupported nsight.analyze metric preset {preset_name!r}; "
                f"supported presets: {supported}"
            )
        metrics.extend(_METRIC_PRESETS[preset_name])
    metrics.extend(explicit_metrics)
    kwargs["metrics"] = _dedupe_metrics(metrics)


def _combine_kernel_metrics(name):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError(
            "nsight.analyze.combine_kernel_metrics must be null or a preset name."
        )
    if name in {"sum", "add"}:
        return lambda x, y: x + y
    if name == "max":
        return max
    if name == "min":
        return min
    raise RuntimeError(
        f"Unsupported combine_kernel_metrics={name!r}; supported presets: sum, max, min"
    )


def _nsight_analyze_kwargs(cfg: DictConfig, run_dir: Path, profile_config):
    analyze_cfg = cfg.nsight.get("analyze", {})
    kwargs = OmegaConf.to_container(analyze_cfg, resolve=True)
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise RuntimeError(
            "nsight.analyze must be a mapping of nsight.analyze.kernel options."
        )

    unsupported = _UNSUPPORTED_ANALYZE_CONFIG_KEYS.intersection(kwargs)
    if unsupported:
        keys = ", ".join(sorted(unsupported))
        raise RuntimeError(
            "The Hydra nsight.analyze block only supports YAML-serializable "
            f"nsight.analyze.kernel options; unsupported keys: {keys}."
        )

    kwargs = dict(kwargs)
    kwargs.setdefault("runs", 1)
    kwargs.setdefault("replay_mode", "range")
    _expand_metric_presets(kwargs)
    kwargs["derive_metric"] = _derive_metric(
        kwargs.get("derive_metric"), kwargs["metrics"]
    )
    kwargs["combine_kernel_metrics"] = _combine_kernel_metrics(
        kwargs.get("combine_kernel_metrics")
    )
    kwargs.setdefault("output", "progress")
    kwargs.setdefault("output_csv", True)
    backend_choice = profile_config[0]
    target_name = str(cfg.nsight.get("target", "p2g"))
    kwargs.setdefault(
        "output_prefix", str(run_dir / f"nsight_{backend_choice}_{target_name}_")
    )
    # One config per Hydra job; sweep across backends/sizes via Hydra multirun.
    kwargs["configs"] = [profile_config]
    return kwargs


# ---------------------------------------------------------------------------
# Result writing (mirrors simulate.py: per-run files + a sweep-level results.csv).
# ---------------------------------------------------------------------------
def _py(value):
    """Coerce numpy/pandas scalars to JSON/CSV-friendly Python scalars."""
    item = value.item() if hasattr(value, "item") else value
    return item


def _flatten_results(results, cfg: DictConfig, profile_config, target_name):
    """nsight long-format DataFrame (one row per Metric) -> one wide metrics dict."""
    df = results.to_dataframe()
    first = df.iloc[0]
    metrics = {
        "backend": str(first["variant_backend"]),
        "target": target_name,
        "n_particles": int(profile_config[1]),
        "num_grids": int(profile_config[2]),
        "steps_per_frame": int(profile_config[3]),
        "kernel": str(first.get("Kernel", "")),
        "gpu": str(first.get("GPU", "")),
        "gpu_kind": _slugify(first.get("GPU", "")),
        "host": str(first.get("Host", "")),
        "compute_clock": _py(first.get("ComputeClock")),
        "memory_clock": _py(first.get("MemoryClock")),
    }
    for _, row in df.iterrows():
        metrics[str(row["Metric"])] = _py(row["AvgValue"])
    return metrics


def _analysis_csv_path(cfg: DictConfig):
    """Sweep-level results.csv path for a multirun (None for a single run)."""
    hydra_cfg = HydraConfig.get()
    if OmegaConf.select(hydra_cfg, "job.num") is None:
        return None
    sweep_dir = Path(str(OmegaConf.select(hydra_cfg, "sweep.dir")))
    if not sweep_dir.is_absolute():
        sweep_dir = Path(hydra.utils.get_original_cwd()) / sweep_dir
    try:
        runs_index = sweep_dir.parts.index("runs")
        sweep_root = Path(*sweep_dir.parts[:runs_index])
    except ValueError:
        sweep_root = sweep_dir
    return sweep_root / "results.csv"


def _write_csv_row(path: Path, metrics: Mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def _append_aggregate_row(path: Path, metrics: Mapping):
    """Append one wide row to the sweep results.csv, unioning columns across
    backends (different kernels emit slightly different derived metrics)."""
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([metrics])
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False)


def _write_results(results, cfg, run_dir, profile_config, target_name):
    metrics = _flatten_results(results, cfg, profile_config, target_name)

    hydra_cfg = HydraConfig.get()
    metrics["output_dir"] = str(run_dir)
    metrics["hydra_job_num"] = OmegaConf.select(hydra_cfg, "job.num")
    metrics["hydra_override_dirname"] = OmegaConf.select(
        hydra_cfg, "job.override_dirname"
    )

    if bool(cfg.nsight.get("write_json", True)):
        (run_dir / "nsight_metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_csv_row(run_dir / "nsight_metrics.csv", metrics)

    aggregate = _analysis_csv_path(cfg)
    if aggregate is not None:
        _append_aggregate_row(aggregate, metrics)
        logger.info("Appended %s row to %s", metrics["backend"], aggregate)

    logger.info("Nsight metrics for %s:\n%s", metrics["backend"], metrics)
    return metrics


def _run_nsight_profile(profiled_func):
    try:
        return profiled_func()
    except Exception as exc:
        if "ERR_NVGPUCTRPERM" in str(exc):
            raise RuntimeError(
                "Nsight Compute denied access to GPU performance counters "
                "(ERR_NVGPUCTRPERM). Enable NVIDIA performance counter access "
                "for this host/user, then rerun this script. See "
                "https://developer.nvidia.com/ERR_NVGPUCTRPERM"
            ) from exc
        raise


@hydra.main(version_base=None, config_path="conf", config_name="nsight_profile")
def main(cfg: DictConfig):
    nsight = _require_nsight()
    unexpected = set(cfg.nsight.keys()) - _SCRIPT_NSIGHT_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unknown nsight config keys: {keys}.")

    backend_choice = _backend_choice_from_cfg(cfg)
    target_name = str(cfg.nsight.get("target", "p2g"))
    profile_config = _profile_config(cfg, backend_choice)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    analyze_kwargs = _nsight_analyze_kwargs(cfg, run_dir, profile_config)

    launcher = _profile_runner(cfg, nsight)

    def profiled_variant(variant_backend, n_particles, num_grids, steps_per_frame):
        launcher()

    profiled_variant = nsight.analyze.kernel(**analyze_kwargs)(profiled_variant)

    logger.info(
        "Nsight profile: backend=%s target=%s config=%s",
        backend_choice,
        target_name,
        profile_config,
    )
    results = _run_nsight_profile(profiled_variant)
    # In an NCU child re-exec, a non-matching Hydra-multirun job returns None
    # (the matching job os._exit(0)s inside NCU). Only the parent writes.
    if results is not None:
        _write_results(results, cfg, run_dir, profile_config, target_name)


if __name__ == "__main__":
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
