"""Canonical NCU metric presets for the ``nsight_metrics`` Hydra group.

Single source of truth for the metric lists that ``profile_nsight.py`` hands to
Nsight Compute. The ``full`` preset is *composed* from the roofline subset plus
the atomics / occupancy / scheduler counters, so the roofline metrics are never
copy-pasted between presets (the previous ``conf/nsight_metrics/{roofline,full}``
YAML duplicated them by hand).

Importing this module registers ``nsight_metrics={timing,roofline,full}`` as a
Hydra config-group selection whose content lands under ``nsight.analyze`` —
equivalent to the old ``# @package nsight.analyze`` YAML stubs. ``profile_nsight``
imports it before Hydra composes; there is no ``conf/nsight_metrics/`` directory.

Roofline entries follow Yang's Nsight Compute 2020 metric set
(arXiv:2009.02449, Table III), with peak counters for the reference roofs.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

# Kernel duration only (smoke-test + fallback).
TIMING_METRICS = [
    "gpu__time_duration.sum",
]

# Roofline counters + peak-sustained roofs.
ROOFLINE_METRICS = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "l1tex__t_bytes.sum",
    "lts__t_bytes.sum",
    "dram__bytes.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "l1tex__t_bytes.sum.peak_sustained",
    "l1tex__cycles_elapsed.avg.per_second",
    "lts__t_bytes.sum.peak_sustained",
    "lts__cycles_elapsed.avg.per_second",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
]

# Extra atomics/occupancy/scheduler counters on top of roofline.
FULL_EXTRA_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red.ratio",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_red.sum.pct_of_peak_sustained_elapsed",
    "lts__d_atomic_input_cycles_active.sum.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__t_sector_hit_rate.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__maximum_warps_avg_per_active_cycle",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_static",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
]

# All NCU metrics.
FULL_METRICS = [*TIMING_METRICS, *ROOFLINE_METRICS, *FULL_EXTRA_METRICS]

METRIC_PRESETS = {
    "timing": TIMING_METRICS,
    "roofline": ROOFLINE_METRICS,
    "full": FULL_METRICS,
}


def register() -> None:
    """Register the metric presets as the ``nsight_metrics`` config group.

    Each preset is stored under package ``nsight.analyze`` so selecting
    ``nsight_metrics=<preset>`` yields ``nsight.analyze.metrics=[...]``.
    """
    cs = ConfigStore.instance()
    for name, metrics in METRIC_PRESETS.items():
        cs.store(
            group="nsight_metrics",
            name=name,
            package="nsight.analyze",
            node={"metrics": list(metrics)},
        )


register()
