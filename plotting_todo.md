# Report plotting todo

This file tracks the agreed report figures and the data each one should collect. The
plotting layer can stay in pandas/seaborn; Nsight Python should mainly produce clean
CSV/JSON rows with useful metric columns.

## 1. Whole-solver particle throughput

- Scope: end-to-end solver.
- Question: how fast is the actual frame/substep loop for each backend?
- Source: `simulate.py benchmark=true` benchmark outputs or future benchmark CSV.
- X axis: problem size, usually particle count at 8 particles/cell.
- Y axis: `solver_mparticles_per_s = n_particles * num_substeps / elapsed_seconds`.
- Variant columns: `kernel`, `material`, `num_grids`, `n_particles`, `steps_per_frame`,
  `num_frames`, `cuda_graph` or `graph_mode`.
- Candidate data columns: `elapsed_seconds`, `ms_per_step`, `steps_per_second`,
  `solver_mparticles_per_s`.
- Notes: include the JAX baseline here because this is the user-visible solver result.

## 2. Isolated P2G throughput

- Scope: P2G-only Nsight Python stage.
- Question: because only P2G varies across variants, how much does the backend P2G
  implementation itself improve?
- Source: `profile_nsight.py` with the `time` preset and CSV output.
- X axis: particle count at fixed particles/cell, or particles/cell at fixed grid.
- Y axis: `p2g_mparticles_per_s`.
- Variant columns: `kernel`, `n_particles`, `num_grids`, `steps_per_frame`,
  `Annotation`, `Kernel`, `GPU`.
- Candidate data columns: `gpu__time_duration.sum`, `time_ms`,
  `p2g_mparticles_per_s`, `StdDev`, `CI95`, `StableMeasurement`.
- Notes: use additive aggregation for timing/count metrics when profiling multi-kernel
  annotated ranges.

## 3. Hierarchical roofline

- Scope: P2G-only, CUDA/Warp-oriented.
- Question: what hardware ceilings are the P2G variants approaching at L1/M1, L2/M2,
  and HBM?
- Source: `benchmarking/hiearchical_roofline.py`.
- X axis: arithmetic intensity per memory level.
- Y axis: achieved work rate.
- Variant columns: `kernel`, `n_particles`, `memory_level`.
- Candidate data columns: `flop_count`, `elapsed_seconds`, `bytes_l1`, `bytes_l2`,
  `bytes_hbm`, `ai_l1`, `ai_l2`, `ai_hbm`, `achieved_gflops`.
- Notes: roofline ceilings should be annotated; datapoints should not be annotated.

## 4. Speed-of-light utilization

- Scope: P2G-only Nsight Python stage.
- Question: how much of the GPU's SM and memory systems are used by each P2G variant?
- Source: `profile_nsight.py -cn nsight_p2g_sol`.
- Plot: grouped bars by kernel.
- Candidate raw metrics:
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  - `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- Candidate derived columns: `sol_sm_pct`, `sol_compute_memory_pct`, `sol_dram_pct`,
  `sol_max_pct`, `time_ms`, `p2g_mparticles_per_s`.
- Notes: prefer `replay_mode=range` for percentage metrics over multi-kernel annotated
  ranges. Do not sum percent-of-peak metrics.

## 5. Atomic/scatter pressure

- Scope: P2G-only Nsight Python stage.
- Question: where do particle-owned atomics and scatter entropy dominate performance?
- Source: `profile_nsight.py -cn nsight_p2g_contention`.
- Plot options:
  - bars: atomic/reduction requests per particle by variant
  - scatter: atomic/reduction pressure vs `p2g_mparticles_per_s`
  - bars: global sectors/request by operation
- Candidate raw metrics:
  - `l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum`
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum`
  - `l1tex__t_requests_pipe_lsu_mem_global_op_red.sum`
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum`
  - `dram__bytes.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum`
- Candidate derived columns: `global_atom_requests_per_particle`,
  `global_atom_sectors_per_particle`, `global_atom_sectors_per_request`,
  `global_red_requests_per_particle`, `global_red_sectors_per_particle`,
  `global_red_sectors_per_request`,
  `global_atomic_or_reduction_requests_per_particle`,
  `expected_p2g_float_atomic_ops_per_particle`, `dram_bytes_per_particle`.
- Notes: the naive particle-owned P2G expectation is 27 stencil nodes * 4 float
  atomic adds = 108 float atomics per particle.

## 6. Occupancy and scheduler stalls

- Scope: P2G-only Nsight Python stage.
- Question: are optimized variants limited by register/shared-memory pressure, low
  eligible warps, barriers, memory scoreboards, or scheduler issue behavior?
- Source: `profile_nsight.py -cn nsight_p2g_occupancy`.
- Plot options:
  - grouped bars: active warps, eligible warps/cycle, issue active
  - stacked bars: major stall reason percentages
- Candidate raw metrics:
  - `sm__warps_active.avg.pct_of_peak_sustained_active`
  - `smsp__warps_eligible.avg.per_cycle_active`
  - `smsp__issue_active.avg.pct_of_peak_sustained_active`
  - `smsp__warps_issue_stalled_*.avg.pct_of_peak_sustained_elapsed`
- Candidate derived columns: `active_warps_pct`, `eligible_warps_per_cycle`,
  `issue_active_pct`, `stall_barrier_pct`, `stall_long_scoreboard_pct`,
  `stall_lg_throttle_pct`, `stall_mio_throttle_pct`, `stall_not_selected_pct`,
  `stall_short_scoreboard_pct`, `stall_wait_pct`.
- Notes: scheduler metric names are the most Nsight-version-sensitive; validate with
  `ncu --query-metrics --query-metrics-mode all` on a new machine.

## 7. Particles-per-cell contention sweep

- Scope: P2G-only Nsight Python stage.
- Question: when does atomic contention or supercell/bin overhead change the ranking?
- Source: `profile_nsight.py -cn nsight_p2g_ppc_sweep`.
- X axis: particles per cell, computed as `n_particles / num_grids**3`.
- Y axes: `p2g_mparticles_per_s`, atomic/reduction requests per particle, and/or
  bytes per particle.
- Variant columns: `kernel`, `n_particles`, `num_grids`, computed `particles_per_cell`.
- Candidate data columns: all columns from the time and contention presets.
- Notes: this controlled sweep covers the main distribution-sensitivity question for
  the report. More exotic distributions can stay as appendix/future work.

## Data workflow

1. Keep Nsight Python collection native and Hydra-configured.
2. Use `nsight.analyze.metric_preset` or `nsight.analyze.metric_presets` to choose
   bundles such as `time`, `speed_of_light`, `memory_locality`, `atomics`,
   `occupancy`, and `scheduler`.
3. Always write CSV (`output_csv: true`) and optionally JSON (`write_json: true`).
4. Build final figures from collected CSVs using pandas/seaborn rather than relying
   on Nsight Python's simple native plotting for multi-metric report dashboards.
