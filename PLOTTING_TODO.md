# Plotting TODO

Plots to add to `postprocessing/nsight_plots.py` from metrics collected by
`profile_nsight.py` (`metric_preset: full`, run across backends + N sweep).

- [ ] **Atomic throughput** — `l2_atomic_unit_pct`
      (`lts__d_atomic_input_cycles_active.sum.pct_of_peak_sustained_elapsed`).
      Saturation of the L2 atomic unit; the canonical contention metric for
      the P2G scatter.
- [ ] **Atomic replay** — `l2_red_throughput_pct` paired with
      `global_atomic_or_reduction_requests_per_particle` and
      `expected_p2g_float_atomic_ops_per_particle`. Shows how many of the
      issued atomics are getting replayed under contention.
- [ ] **Warp stall reasons** — stacked bar of the `stallr_<reason>` family
      (`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`),
      normalized so reasons sum to `stallr_total`. One bar per backend.
- [ ] **L2 hit rate** — `l2_hit_rate_pct` (`lts__t_sector_hit_rate.pct`),
      alongside `l1_hit_rate_pct` if useful as a companion line.

- [ ] Strong scaling plots using mps control for controlling SM percentage

- [ ] hiearchecal roofline plots (sweeping over say particle count or so)
- [ ] roofline trajectories (saborn relplot with cols for HBM, L1, L2 rooflines with ref-lines for SM Low/High (mps controlled))
