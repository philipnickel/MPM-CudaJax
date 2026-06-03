# Status — class refactor + Warp programming-model investigation (2026-06-02)

Branch `worktree-refactor-mpm-solver-class` (PR #5). This note captures the current
state and findings; it sits on top of the completed class-based refactor.

## What's done

**1. Class-based refactor (merged into this branch, earlier).**
`src/` layout; pure math in `blocks/`; per-variant frame builders in `stepping/`;
`registry.py` (`KERNELS` + `build_solver`) replaced the `if/elif` dispatch; `MPMSolver`
(stateful shell over a functional JAX core) + `WarpGraphSolver` (pure-Warp capture/replay).
See `docs/superpowers/specs/2026-06-02-mpm-solver-class-design.md`.

**2. Warp lineup pruned to fit the project narrative.**
The project arc is: JAX baseline → profile (canonical MPM bottlenecks) → custom CUDA via
FFI for the bottlenecks → investigate the **tiled programming model (Warp)** as an
alternative, which needs a *separate* pure-Warp solver. Accordingly:
- Dropped `warp_v2_tile` (tied `warp_v1`, tiling bought nothing).
- Dropped the **hybrid** `warp_v1_inline` / `warp_v3_supercell_tile` (Warp P2G inside the JAX
  frame) — they don't isolate the programming-model comparison.
- Added **`warp_baseline_graph`**: the fully-Warp *baseline* (simple per-particle
  atomic-scatter P2G, no super-cell sort) that mirrors the JAX baseline. This fills the gap
  so the pure-Warp track has baseline + tiled, paralleling the JAX→CUDA arc.

Current kernels (9): `jax`, `jax_v1_5`, `cuda_v1/v2/v3/v4_inline`, `warp_baseline_graph`,
`warp_bonus_graph`, `warp_bonus_v2_graph`.

**3. Methodology + benchmark protocol recorded in CLAUDE.md.**
Project narrative; 3-step analysis method (JAX trace → `ncu` roofline → `nsight-python`
sweep across variants/sizes/architectures); and the **standard benchmark** (Taichi/MLS-MPM):
8 ppc, dx=8e-3 (`num_grids=125`), particles in [0.1,0.9]³, **8M particles**, APIC + corotated
elasticity, with broad P2G/G2P phase definitions. Wired the particle region to `sim.size`
(was hardcoded 0.5) and added `conf/sim/benchmark.yaml`.

## Findings (RTX 5090, sm_120, Warp 1.14)

**Tiled pure-Warp (`warp_bonus_*`) only wins in a narrow band.** Cost ∝ number of
super-cells ≈ (G/2)³. At N=200K: G=16 → 2.1 ms/step, **G=32 → 0.39 (beats `cuda_v3`'s 0.68)**,
G=64 → 35–49 (catastrophic). Also requires an **even grid**, so it cannot run the standard
benchmark (`num_grids=125`, odd) — use 124/126 for those.

**`warp_baseline_graph` is robust across grid sizes** (no super-cell cliff) and is the
fastest at the well-resolved standard benchmark.

**Standard benchmark — 8M particles, 125³, 8 ppc, dt=5e-5 (CFL-safe), `jelly_jacobi`:**

| kernel | ms/step |
|---|---|
| `warp_baseline_graph` | **13.87** |
| `cuda_v3_inline` | 38.78 |
| `cuda_v1_inline` | 43.95 |
| `cuda_v2_inline` | 44.23 |
| `jax_v1_5` | 89.86 |
| `jax` | 103.65 |

**`ncu` on the baseline P2G** (G=64): global-atomic-**reduction**-bound. RED sectors =
exactly N×27×4 (108 reductions/particle: 27 stencil nodes × 3 momentum + 1 mass); compute
<1% SoL, DRAM <0.5%, L2 hit 97% → **latency-bound on L2 atomic throughput**, set by particle
clustering. Cost is monotonic in N at a fixed state (14 ms @1M → 24 ms @2M).

**Earlier wall-clock non-monotonicity was numerical instability, not a real effect.**
At G=64 with the old default `dt=3e-4`, the elastic CFL limit (c=√(E/ρ)≈45 m/s, dt_max≈9e-5)
was violated, and under-resolved counts (1M/64³ ≈ 3.8 ppc) exploded → particles clamp to the
bounds and cluster → worst-case atomic contention → erratic timings. The 8-ppc / CFL-safe
benchmark removes this.

**Fairness check (why `warp_baseline` wins — NOT dispatch overhead).** The JAX path is fully
jitted: `build_backend_frame` compiles the whole frame (10 substeps) into one XLA program
(`lax.fori_loop`), and benchmark mode dispatches frames back-to-back with one sync — so host
dispatch is already eliminated on the JAX side. CUDA-path decomposition at the benchmark:
`cuda_v1`=43.95, `cuda_v2`=44.23, `cuda_v3`=38.78, `cuda_v3 cuda_graph=true`=38.86. So
warp-shuffle (v2) doesn't help, Morton sort (v3) buys only ~12% at 8 ppc, and XLA CUDA-graph
capture is a no-op. The real causes of the gap: vs `jax`, avoiding the `(N,27,*)`
materialization (the canonical bottleneck); vs `cuda_v3`, no per-substep Morton `argsort` and
no FFI marshalling. The residual graph-vs-XLA-launch difference is minor at 8M.

## Open items / next

- **Rigorous per-kernel attribution** across all variants via `ncu` (roofline/atomic/L2) and
  then `nsight-python` sweeps over (N, G) at fixed 8 ppc and across architectures (A100/H100/
  GH200) — steps 2–3 of the analysis method. The wall-clock numbers above are directional.
- **`warp_bonus_*` even-grid limitation:** benchmark them at `num_grids=124/126`, and consider
  retuning `SUPER_CELL_WIDTH` (currently 2) to reduce the G=64 cliff.
- Confirm the standard benchmark at 8M is non-degenerate for the pure-Warp path (state evolves
  correctly under graph replay) before treating its numbers as final.
