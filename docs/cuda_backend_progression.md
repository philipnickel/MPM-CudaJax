# CUDA P2G Backend Progression

This branch restructures the hand-written CUDA P2G variants into a cleaner
algorithmic ladder. Each version should add one idea, and the Nsight story should
make clear whether a plot is measuring the scatter kernel, the full P2G stage, or
the full solver step.

## Final Backend Spec

### `cuda_v1`: Direct Atomics

- `prepare`: none beyond the default particle state packaging.
- `scatter`: one CUDA thread per particle, compute the 27-node MLS-MPM stencil,
  and issue direct global `atomicAdd` updates for grid momentum and mass.
- Purpose: baseline for particle-owned scatter with unsorted particles.

### `cuda_v2`: Morton-Sorted Warp-Coalesced Atomics

- `prepare`: `morton_order`.
- `scatter`: one CUDA thread per particle, compute the 27-node stencil, use
  warp-level matching/reduction for lanes targeting the same grid node, and have
  one elected lane issue the global atomic update.
- Purpose: isolate the benefit of Morton ordering plus warp aggregation while
  keeping particle-owned global atomics.

### `cuda_v3`: Super-Cell Local Reduction

- `prepare`: `supercell_order`.
- `scatter`: one CUDA block owns one super-cell, loops over that super-cell's
  particles, accumulates into a shared `(SC+2)^3` grid tile, and flushes the
  reduced tile to global memory.
- Purpose: change ownership from particle-owned scatter to super-cell-owned
  local reduction.

### Reference/Comparison Backends

- `jax`: correctness and full-solver baseline. It is not the primary target for
  Nsight kernel-roofline comparisons because XLA scatter can be multi-kernel.
- `CuTile`: cuTile implementation of home-cell local reduction. Compare it
  primarily against `cuda_v3` to separate algorithmic wins from code-generation
  wins.

## Profiling Methodology

- Scatter roofline: warmed `backend.scatter` kernel only. This is fair for
  scatter-kernel efficiency.
- P2G stage timing: `prepare + scatter`, ideally as stacked bars. This is fair
  for backend P2G cost.
- Solver benchmark: full frame/substep timing. This is fair for application
  performance.

## Validation Bar

Each CUDA backend variant must:

- match the JAX P2G reference on small randomized tests;
- match `cuda_v1` or JAX within expected floating-point tolerance at the benchmark
  operating point (`G=96`, `N=10M`);
- report one real scatter kernel in Nsight after ignoring known tiny cuTile/XLA
  wrappers, unless a test intentionally exercises a multi-kernel path;
- include benchmark-config timing for prepare, scatter, and full solver
  `ms/step`.
