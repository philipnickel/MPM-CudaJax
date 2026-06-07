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

### `cuda_v2`: Home-Sorted Warp-Coalesced Atomics

- `prepare`: `home_cell_order`.
- `scatter`: one CUDA thread per particle, compute the 27-node stencil, use
  warp-level matching/reduction for lanes targeting the same grid node, and have
  one elected lane issue the global atomic update.
- Purpose: isolate the benefit of home-cell ordering plus warp aggregation while
  keeping particle-owned global atomics.

### `cuda_v3`: Home-Cell Local Reduction

- `prepare`: `home_cell_order`.
- `scatter`: one CUDA block or cooperative group owns one home cell, loops over
  that cell's particles, accumulates the 27-node stencil locally, and flushes the
  reduced stencil to global memory.
- Purpose: change ownership from particle-owned scatter to cell-owned local
  reduction.

### `cuda_v4`: Optimized Home-Cell Local Reduction

- `prepare`: `home_cell_order`.
- `scatter`: same home-cell ownership as `cuda_v3`, but reduce shared-memory
  atomic pressure with structured warp/block reductions over stencil
  offsets/channels before the final global flush.
- Purpose: optimized hand-written CUDA implementation of the home-cell reduction
  idea.

### Reference/Comparison Backends

- `jax`: correctness and full-solver baseline. It is not the primary target for
  Nsight kernel-roofline comparisons because XLA scatter can be multi-kernel.
- `cutile_v3`: cuTile implementation of home-cell local reduction. Compare it
  primarily against `cuda_v3` and `cuda_v4` to separate algorithmic wins from
  code-generation wins.

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
