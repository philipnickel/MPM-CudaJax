# MPM-CudaJax Agent Guide

This is the Codex-facing companion to the existing Claude guide. Treat
[`.claude/CLAUDE.md`](.claude/CLAUDE.md) as the canonical, detailed source of
truth for project narrative, architecture, benchmark methodology, kernel
variants, commands, and known caveats. Keep this file short; when behavior or
workflow guidance changes, update `.claude/CLAUDE.md` first and mirror only the
essential Codex-specific notes here.

## Repository Overview

MPM-CudaJax is a 3D MLS-MPM benchmark and investigation project comparing:

- pure JAX/XLA solver paths,
- hand-written CUDA kernels registered through JAX FFI, and
- NVIDIA Warp kernels called from the shared JAX-owned frame loop.

The main entry points are:

- `simulate.py` - Hydra entry point for simulation, benchmark timing, and GIF
  rendering.
- `profile_nsight.py` - Nsight Python profiling for per-stage and per-kernel
  analysis.
- `src/mpm_jax/registry.py` - kernel registry and `build_solver(cfg)`.
- `src/mpm_jax/solver.py` - Equinox-based `MPMSolver`.
- `src/mpm_jax/backends.py` - shared backend interface and JAX-owned frame
  loop.
- `src/mpm_jax/blocks/` - pure math blocks for weights, P2G, G2P, grid update,
  SVD, sorting, and initialization.
- `src/mpm_jax/warp_p2g.py` - Warp tiled kernel bridge helpers.
- `src/mpm_jax/cuda/` - JAX FFI CUDA loading plus CUDA kernel sources.
- `conf/` - Hydra config groups for simulation, materials, kernels, profiling,
  and sweeps.
- `tests/` - pytest coverage for solver behavior, CUDA equivalence, registry,
  Warp paths, boundaries, and constitutive models.

## Working Rules

- Use `pixi` for installs, tests, linting, and running commands. Do not use
  system Python or direct `pip install` workflows.
- Prefer `pixi run test` for the full test suite and `pixi run -e gpu ...` for
  GPU/CUDA/Warp runs.
- CUDA kernels are built through scikit-build-core and CMake. CPU-only installs
  intentionally skip CUDA when `nvcc` is unavailable.
- Kernel selection is registry-driven. Add or change kernel variants through
  `src/mpm_jax/backends.py`, `src/mpm_jax/registry.py`, and the relevant
  config files rather than adding dispatch logic to `simulate.py`.
- Preserve the benchmark methodology described in `.claude/CLAUDE.md`,
  especially the standard 8-particles-per-cell setup and the JAX trace -> Nsight
  Compute -> `nsight-python` profiling order.

## Common Commands

```bash
pixi install
pixi install -e gpu
pixi run test
pixi run lint
pixi run python simulate.py sim.num_frames=5
pixi run -e gpu python simulate.py benchmark=true
pixi run -e gpu python simulate.py kernel=jax_baseline
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=sand_jacobi
pixi run -e gpu python simulate.py -cn sweep_quick
```

For fuller command examples, kernel descriptions, environment details, and the
current benchmark caveats, read [`.claude/CLAUDE.md`](.claude/CLAUDE.md).
