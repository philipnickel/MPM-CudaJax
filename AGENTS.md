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
- NVIDIA cuTile kernels called from the shared JAX-owned frame loop.

The main entry points are:

- `simulate.py` - Hydra entry point for simulation, benchmark timing, and GIF
  rendering.
- `profile_nsight.py` - Nsight Python profiling for per-stage and per-kernel
  analysis.
- `src/mpm_jax/solver.py` - Equinox-based `MPMSolver`.
- `src/mpm_jax/backends/` - Hydra-registered backend implementations.
- `src/mpm_jax/grid.py`, `sort.py` - pure-math helpers (grid update; Morton +
  super-cell sorting).
- `src/mpm_jax/cutile_p2g.py` - cuTile tiled kernel bridge helpers.
- `src/mpm_jax/cuda/` - nanobind-backed JAX FFI CUDA registration plus CUDA
  kernel sources.
- `conf/` - Hydra config groups for simulation, materials, kernels, profiling,
  and sweeps.
- `tests/` - pytest coverage for solver behavior, CUDA equivalence, backends,
  cuTile paths, boundaries, and constitutive models.

## Working Rules

- Use `pixi` for installs, tests, linting, and running commands. Do not use
  system Python or direct `pip install` workflows.
- Prefer `pixi run test` for the full test suite and plain `pixi run ...` for
  GPU/CUDA/cuTile runs. The default Pixi environment is the GPU environment.
- CUDA kernels are built as an importable nanobind extension through
  scikit-build-core and CMake during `pixi install`.
- Kernel selection is Hydra-target-driven. Backend implementation modules under
  `src/mpm_jax/backends/` register their own Hydra config-group choices via
  hydra-zen; do not add dispatch logic to `simulate.py`.
- Preserve the benchmark methodology described in `.claude/CLAUDE.md`,
  especially the standard 8-particles-per-cell setup and the
  `profile_nsight.py` / Nsight Compute profiling workflow.

## Common Commands

```bash
pixi install
pixi run test
pixi run lint
pixi run python simulate.py sim.num_frames=5
pixi run python simulate.py sim=benchmark render.enabled=false
pixi run python simulate.py backend=jax
pixi run python simulate.py backend=cuda_v3 material=jelly
pixi run python simulate.py -cn sweep_quick
```

For fuller command examples, kernel descriptions, environment details, and the
current benchmark caveats, read [`.claude/CLAUDE.md`](.claude/CLAUDE.md).
