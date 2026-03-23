# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unified MLS-MPM benchmark comparing JAX, PyTorch, and hand-written CUDA implementations. The PyTorch version lives in a git submodule (`vendor/MPM-PyTorch`), the JAX version is native to this repo (`mpm_jax/`).

## Build & Run Commands

```bash
# Install with both backends
uv run --python 3.11 --extra all python simulate.py

# JAX backend (default)
python simulate.py backend=jax material=jelly

# PyTorch backend
python simulate.py backend=pytorch material=jelly

# Override params
python simulate.py backend=jax sim.num_frames=50 sim.dt=1e-4

# Run CUDA P2G benchmark
python -m mpm_jax.cuda.benchmark

# Run tests
python -m pytest tests/ -v
```

## Architecture

### Unified Driver (`simulate.py`)

`@hydra.main` entry point. Selects backend via `cfg.backend.name` ("jax" or "pytorch"), delegates to `run_jax()` or `run_pytorch()`. Both produce a list of numpy frame arrays, then shared visualization code renders the GIF.

### JAX Backend (`mpm_jax/`)

Pure functional JAX. Solver split into individually-callable `compute_weights_and_indices`, `p2g`, `grid_update`, `g2p`. The `step()` orchestrator accepts an optional `p2g_fn` to swap in the CUDA kernel.

### PyTorch Backend (`vendor/MPM-PyTorch/`)

Git submodule from `github.com/philipnickel/MPM-PyTorch` (fork with warp-lang fix). Added to `sys.path` at runtime.

### Hydra Config (`conf/`)

Three config groups: `backend/` (jax or pytorch), `material/` (jelly or sand), `sim/` (simulation params). Compose from CLI.

### CUDA Benchmark (`mpm_jax/cuda/`)

Hand-written P2G kernel in `p2g_kernel.cu`, wrapped via PyCUDA in `p2g_custom_op.py`. Benchmark script compares JAX XLA vs CUDA timings.

## Submodule

After cloning, init the submodule: `git submodule update --init --recursive`
