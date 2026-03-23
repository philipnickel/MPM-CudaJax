# MPM-CudaJax

Unified MLS-MPM (Moving Least Squares Material Point Method) benchmark comparing **JAX**, **PyTorch**, and hand-written **CUDA** implementations.

A single Hydra-configured entry point lets you switch backends and kernels from the command line.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone --recurse-submodules git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Build CUDA kernels (GPU nodes only)

```bash
cd mpm_jax/cuda/kernels && make && cd -
```

## Usage

All commands use `uv run` — no manual venv or `pip install` needed.

### Run a simulation

```bash
# JAX backend (default)
uv run --python 3.11 --extra all python simulate.py

# PyTorch backend
uv run --python 3.11 --extra all python simulate.py backend=pytorch

# Sand instead of jelly
uv run --python 3.11 --extra all python simulate.py material=sand

# Override simulation parameters
uv run --python 3.11 --extra all python simulate.py sim.num_frames=300 sim.dt=1e-4
```

### Benchmark mode (timing only, no GIF)

```bash
uv run --python 3.11 --extra all python simulate.py benchmark=true
```

### Switch P2G kernel (JAX backend only)

```bash
# Pure JAX (default)
uv run --python 3.11 --extra all python simulate.py kernel=jax

# CUDA v1: JAX compute + CUDA scatter (naive atomicAdd)
uv run --python 3.11 --extra all python simulate.py kernel=cuda_v1

# CUDA v2: fused stress + weights + scatter in one CUDA kernel
uv run --python 3.11 --extra all python simulate.py kernel=cuda_v2
```

### Sweep all backends and kernels (Hydra multirun)

```bash
# Compare JAX vs PyTorch
uv run --python 3.11 --extra all python simulate.py -m backend=jax,pytorch benchmark=true

# Sweep all P2G kernels (JAX backend)
uv run --python 3.11 --extra all python simulate.py -m kernel=jax,cuda_v1,cuda_v2 benchmark=true

# Full sweep: backends × materials × kernels
uv run --python 3.11 --extra all python simulate.py -m \
    backend=jax,pytorch \
    material=jelly,sand \
    kernel=jax,cuda_v1,cuda_v2 \
    benchmark=true
```

All runs log to [wandb](https://wandb.ai) project `MPM-CudaJAX` with per-frame stage timings.

### Run tests

```bash
uv run --python 3.11 --extra jax --with pytest python -m pytest tests/ -v
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|-------|---------|-------------|
| `backend` | `jax` (default), `pytorch` | Simulation backend |
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | Simulation parameters + boundary conditions |
| `kernel` | `jax` (default), `cuda_v1`, `cuda_v2` | P2G kernel implementation |

All parameters are overridable from the CLI, e.g. `sim.num_grids=50 sim.rho=2000`.

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Unified Hydra entry point
├── conf/                    # Hydra config groups
│   ├── backend/             #   jax.yaml, pytorch.yaml
│   ├── material/            #   jelly.yaml, sand.yaml
│   ├── sim/                 #   default.yaml
│   └── kernel/              #   jax.yaml, cuda_v1.yaml, cuda_v2.yaml
├── mpm_jax/                 # JAX implementation
│   ├── solver.py            #   vmap-based P2G, G2P, grid update
│   ├── constitutive.py      #   5 elasticity + 4 plasticity models
│   ├── boundary.py          #   6 boundary condition types
│   └── cuda/
│       ├── p2g_cuda.py      #   JAX FFI registration + Python wrappers
│       └── kernels/
│           ├── p2g_scatter.cu   # v1: naive scatter (atomicAdd only)
│           ├── p2g_fused.cu     # v2: fused stress+weights+scatter
│           └── Makefile
├── vendor/MPM-PyTorch/      # PyTorch implementation (git submodule)
└── tests/                   # 24 tests
```

## Architecture

The solver is structured around three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: compute stress (SVD), B-spline weights, affine momentum → scatter to grid (atomicAdd)
2. **Grid update** — per-node: normalize momentum, apply gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/deformation gradient

The per-particle functions are written as single-particle JAX functions and batched via `jax.vmap`. The P2G scatter (the reduction) is the only cross-particle operation and the primary CUDA optimisation target.

CUDA kernels integrate via `jax.ffi` (Foreign Function Interface) for zero-copy GPU memory access and proper CUDA stream integration.

## References

- Hu et al., "A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
