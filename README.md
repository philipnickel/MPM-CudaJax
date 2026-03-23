# MPM-CudaJax

Unified MLS-MPM (Moving Least Squares Material Point Method) benchmark comparing **JAX**, **PyTorch**, and hand-written **CUDA** implementations.

A single Hydra-configured entry point lets you switch backends from the command line.

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
uv run --python 3.11 --extra all python simulate.py backend=jax sim.num_frames=300 sim.dt=1e-4
```

Output GIFs are saved to `output/`.

### Run tests

```bash
uv run --python 3.11 --extra jax python -m pytest tests/ -v
```

### Run CUDA P2G benchmark

Requires a CUDA-capable GPU and `pycuda`:

```bash
uv run --python 3.11 --extra jax python -m mpm_jax.cuda.benchmark
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|-------|---------|-------------|
| `backend` | `jax` (default), `pytorch` | Simulation backend |
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | Simulation parameters + boundary conditions |

All parameters are overridable from the CLI, e.g. `sim.num_grids=50 sim.rho=2000`.

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Unified Hydra entry point
├── conf/                    # Hydra config groups
│   ├── backend/             #   jax.yaml, pytorch.yaml
│   ├── material/            #   jelly.yaml, sand.yaml
│   └── sim/                 #   default.yaml
├── mpm_jax/                 # JAX implementation
│   ├── solver.py            #   P2G, G2P, grid update (individually jittable)
│   ├── constitutive.py      #   5 elasticity + 4 plasticity models
│   ├── boundary.py          #   6 boundary condition types
│   └── cuda/                #   Hand-written CUDA P2G kernel + benchmark
├── vendor/MPM-PyTorch/      # PyTorch implementation (git submodule)
└── tests/                   # 24 tests
```

## References

- Hu et al., "A Moving Least Squares Material Point Method with Displacement Discontinuity and Two-Way Rigid Body Coupling", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
