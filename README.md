# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels and an **NVIDIA cuTile** (tiled
programming model) kernel. Investigates where JAX/XLA's automatic GPU
compilation is sufficient and where custom kernels win.

The solver is **constructed from config** by Hydra-instantiating `cfg.solver`
into a `RuntimeConfig` and passing it to `MPMSolver`: the `backend` config
targets the backend class directly, and `MPMSolver` builds params, particles,
boundaries, and initial state. `solver.step()` advances one frame;
`solver.solve(num_frames, on_frame=...)` runs the full simulation with an
optional IO callback.

## Quickstart

You need [pixi](https://pixi.sh/) and an NVIDIA GPU on Linux. Everything else
(Python, JAX, CUDA toolkit deps) is pinned in `pyproject.toml` and `pixi.lock`
and managed by pixi — do **not** run `pip install` directly.

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

Install the default GPU environment and run a short simulation:
```bash
pixi install
pixi run python simulate.py sim.num_frames=20
```
A jelly block falls onto a sticky floor and renders to
`render.gif` inside the Hydra run directory. With `sim.num_frames=20` it takes
a few seconds.

The default environment also builds the custom CUDA kernels via CMake. `nvcc`
and `gxx` ship from conda-forge inside the env, no system module load needed:
```bash
pixi install
pixi run python simulate.py backend=cuda_v3 material=jelly
```

To benchmark instead of rendering:
```bash
pixi run python simulate.py \
    backend=cuda_v3 material=jelly \
    sim=benchmark render.enabled=false
```
Prints `total_steps`, `elapsed_s`, `steps_per_sec`, and average
`ms/step`. No GIF, no per-frame state capture — just wall-clock timing.

Outputs:
- GIF renders, `results.json`, Hydra logs, config snapshots → `outputs/<date>/<run>/`
- Multirun sweep results → `multirun/<date>/<run>/`
- Native CUDA extension → `mpm_jax.cuda._p2g_ffi` (rebuilds on native-source edit via `editable.rebuild=true`)

## Setup

Requires [pixi](https://pixi.sh/).

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**Default GPU environment:**
```bash
pixi install
pixi run python simulate.py sim.num_frames=5
```

CUDA kernels are built by [scikit-build-core](https://scikit-build-core.readthedocs.io/)
+ CMake during `pixi install` into one nanobind extension module:
`mpm_jax.cuda._p2g_ffi`. `p2g_cuda.py` imports that module, gets PyCapsule
handlers for the CUDA FFI targets, and registers them with
`jax.ffi.register_ffi_target`.

Override the CUDA architecture at install time:
```bash
MPM_CUDA_ARCH=sm_86 pixi install     # Ampere
MPM_CUDA_ARCH=sm_90 pixi install     # Hopper
# default is 'native' (CMake auto-detects the local GPU)
```

**DTU HPC:** no `module load` is needed for this Pixi environment — conda-forge
ships `cuda-nvcc`, `gxx`, and CUDA runtime libraries inside the default env.

**Warp 1.14:** `warp-lang==1.14.0` is kept in the default env for the optional
Warp OpenGL renderer (`warp.render`); it is no longer used for any P2G kernel.
The `glibc 2.34` system-requirement lets both the
`manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64
wheel (H100/A100) resolve correctly.

## Usage

```bash
# Default run (renders GIF in the Hydra run directory)
pixi run python simulate.py

# Timing run (no GIF, no per-frame state capture)
pixi run python simulate.py sim=benchmark render.enabled=false

# Pick a kernel
pixi run python simulate.py backend=jax                              # JAX/XLA baseline (scan P2G + MLS G2P)
pixi run python simulate.py backend=cuda_v1 material=jelly
pixi run python simulate.py backend=cuda_v2 material=jelly            # warp-shuffle coalescing
pixi run python simulate.py backend=cuda_v3 material=jelly            # Morton sort
pixi run python simulate.py backend=cuda_v4 material=jelly            # super-cell grid tile
pixi run python simulate.py backend=cutile material=jelly sim=benchmark render.enabled=false  # cuTile (tiled model)

# Override sim params
pixi run python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

## Kernel variants

| `backend=` | What it does |
|---|---|
| `jax` | The JAX/XLA baseline: `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), closed-form StVK stress. Every other kernel reuses this G2P, so only the P2G varies. |
| `cuda_v1` | CUDA inline-weight P2G (one thread/particle, global `atomicAdd`) + JAX baseline G2P. |
| `cuda_v2` | CUDA warp-shuffle coalesced inline P2G + JAX baseline G2P. |
| `cuda_v3` | CUDA Morton-sorted inline P2G + JAX baseline G2P. (XLA command-buffer / CUDA-Graph capture is on for every kernel via the default env's `XLA_FLAGS`.) |
| `cuda_v4` | CUDA super-cell-owned grid tile inline P2G + JAX baseline G2P. |
| `cutile` | NVIDIA cuTile (tiled programming model) P2G + JAX baseline G2P: SPGrid-style arena scatter — sort by SC=2 home super-cell, reduce each super-cell into a 4³ L1 arena, write back with one tile-coalesced `atomic_store_add` (no coloring). Requires `cuda-tile`. |

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (StVK) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

The solver is class-based:

- **`MPMSolver`** is a plain Python class. Particle/grid state is mutated in place by the driver API; the backend, constitutive closure, sticky-floor mask, and the compiled `_frame` are fixed for the solver's lifetime. `stepped()` returns a new solver with advanced state (shallow copy + new state); `step()` is the mutating driver and advances one frame by running `_frame(self.state)`. The frame runs `steps_per_frame` substeps as a single XLA program via `lax.fori_loop`.

Construction (`RuntimeConfig` + `MPMSolver` in `src/mpm_jax/solver.py`):
- Hydra instantiates `cfg.solver` into `RuntimeConfig`; backend choices are Python-backed hydra-zen registrations in `src/mpm_jax/backends/`, with each backend passing `num_grids` for validation. `simulate.py` / `profile_nsight.py` import `mpm_jax.backends` before composition, then call `MPMSolver(hydra.utils.instantiate(cfg.solver))`.
- `MPMSolver` reads the runtime config and builds params (with derived dx/vol/p_mass), particles, and initial state. The backend object is already instantiated by Hydra and owns CUDA/cuTile registration and grid-divisibility validation; the sticky floor is fixed in the solver frame.
- `src/mpm_jax/backends/` is a small backend hierarchy (base = `jax`; one subclass per variant overriding `prepare()`/`p2g()`, with `g2p()` shared on the base). The implementation modules register the user-facing Hydra choices (`jax`, `cuda_v1`, etc.) directly via hydra-zen. The frame loop calls `backend.step()` (order + scatter) and `backend.g2p()`.

All solver variants now run through the same JAX-owned frame loop. The pure-JAX path compiles the entire frame (multiple substeps) as one XLA program. The CUDA variants (`cuda_v*`) move per-particle stencil work into CUDA kernels so the `(N, 27, *)` intermediate tensors never materialize in HBM. The cuTile variant (`cutile`) launches a tiled-programming-model P2G kernel from inside that same JAX frame via the cuTile/JAX bridge.

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
pixi run python simulate.py -cn sweep_baseline    # JAX-only scaling
pixi run python simulate.py -cn sweep_all
pixi run python simulate.py -cn sweep_quick
pixi run python simulate.py -cn sweep_scaling
```

Each combination gets its own `multirun/<date>/<run>/` subdir with a `results.json`.
Runs with `render.enabled=true` also place `render.gif` in that same run
directory and record its path as `render_path`. Sweeps should use Hydra
multirun so log parsers see the expected directory structure.

For an ad-hoc sweep: `pixi run python simulate.py -m sim.n_particles=5000,50000,200000 backend=jax,cuda_v2 render.enabled=false`.

## Profiling

Use the dedicated Nsight Python entrypoint for per-stage kernel analysis:

```bash
pixi run python profile_nsight.py -cn nsight_profile \
    backend=jax material=jelly nsight.phase=p2g sim.n_particles=4096
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `jelly` (default) | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `backend` | `jax` (default), `cuda_v1`, `cuda_v2`, `cuda_v3`, `cuda_v4`, `cutile` | P2G implementation (G2P shared) |

Top-level fields: `tag`, `render`. All overridable from CLI:

```bash
pixi run python simulate.py sim.n_particles=100000 backend=cuda_v3 render.enabled=false
```

XLA command-buffer / CUDA-Graph capture is always on via `XLA_FLAGS` in the default env — no per-kernel flag.

## Tests

```bash
pixi run test
```

Run focused GPU checks:

```bash
pixi run pytest tests/test_cuda_ffi_loader.py tests/test_p2g_scan.py \
    tests/test_cuda_v2_inline_matches_v1.py -q
```


## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry + benchmark + GIF rendering
├── profile_nsight.py        # Nsight Python P2G profiler
├── pyproject.toml           # scikit-build-core build + default Pixi GPU env
├── pixi.lock                # locked deps (commit this)
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
├── conf/
│   ├── config.yaml
│   ├── nsight_profile.yaml
│   ├── material/            # jelly.yaml
│   ├── sim/default.yaml
│   └── sweep_*.yaml
└── src/
    └── mpm_jax/
        ├── types.py         # MPMState, MPMParams
        ├── solver.py        # MPMSolver
        ├── constitutive.py  # StVK elastic stress (jelly material)
        ├── grid.py          # grid_update + build_grid_x
        ├── sort.py          # morton_argsort, home_super_cell_id
        ├── backends/        # backend implementations + hydra-zen registrations
        ├── cutile_p2g.py    # cuTile arena-scatter P2G kernel + jax bridge
        └── cuda/
            ├── p2g_cuda.py  # FFI capsule registration + kernel objects
            └── kernels/     # p2g_ffi_module.cc plus p2g_inline.cu,
                             # p2g_v2_inline.cu, p2g_v3_inline.cu,
                             # p2g_v4_inline.cu
```

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
