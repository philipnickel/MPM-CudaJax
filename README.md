# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** and **Warp** kernels. Investigates where
JAX/XLA's automatic GPU compilation is sufficient and where custom kernels
win.

The solver uses a **registry-based class API**: `build_solver(cfg)` reads
`KERNELS[kernel.name]`, constructs an `MPMSolver`, and returns it ready to call.
`solver.step()` advances one frame;
`solver.solve(num_frames, on_frame=...)` runs the full simulation with an
optional IO callback.

## Quickstart

You need [pixi](https://pixi.sh/). Everything else (Python, JAX, CUDA
toolkit deps) is pinned in `pyproject.toml` and `pixi.lock` and managed
by pixi — do **not** run `pip install` directly.

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**No GPU?** Install the default (CPU) env and run a short simulation:
```bash
pixi install
pixi run python simulate.py sim.num_frames=20
```
A sand block falls onto a sticky floor and renders to
`output/sand_jacobi_jax_v1_5.gif`. With `sim.num_frames=20` it takes a few seconds.

**Have an NVIDIA GPU (Linux)?** Install the `gpu` env (this also builds
the custom CUDA kernels via CMake — `nvcc` and `gxx` ship from
conda-forge inside the env, no system module load needed):
```bash
pixi install -e gpu
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=sand_jacobi
```

To benchmark instead of rendering:
```bash
pixi run -e gpu python simulate.py \
    kernel=cuda_v3_inline material=sand_jacobi \
    sim.n_particles=500000 sim.num_grids=64 sim.num_frames=15 \
    benchmark=true
```
Prints `total_steps`, `elapsed_s`, `steps_per_sec`, and average
`ms/step`. No GIF, no per-frame state capture — just wall-clock timing.

Outputs:
- GIF renders → `output/<tag>_<kernel>.gif`
- Hydra logs / config snapshots → `outputs/<date>/<run>/`
- Multirun sweep results → `multirun/<date>/<run>/`
- Built CUDA `.so` files → `src/mpm_jax/cuda/_lib/` (rebuilds on `.cu` edit via `editable.rebuild=true`)

## Setup

Requires [pixi](https://pixi.sh/).

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**Local (CPU only):**
```bash
pixi install
pixi run python simulate.py sim.num_frames=5
```

**GPU (Linux):**
```bash
pixi install -e gpu        # builds CUDA kernels via CMake at install time
pixi run -e gpu python simulate.py
```

CUDA kernels are built by [scikit-build-core](https://scikit-build-core.readthedocs.io/)
+ CMake during `pixi install -e gpu`. Output `.so` files land in
`src/mpm_jax/cuda/_lib/` and are loaded at runtime via
`jax.ffi.register_ffi_target`. The build is best-effort: when `nvcc` is
missing (the default CPU env) CMake's `check_language(CUDA)` returns
early, the wheel installs cleanly, and the JAX baseline still works.

Override the CUDA architecture at install time:
```bash
MPM_CUDA_ARCH=sm_86 pixi install -e gpu     # Ampere
MPM_CUDA_ARCH=sm_90 pixi install -e gpu     # Hopper
# default is 'native' (CMake auto-detects the local GPU)
```

**DTU HPC:** no `module load` is needed for the `gpu` env — conda-forge ships `cuda-nvcc`
and `gxx` inside the `gpu` env. For clusters where the conda-forge CUDA
lags the driver, use the `hpc` env with `module load` instead:
```bash
# gpu env (self-contained):
MPM_CUDA_ARCH=sm_90 pixi install -e gpu

# hpc env (links against site-provided CUDA toolkit):
module load nvhpc/24.1
MPM_CUDA_ARCH=sm_90 pixi install -e hpc
pixi run -e hpc python simulate.py ...
```

**Warp 1.14:** The `gpu` env pins `warp-lang==1.14.0` from PyPI.
conda-forge only carries 1.13, which has a tile-kernel bug on
sm_120/Blackwell GPUs. The `glibc 2.34` system-requirement lets both the
`manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64
wheel (H100/A100) resolve correctly.

## Usage

```bash
# Default run (renders GIF to ./output)
pixi run -e gpu python simulate.py

# Benchmark mode (no GIF, no per-frame state capture, wall-clock timing)
pixi run -e gpu python simulate.py benchmark=true

# Pick a kernel
pixi run -e gpu python simulate.py kernel=jax_v1_5                                     # JAX/XLA baseline
pixi run -e gpu python simulate.py kernel=cuda_v1_inline material=sand_jacobi
pixi run -e gpu python simulate.py kernel=cuda_v2_inline material=sand_jacobi         # warp-shuffle (default: fori loop)
pixi run -e gpu python simulate.py kernel=cuda_v2_inline kernel.loop_kind=python material=sand_jacobi
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=sand_jacobi         # Morton sort
pixi run -e gpu python simulate.py kernel=cuda_v3_inline kernel.cuda_graph=true material=sand_jacobi
pixi run -e gpu python simulate.py kernel=warp_v3_supercell_tile material=sand_jacobi benchmark=true

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

## Kernel variants

| `kernel=` | What it does |
|---|---|
| `jax_v1_5` | Pure JAX/XLA baseline. P2G scans over the 27 stencil offsets (`lax.scan`) to avoid `(N, 27, *)` HBM intermediates. |
| `cuda_v1_inline` | Inline-weight CUDA P2G (one thread/particle, global `atomicAdd`) + CUDA G2P; no `(N, 27, *)` tensors. |
| `cuda_v2_inline` | Warp-shuffle coalesced inline CUDA P2G + CUDA G2P. Default `loop_kind=fori`. Override with `kernel.loop_kind=python`. |
| `cuda_v3_inline` | Morton-sorted inline CUDA P2G + CUDA G2P. `kernel.cuda_graph=true` enables XLA command-buffer (CUDA Graph) replay. |
| `cuda_v4_inline` | Super-cell-owned grid tile inline CUDA P2G + CUDA G2P. |
| `warp_v3_supercell_tile` | Hybrid JAX/Warp backend: JAX frame + stress/sort/grid/G2P orchestration, Warp `jax_callable` tiled P2G (`wp.launch_tiled` + `tile_scatter_add`). Default `kernel.graph_mode=jax`. |

Removed kernels (raise `ValueError` with a migration message):

| Old name | Replacement |
|---|---|
| `jax` | `jax_v1_5` |
| `cuda_v1`, `cuda_v2`, `cuda_v4` | `cuda_v1_inline`, `cuda_v2_inline`, `cuda_v4_inline` |
| `cuda_fused` | Removed; use an inline backend and `profile=jax` |
| `cuda_v2_fori_inline` | `kernel=cuda_v2_inline` (fori is the default) |
| `cuda_v3_fori_inline` | `kernel=cuda_v3_inline kernel.loop_kind=fori` |
| `cuda_v6_inline` | `kernel=cuda_v3_inline kernel.cuda_graph=true` |
| `warp_baseline_graph`, `warp_bonus_graph`, `warp_bonus_v2_graph` | Removed pure-Warp solver path; use `warp_v3_supercell_tile` for fair JAX-loop Warp comparisons |

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

The solver is class-based:

- **`MPMSolver`** is an Equinox module. Particle/grid state is stored as dynamic JAX leaves, while backend choices, constitutive functions, boundary functions, and the compiled `_frame` are static fields. `stepped()` returns a new solver with updated state; `step()` keeps the driver-friendly mutating API and advances one frame by running `_frame(self.state)`. The frame contains `steps_per_frame` substeps as a single XLA program (via `lax.fori_loop` by default, or unrolled with `loop_kind="python"`).

Kernel selection is driven by `src/mpm_jax/registry.py`:
- `KERNELS` maps each `kernel=<name>` to a `KernelSpec(solver_cls, backend_factory, defaults)`.
- `build_solver(cfg)` reads the registry, builds all closures (particles, params, BCs, constitutive fns), and returns the fully initialised solver.
- No if/elif dispatch in `simulate.py`; the routing is entirely in the registry.

All solver variants now run through the same JAX-owned frame loop. The pure-JAX path compiles the entire frame (multiple substeps) as one XLA program. The inline CUDA variants (`cuda_v*_inline`) move per-particle stencil work into CUDA kernels so the `(N, 27, *)` intermediate tensors never materialize in HBM. The Warp variant uses the official Warp/JAX bridge (`jax_callable`) to launch a tiled Warp P2G kernel from inside that same JAX frame.

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
pixi run -e gpu python simulate.py -cn sweep_baseline    # JAX-only scaling
pixi run -e gpu python simulate.py -cn sweep_all
pixi run -e gpu python simulate.py -cn sweep_quick
pixi run -e gpu python simulate.py -cn sweep_scaling
pixi run -e gpu python simulate.py -cn sweep_profile
```

Each combination gets its own `multirun/<date>/<run>/` subdir with a `results.json`. Sweeps
should use Hydra multirun so log parsers see the expected directory structure.

For an ad-hoc sweep: `pixi run -e gpu python simulate.py -m sim.n_particles=5000,50000,200000 kernel=jax_v1_5,cuda_v2_inline benchmark=true`.

## Profiling

**JAX profiler** (in-process, writes a TensorBoard trace):

```bash
pixi run -e gpu python simulate.py profile=jax benchmark=true \
    kernel=cuda_v3_inline material=sand_jacobi
```

The trace is written to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/jax_trace/` and includes
`jax.named_scope` regions for elasticity, P2G, grid update, G2P, and plasticity.

**Nsight Python profiler** (per-stage kernel analysis, requires `nsight-python`):

```bash
pixi run -e gpu python profile_nsight.py -cn nsight_profile \
    kernel=jax_v1_5 material=sand_jacobi nsight.phase=p2g sim.n_particles=4096
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `sand_jacobi` (default) | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `kernel` | `jax_v1_5` (default), `cuda_v*_inline`, `warp_v3_supercell_tile` | P2G/G2P implementation |
| `profile` | `none` (default), `jax` | Profiling backend |

Top-level fields: `benchmark`, `tag`, `output_dir`. All overridable from CLI:

```bash
pixi run -e gpu python simulate.py sim.n_particles=100000 kernel=cuda_v3_inline benchmark=true
```

Kernel-specific knobs passed as top-level CLI overrides (merged into `cfg.kernel`):

```bash
# loop_kind: fori (default) | python (unrolled)
pixi run -e gpu python simulate.py kernel=cuda_v2_inline kernel.loop_kind=python

# cuda_graph: enable XLA command-buffer capture for cuda_v3_inline
pixi run -e gpu python simulate.py kernel=cuda_v3_inline kernel.cuda_graph=true

# graph_mode for Warp's jax_callable bridge: jax (default) | none | warp | warp_staged | warp_staged_ex
pixi run -e gpu python simulate.py kernel=warp_v3_supercell_tile kernel.graph_mode=jax
```

## Tests

```bash
pixi run test
```

Run focused GPU checks:

```bash
pixi run -e gpu pytest tests/test_cuda_ffi_loader.py tests/test_jax_v1_5.py \
    tests/test_cuda_v2_inline_matches_v1.py -q
```


## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry + benchmark + GIF rendering
├── profile_nsight.py        # Nsight Python per-stage profiler
├── pyproject.toml           # scikit-build-core build + pixi cpu / gpu / hpc envs
├── pixi.lock                # locked deps for all envs (commit this)
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
├── conf/
│   ├── config.yaml
│   ├── nsight_profile.yaml
│   ├── material/            # sand_jacobi.yaml
│   ├── sim/default.yaml
│   ├── kernel/              # jax_v1_5.yaml, cuda_v*.yaml, warp_*.yaml
│   ├── profile/             # none.yaml, jax.yaml
│   └── sweep_*.yaml
└── src/
    └── mpm_jax/
        ├── types.py         # MPMState, StepIntermediates, MPMParams, make_params
        ├── solver.py        # MPMSolver + build_jit_stages
        ├── registry.py      # KERNELS, REMOVED_KERNELS, build_solver(cfg)
        ├── constitutive.py  # sand Jacobi elasticity + plasticity
        ├── boundary.py      # sticky surface collider
        ├── blocks/          # Pure math: weights, p2g, g2p, grid, svd, sort, init
        ├── backends.py      # Backend interface + shared JAX-owned frame loop
        ├── stepping/        # Warp tiled P2G bridge helpers
        └── cuda/
            ├── p2g_cuda.py  # FFI registration + kernel wrappers
            ├── _lib/        # built .so files (gitignored)
            └── kernels/     # p2g_inline.cu, p2g_v2_inline.cu, p2g_v3_inline.cu,
                             # p2g_v4_inline.cu, g2p_fused.cu
```

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
