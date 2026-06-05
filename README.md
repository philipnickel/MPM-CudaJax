# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels and an **NVIDIA cuTile** (tiled
programming model) kernel. Investigates where JAX/XLA's automatic GPU
compilation is sufficient and where custom kernels win.

The solver is **constructed from config** by the config-aware constructor
`MPMSolver.from_cfg(cfg)` (aliased as `build_solver(cfg)`): it reads the
`sim`/`material`/`p2g` sections, builds the pieces (params, particles, backend,
boundary, initial state), and returns a ready `MPMSolver`. `solver.step()` advances one frame;
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
pixi run -e gpu python simulate.py p2g=cuda_v3_inline material=sand_jacobi
```

To benchmark instead of rendering:
```bash
pixi run -e gpu python simulate.py \
    p2g=cuda_v3_inline material=sand_jacobi \
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

**Warp 1.14:** `warp-lang==1.14.0` is kept in the `gpu` env for the optional
`warp_opengl` / `warp_usd` render backends (`warp.render`); it is no longer
used for any P2G kernel. The `glibc 2.34` system-requirement lets both the
`manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64
wheel (H100/A100) resolve correctly.

## Usage

```bash
# Default run (renders GIF to ./output)
pixi run -e gpu python simulate.py

# Benchmark mode (no GIF, no per-frame state capture, wall-clock timing)
pixi run -e gpu python simulate.py benchmark=true

# Pick a kernel
pixi run -e gpu python simulate.py p2g=jax_baseline                                 # JAX/XLA baseline (scan P2G + MLS G2P)
pixi run -e gpu python simulate.py p2g=cuda_v1_inline material=sand_jacobi
pixi run -e gpu python simulate.py p2g=cuda_v2_inline material=sand_jacobi         # warp-shuffle coalescing
pixi run -e gpu python simulate.py p2g=cuda_v3_inline material=sand_jacobi         # Morton sort
pixi run -e gpu python simulate.py p2g=cuda_v4_inline material=sand_jacobi         # super-cell grid tile
pixi run -e gpu python simulate.py p2g=cutile_v6_atomic_tile material=sand_jacobi benchmark=true  # cuTile (tiled model)

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

## Kernel variants

| `p2g=` | What it does |
|---|---|
| `jax_baseline` | The JAX/XLA baseline: `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), scatter-free Jacobi SVD. Every other kernel reuses this G2P, so only the P2G varies. |
| `cuda_v1_inline` | CUDA inline-weight P2G (one thread/particle, global `atomicAdd`) + JAX baseline G2P. |
| `cuda_v2_inline` | CUDA warp-shuffle coalesced inline P2G + JAX baseline G2P. |
| `cuda_v3_inline` | CUDA Morton-sorted inline P2G + JAX baseline G2P. (XLA command-buffer / CUDA-Graph capture is on for every kernel via the gpu env's `XLA_FLAGS`.) |
| `cuda_v4_inline` | CUDA super-cell-owned grid tile inline P2G + JAX baseline G2P. |
| `cutile_v6_atomic_tile` | NVIDIA cuTile (tiled programming model) P2G + JAX baseline G2P: SPGrid-style arena scatter — sort by SC=2 home super-cell, reduce each super-cell into a 4³ L1 arena, write back with one tile-coalesced `atomic_store_add` (no coloring). Occupancy autotuned per-GPU. Requires `cuda-tile`. |

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

The solver is class-based:

- **`MPMSolver`** is a plain Python class. Particle/grid state is mutated in place by the driver API; the backend, constitutive/boundary closures, and the compiled `_frame` are fixed for the solver's lifetime. `stepped()` returns a new solver with advanced state (shallow copy + new state); `step()` is the mutating driver and advances one frame by running `_frame(self.state)`. The frame runs `steps_per_frame` substeps as a single XLA program via `lax.fori_loop`.

Construction (`MPMSolver.from_cfg(cfg)` in `src/mpm_jax/solver.py`):
- The config-aware constructor reads the `sim`/`material`/`p2g` config sections and builds the pieces — params (with derived dx/vol/p_mass), particles, the name-selected backend, boundary closures, initial state — threading the shared scalars (`n_particles`/`num_grids`) as locals. `simulate.py` / `profile_nsight.py` call `build_solver(cfg)` (an alias). `MPMSolver.__init__` itself takes the built pieces (config-agnostic, so it's directly unit-testable).
- `backends.py` is a small `Backend` class hierarchy (base = jax_baseline; one subclass per variant overriding `prepare()`/`p2g()`, with `g2p()` shared on the base). `build_backend(name, num_grids)` maps the name to a constructed backend and validates the super-cell grid-divisibility at init. `KERNEL_NAMES` lists the valid names. The frame loop calls `backend.step()` (order + scatter) and `backend.g2p()`.
- No if/elif dispatch anywhere; routing is the `p2g=` config selecting the backend by name.

All solver variants now run through the same JAX-owned frame loop. The pure-JAX path compiles the entire frame (multiple substeps) as one XLA program. The inline CUDA variants (`cuda_v*_inline`) move per-particle stencil work into CUDA kernels so the `(N, 27, *)` intermediate tensors never materialize in HBM. The cuTile variant (`cutile_v6_atomic_tile`) launches a tiled-programming-model P2G kernel from inside that same JAX frame via the cuTile/JAX bridge.

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

For an ad-hoc sweep: `pixi run -e gpu python simulate.py -m sim.n_particles=5000,50000,200000 p2g=jax_baseline,cuda_v2_inline benchmark=true`.

## Profiling

**JAX profiler** (in-process, writes a TensorBoard trace):

```bash
pixi run -e gpu python simulate.py profile=jax benchmark=true \
    p2g=cuda_v3_inline material=sand_jacobi
```

The trace is written to `outputs/<YYYY-MM-DD>/<HH-MM-SS>/jax_trace/` and includes
`jax.named_scope` regions for elasticity, P2G, grid update, G2P, and plasticity.

View the trace with TensorBoard/XProf:

```bash
pixi run -e gpu tensorboard \
    --logdir outputs/<YYYY-MM-DD>/<HH-MM-SS>/jax_trace \
    --port 6006 \
    --bind_all
```

Then open `http://localhost:6006`, select the **Profile** tab, choose the run,
and open **Tools -> trace_viewer**. On a remote GPU machine, keep TensorBoard
running there and forward the port from your laptop:

```bash
ssh -L 6006:localhost:6006 <user>@<remote-host>
```

The `gpu` Pixi environment includes `tensorboard`, `xprof`, and a
`setuptools<81` pin because TensorBoard 2.20 still imports `pkg_resources`.

For single-frame traces where compilation should not dominate the timeline,
warm up once before starting the profiler, then block on the frame inside the
trace:

```python
import jax

solver.step()
jax.block_until_ready(solver.state.x)
solver.reset_to_initial()

jax.profiler.start_trace("traces/xprof_mpm_step")
with jax.profiler.StepTraceAnnotation("mpm_frame", step_num=0):
    solver.step()
    jax.block_until_ready(solver.state.x)
jax.profiler.stop_trace()
```

This workflow works for all registered kernels because they all enter the same
JAX-owned frame loop through `build_solver(cfg)`. Pure JAX variants are shown as
XLA-generated operations and fusion kernels. CUDA and cuTile variants appear as
JAX/XLA custom calls plus their GPU kernels.

**Nsight Python profiler** (per-stage kernel analysis, requires `nsight-python`):

```bash
pixi run -e gpu python profile_nsight.py -cn nsight_profile \
    p2g=jax_baseline material=sand_jacobi nsight.phase=p2g sim.n_particles=4096
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `sand_jacobi` (default) | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `p2g` | `jax_baseline` (default), `cuda_v*_inline`, `cutile_v6_atomic_tile` | P2G implementation (G2P shared) |
| `profile` | `none` (default), `jax` | Profiling backend |

Top-level fields: `benchmark`, `tag`, `output_dir`. All overridable from CLI:

```bash
pixi run -e gpu python simulate.py sim.n_particles=100000 p2g=cuda_v3_inline benchmark=true
```

Kernel-specific knobs passed as top-level CLI overrides (merged into `cfg.p2g`):

```bash
# autotune: cutile_v6 occupancy is tuned per-GPU and cached; disable with autotune=false
pixi run -e gpu python simulate.py p2g=cutile_v6_atomic_tile p2g.autotune=false
```

(XLA command-buffer / CUDA-Graph capture is always on via `XLA_FLAGS` in the gpu env — no per-kernel flag.)

## Tests

```bash
pixi run test
```

Run focused GPU checks:

```bash
pixi run -e gpu pytest tests/test_cuda_ffi_loader.py tests/test_p2g_scan.py \
    tests/test_cuda_v2_inline_matches_v1.py -q
```


## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry + benchmark + GIF rendering
├── profile_nsight.py        # Nsight Python P2G profiler
├── pyproject.toml           # scikit-build-core build + pixi cpu / gpu / hpc envs
├── pixi.lock                # locked deps for all envs (commit this)
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
├── conf/
│   ├── config.yaml
│   ├── nsight_profile.yaml
│   ├── material/            # sand_jacobi.yaml
│   ├── sim/default.yaml
│   ├── p2g/              # jax_baseline.yaml, cuda_v*.yaml, cutile_v6_atomic_tile.yaml
│   ├── profile/             # none.yaml, jax.yaml
│   └── sweep_*.yaml
└── src/
    └── mpm_jax/
        ├── types.py         # MPMState, MPMParams, make_params
        ├── solver.py        # MPMSolver
        ├── registry.py      # build_solver(cfg) alias + build_backend / KERNEL_NAMES re-exports
        ├── constitutive.py  # sand Jacobi elasticity + plasticity
        ├── boundary.py      # sticky surface collider
        ├── blocks/          # Pure math: weights, g2p, grid, svd, sort, init
        ├── backends.py      # Backend interface + shared JAX-owned frame loop
        ├── cutile_p2g.py    # cuTile arena-scatter P2G kernel + jax bridge
        ├── cutile_autotune.py  # per-GPU occupancy autotune for the cuTile kernel
        └── cuda/
            ├── p2g_cuda.py  # FFI registration + kernel wrappers
            ├── _lib/        # built .so files (gitignored)
            └── kernels/     # p2g_inline.cu, p2g_v2_inline.cu, p2g_v3_inline.cu,
                             # p2g_v4_inline.cu
```

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
