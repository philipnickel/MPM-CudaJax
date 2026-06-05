# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**, progressively optimised with hand-written **CUDA** kernels (via JAX FFI) and, as an alternative, an **NVIDIA cuTile** (tiled programming model) implementation. The project investigates three questions in sequence: where JAX/XLA's automatic GPU compilation is sufficient, where custom CUDA kernels are needed, and whether the tiled programming model (cuTile) can match or beat them.

## Project narrative & analysis method

This is a benchmarking/investigation project; the code is shaped by the following arc. Keep it in mind when adding variants or interpreting results.

**The story (why the code is the way it is):**

1. **Start in JAX.** Implement the full MLS-MPM timestep in pure JAX/XLA (`kernel=jax_baseline`) — the baseline. It scans over the 27 stencil offsets for **both** P2G and G2P (so neither `(N, 27, *)` intermediate materialises), uses the unified MLS-MPM G2P (the APIC affine `C` doubles as ∇v), and a scatter-free Jacobi SVD for stress + plasticity. **Every other variant reuses this exact JAX G2P (`_make_jax_scan_g2p_mls`), so across the registry only the P2G implementation varies.**
2. **Profile to find the bottlenecks.** Use the JAX profiler trace (`profile=jax`) to locate the *canonical MPM bottlenecks* — chiefly the P2G scatter and the `(N, 27, *)` stencil materialisation.
3. **Optimise the P2G with custom CUDA.** JAX FFI lets us drop hand-written CUDA **P2G** kernels in (`cuda_v1..v4_inline`) while the rest of the timestep — G2P, grid, constitutive — stays the JAX baseline. Holding G2P fixed isolates the P2G question: "where is XLA's scatter enough vs. where does a custom kernel win?"
4. **Try a different programming model — tiled (cuTile).** Finally, investigate whether the **tiled programming model** (NVIDIA cuTile) can reach similar or better P2G performance while keeping the same JAX-owned frame loop. The cuTile path is `cutile_v6_atomic_tile`: JAX computes stress, sorts particles by super-cell, updates the grid, and runs the **same JAX baseline G2P**; cuTile owns only the tiled P2G kernel (an SPGrid-style arena scatter — reduce each SC=2 super-cell into a 4³ L1 arena, then one tile-coalesced `atomic_store_add` per arena) via the cuTile/JAX bridge. At the standard 8M-particle benchmark it is the fastest P2G in the registry, ahead of the hand-written CUDA kernels. (An NVIDIA Warp tiled path was also explored and then dropped — it was slower than both cuTile and the CUDA kernels at this resolution.)

**The analysis method (applied to every variant, in this order):**

1. **JAX trace** (`profile=jax`) — on a *representative* particle count, identify where time goes for the jax/cuda-based variants.
2. **`ncu` (Nsight Compute), from the CLI** — same particle count, get roofline / occupancy / memory-throughput for the initial deeper per-kernel analysis.
3. **`nsight-python`** (`profile_nsight.py`) — automate sweeping metrics across variants *and* particle counts. This is the broad sweep, run across **multiple architectures / systems** (e.g. A100 / H100 / GH200).

## Standard benchmark settings

All performance comparisons use one fixed, well-resolved configuration (adapted from the Taichi / MLS-MPM high-performance benchmark) so numbers are comparable across variants and architectures **and the simulation stays numerically stable** — under-resolution (too few particles per cell) makes MLS-MPM go unstable, which corrupts wall-clock timings (the falling material explodes, particles clamp to the bounds and cluster, and the atomic-scatter P2G becomes contention-bound; see below).

- **8 particles per cell** — the MLS-MPM resolution sweet spot (2³ per cell). Do not benchmark below this.
- **∆x = 8×10⁻³** → `sim.num_grids = 125` (the solver uses `dx = 1/num_grids`). The particle-filled region then spans **100³ active cells**. (The committed `sim=benchmark` preset uses `num_grids = 124`, an even grid required by the super-cell-tiled cuTile/CUDA kernels.)
- **Particles uniformly sampled in [0.1, 0.9]³** → `sim.center = [0.5, 0.5, 0.5]`, region side 0.8. 100³ active cells × 8 ppc = **8 M particles** → `sim.n_particles = 8_000_000`.
- **APIC transfer** (codebase default) + **StVK/Drucker-Prager sand** (`material=sand_jacobi`).
- Per-particle volume follows the region: `vol = 0.8³ / N` (`make_params` computes `vol = prod(sim.size)/n`, so set `sim.size=[0.8,0.8,0.8]`).
- **∆t = 5×10⁻⁵** (CFL). Sand's elastic wave speed `c = √(E/ρ) ≈ 45 m/s` with `∆x ≈ 8×10⁻³` gives `∆t_max ≈ 9×10⁻⁵`, so the `sim=default` `∆t = 3×10⁻⁴` **explodes** at this resolution — vmax → 10⁶ within ~30 substeps for *every* kernel, including the `jax_baseline`. **Always run the standard benchmark via `sim=benchmark`** (below), which bakes in all of these settings at a CFL-safe `∆t`; do not hand-override `sim.num_grids`/`sim.n_particles` on top of `sim=default`.

**Phase definitions** (kept broad so timing reflects real work, and to attribute the grid step):
- **P2G**: compute force from the deformation gradient F; scatter mass + elastic force + APIC affine momentum to grid nodes; normalize grid velocity and apply gravity. *(The grid normalize+gravity step is grouped under P2G for timing.)*
- **G2P**: gather velocity from the grid; gather velocity gradient (if needed) + the APIC affine velocity field; update F; advect particles.

**Reproduce:**
```bash
# sim=benchmark encodes 8M particles, num_grids=124, ∆t=5e-5, center/size, sticky floor
pixi run -e gpu python simulate.py -cn config sim=benchmark \
    kernel=<k> material=sand_jacobi benchmark=true
```

## Package manager: pixi

**Always use `pixi` to install, sync, and run.** Never invoke `pip`, `pip install`, `python -m pip`, or a bare `python` from the system interpreter — those will miss the project's locked environment.

Three environments, defined in `[tool.pixi.environments]` in `pyproject.toml`:

- `default` — CPU. JAX from conda-forge, no CUDA. Use on macOS / laptop / non-GPU CI.
- `gpu` — Linux only (`linux-64`, `linux-aarch64`). **JAX runs on CUDA 13** (PyPI `jax[cuda13]`); `cuda-tile[tileiras]` (PyPI) is the cuTile runtime the `cutile_*` kernels require; `warp-lang==1.14.0` (now used only by the optional `warp_opengl`/`warp_usd` renderers, not by any P2G kernel) and `nsight-python` are PyPI too. conda-forge supplies the `cuda-nvcc` (**CUDA 12.x**) + `gxx` toolchain that compiles the FFI `.cu` kernels, plus `nsight-compute`. So the *build* toolchain is CUDA 12.x while the JAX *runtime* is CUDA 13 — they coexist and the FFI kernels load fine on the cuda13 runtime. A `[tool.pixi.feature.gpu.activation.env]` block sets `JAX_PLATFORMS=cuda` + a persistent JAX compile cache (`.jax_cache/`). No `module load` required on DTU HPC.
- `hpc` — Linux only. JAX from PyPI with `cuda12-local` extras (links against the site's CUDA toolkit, loaded via `module load`). Use on clusters where conda-forge CUDA lags the driver. `nvcc` and `gxx` are provided by the module; `cuda-nvcc` is NOT included in this env.

Common patterns:

```bash
pixi install                                      # default (CPU) env
pixi install -e gpu                               # GPU env (Linux)
pixi run python simulate.py ...                   # default env
pixi run -e gpu python simulate.py ...            # gpu env
pixi run test                                     # pytest
pixi run -e gpu sim                               # task alias for `python simulate.py`
pixi run -e gpu sweep-quick                       # task alias for `simulate.py -cn sweep_quick`
pixi add <pkg>                                    # add a runtime dep (edits pyproject.toml)
pixi add --feature gpu <pkg>                      # add to the gpu feature only
pixi add --feature cpu <pkg>                      # add to the cpu feature only
```

### CUDA kernel build (scikit-build-core + CMake)

CUDA kernels in `src/mpm_jax/cuda/kernels/*.cu` build via `CMakeLists.txt` driven by **scikit-build-core** at `pixi install` time. Output `.so` files land in `src/mpm_jax/cuda/_lib/` (gitignored) and are loaded by `src/mpm_jax/cuda/p2g_cuda.py` which registers them with JAX FFI (`jax.ffi.register_ffi_target` / `ffi_call`).

Key knobs:

- `MPM_CUDA_ARCH=sm_86` (or `sm_90`, etc.) at install time → CMake picks that arch. Default is `native` (CMake auto-detects the local GPU). Set this before `pixi install -e gpu` on cross-build hosts.
- If `nvcc` is not on PATH (the default CPU env), CMake's `check_language(CUDA)` returns early and the wheel installs fine without CUDA kernels — the JAX baseline still works. Useful for CPU-only dev.
- `editable.rebuild = true` in `pyproject.toml` means edits to `.cu` sources trigger a rebuild on the next `import mpm_jax.cuda.p2g_cuda`. Manual rebuild: `pixi reinstall mpm-cudajax`.
- `[build-system].requires` pulls in `scikit-build-core>=0.10`, `cmake>=3.24`, and `jax>=0.4.20` (jax is needed at build time so CMake can `import jax.ffi` to find the FFI headers).

## Layout

```
simulate.py            Hydra entry point + timing + GIF rendering
profile_nsight.py      Nsight Python profiler for per-stage and per-kernel analysis
pyproject.toml         deps + scikit-build-core build + pixi cpu / gpu / hpc envs + tasks
pixi.lock              locked deps for all envs (committed)
CMakeLists.txt         CUDA kernel build (called by scikit-build-core)
ruff.toml              lint config
conf/                  Hydra config groups
  config.yaml          top-level defaults (material/sim/kernel/profile)
  nsight_profile.yaml  top-level defaults for profile_nsight.py
  material/            sand_jacobi.yaml  (constitutive model)
  sim/default.yaml     n_particles, num_grids, dt, BCs, ...
  kernel/              jax_baseline.yaml, cuda_v*.yaml, cutile_v6_atomic_tile.yaml (P2G impl)
  profile/             none.yaml, jax.yaml
  sweep_*.yaml         pre-baked Hydra multirun sweeps
src/mpm_jax/
  types.py             MPMState, MPMParams, make_params
  solver.py            MPMSolver
  registry.py          build_solver(cfg): resolved Hydra config -> MPMSolver
  constitutive.py      sand Jacobi elasticity + plasticity
  boundary.py          sticky surface collider
  callbacks.py         on_frame callback helpers
  backends.py          Backend class hierarchy + shared frame loop + build_backend
  p2g_scan.py          JAX baseline P2G: lax.scan over 27 offsets
  g2p_scan.py          JAX baseline G2P: lax.scan over 27 offsets + MLS C=∇v (shared by ALL kernels)
  blocks/              Pure-math building blocks (no JIT, no closures)
    weights.py         compute_weights_and_indices: B-spline weights, grid indices
    g2p.py             g2p: Grid-to-Particle gather + APIC update
    grid.py            grid_update: momentum normalise + gravity + damping; build_grid_x
    svd.py             3x3 Jacobi SVD, scatter-free (no .at[].set() -> XLA fuses it; used by StVK elasticity + Drucker-Prager plasticity)
    sort.py            morton_argsort, home_super_cell_id
    init.py            get_particles: uniform particle initialisation
  cutile_p2g.py        cuTile arena-scatter P2G kernel + cutile_call bridge
  cutile_autotune.py   per-GPU occupancy autotune for the cuTile P2G kernel
  cuda/
    p2g_cuda.py        loads prebuilt .so + jax.ffi.register_ffi_target
    _lib/              prebuilt .so files (gitignored, populated by CMake)
    kernels/
      p2g_inline.cu          cuda_v1: one thread/particle, inline weights, global atomicAdd
      p2g_v2_inline.cu       cuda_v2: warp-shuffle coalescing, inline weights
      p2g_v3_inline.cu       cuda_v3: Morton-sorted particles, inline weights
      p2g_v4_inline.cu       cuda_v4: super-cell-owned grid tile, inline weights
tests/                 pytest suite
```

## Architecture (one timestep)

Three embarrassingly parallel phases per substep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

### Class-based API

`MPMSolver` (in `src/mpm_jax/solver.py`) is an Equinox module over the functional JAX core:

- Built once from `params`, an `elasticity_fn`, `plasticity_fn`, boundary functions `pre_fn`/`post_fn`, a `Backend`, and `steps_per_frame`. State arrays are dynamic JAX leaves; backend callables and the compiled `_frame` are static Equinox fields.
- `stepped()` returns a new solver with advanced state. `step()` keeps the existing mutating driver API and advances one frame (= `steps_per_frame` substeps) by calling `_frame(self.state)`.
- `solve(num_frames, on_frame=None)` loops `step()` with an optional IO callback.
- Default loop inside `build_backend_frame` is `lax.fori_loop`; pass `loop_kind="python"` to unroll instead.

### Kernel registry

Kernel selection is a small class hierarchy, not an if/elif chain. Because only the P2G varies, `src/mpm_jax/backends.py` defines a `Backend` base (jax_baseline: identity order + scan P2G + shared MLS-MPM G2P) and one subclass per variant (`CudaInline`→`CudaV1/V2/V3`, `CudaV4`, `CutileV6`). A variant overrides `prepare()` (the "sort") and `p2g()` (the scatter); `g2p()` lives on the base and is shared by all. The frame loop calls `backend.step()` (which orders the particles then scatters) and `backend.g2p()` — it never sees the sort. `build_backend(name, num_grids)` maps the name (via the `_BACKENDS` dict) to a constructed backend and **validates at init** (super-cell grid-divisibility; the CUDA/cuTile kernel handlers are also registered here, at build time, so a persistent compile-cache hit still finds them). `KERNEL_NAMES` exposes the valid names. There is no availability check — the `gpu` pixi env guarantees the kernels exist. `build_solver(cfg)` in `registry.py` builds particles/params/BCs/constitutive functions, calls `build_backend`, and passes the result to `MPMSolver`. The `conf/kernel/<name>.yaml` files are thin — just `name:` (the filename is the identifier; G2P/grid/loop are fixed in code).

Current kernel names:

| `kernel=` | Class | What it does |
|---|---|---|
| `jax_baseline` | MPMSolver | The JAX/XLA baseline. `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), scatter-free Jacobi SVD. The shared G2P every other kernel reuses — so only P2G varies |
| `cuda_v1_inline` | MPMSolver | CUDA inline-weight P2G (one thread/particle, global atomicAdd) + JAX baseline G2P |
| `cuda_v2_inline` | MPMSolver | CUDA warp-shuffle coalesced inline P2G + JAX baseline G2P; default `loop_kind=fori` |
| `cuda_v3_inline` | MPMSolver | CUDA Morton-sorted inline P2G + JAX baseline G2P (XLA command-buffer / CUDA-Graph capture is on for all kernels via the gpu env's `XLA_FLAGS`) |
| `cuda_v4_inline` | MPMSolver | CUDA super-cell-owned grid tile inline P2G + JAX baseline G2P |
| `cutile_v6_atomic_tile` | MPMSolver | NVIDIA cuTile (tiled programming model) P2G + JAX baseline G2P: SPGrid-style arena scatter (SC=2 super-cell → 4³ L1 arena → one tile-coalesced `atomic_store_add`, no coloring), occupancy autotuned per-GPU. Fastest P2G in the registry. Requires `cuda-tile` |

Material baseline:
- `material=sand_jacobi` is the default JAX/CUDA material path: StVK elasticity + Drucker-Prager plasticity, both using the in-repo Jacobi SVD.
- The cuTile backend is part of the same JAX loop as the CUDA/JAX variants, so `profile=jax` and ordinary benchmark timing apply.

## Common commands

```bash
# Default run (renders GIF to ./output)
pixi run -e gpu python simulate.py

# Benchmark mode (timing only, no GIF)
pixi run -e gpu python simulate.py benchmark=true

# Switch kernel
pixi run -e gpu python simulate.py kernel=jax_baseline                                 # JAX/XLA baseline (scan P2G + MLS G2P)
pixi run -e gpu python simulate.py kernel=cuda_v1_inline material=sand_jacobi         # CUDA inline P2G + JAX G2P
pixi run -e gpu python simulate.py kernel=cuda_v2_inline material=sand_jacobi         # warp-shuffle CUDA (fori loop)
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=sand_jacobi         # Morton-sorted CUDA
pixi run -e gpu python simulate.py kernel=cuda_v4_inline material=sand_jacobi         # super-cell grid tile CUDA
pixi run -e gpu python simulate.py kernel=cutile_v6_atomic_tile material=sand_jacobi benchmark=true  # cuTile tiled P2G

# loop_kind override (python = unrolled, fori = lax.fori_loop; fori is the default)
pixi run -e gpu python simulate.py kernel=cuda_v2_inline kernel.loop_kind=python

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=50000 sim.num_grids=64

# Profilers
pixi run -e gpu python simulate.py profile=jax  benchmark=true     # TensorBoard trace

# Nsight Python profiler (per-stage kernel analysis)
pixi run -e gpu python profile_nsight.py -cn nsight_profile kernel=jax_baseline material=sand_jacobi nsight.phase=p2g sim.n_particles=4096

# Sweeps (Hydra multirun)
pixi run -e gpu python simulate.py -cn sweep_baseline    # JAX-only scaling
pixi run -e gpu python simulate.py -cn sweep_all
pixi run -e gpu python simulate.py -cn sweep_quick
pixi run -e gpu python simulate.py -cn sweep_scaling
pixi run -e gpu python simulate.py -cn sweep_profile
pixi run -e gpu sweep-all                                # task alias
pixi run -e gpu sweep-quick                              # task alias

# Tests
pixi run test

# Lint
pixi run lint
```

## DTU HPC notes

The `gpu` environment is fully self-contained — no `module load` is needed because conda-forge provides `cuda-nvcc`, `gxx`, and the CUDA runtime libs inside the env.

```bash
MPM_CUDA_ARCH=sm_90 pixi install -e gpu    # build kernels for Hopper (H100)
pixi run -e gpu sim                        # smoke-test
```

For clusters where the conda-forge CUDA lags the driver (e.g. CUDA 12.9 env vs 12.3 driver), use the `hpc` env instead:

```bash
module load nvhpc/24.1                     # provides nvcc + libcuda matched to driver
MPM_CUDA_ARCH=sm_90 pixi install -e hpc
pixi run -e hpc python simulate.py ...
```

CMake auto-detects the local GPU arch when `MPM_CUDA_ARCH` is unset.

**Warp 1.14 note:** `warp-lang==1.14.0` (PyPI) is kept in the `gpu` env only for the optional `warp_opengl`/`warp_usd` render backends (`warp.render`); no P2G kernel uses Warp anymore. `libc = { family = "glibc", version = "2.34" }` in `pyproject.toml` lets both the `manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64 wheel (H100/A100) resolve correctly.

## Conventions

- **Sweeps must use Hydra multirun**, never a bash `for` loop. Either use a pre-baked sweep config (`-cn sweep_*`) or pass axes inline: `pixi run -e gpu python simulate.py -m sim.n_particles=5000,50000,200000 kernel=jax_baseline,cuda_v1_inline,cuda_v2_inline benchmark=true`. Add new sweep configs under `conf/sweep_<name>.yaml`. Hydra puts each combination in its own `multirun/<date>/<run>/` subdir.
- **Default to short benchmarks.** Steady-state ms/step is stable after the first frame (warmup), so `sim.num_frames=5` (50 substeps) gives reliable timings.
- Single-particle functions live in `src/mpm_jax/blocks/`; vectorise via `jax.vmap`. Don't write batched code by hand — vmap is the contract.
- **Adding a new CUDA P2G kernel** (e.g. `cuda_vX_inline`) — only the P2G varies; G2P stays the JAX baseline:
  1. Add `src/mpm_jax/cuda/kernels/p2g_vX_inline.cu`.
  2. Add the kernel name to the `KERNELS` list in `CMakeLists.txt`.
  3. Add `_register_vX_inline()` + `cuda_p2g_vX_inline()` wrapper in `src/mpm_jax/cuda/p2g_cuda.py`.
  4. Add a `Backend` subclass in `src/mpm_jax/backends.py` overriding `p2g()` (and `prepare()` if it needs a sort); `g2p()` is inherited from the base, so only P2G differs. Register the kernel in `__init__` (so a compile-cache hit still finds the handler), and return the super-cell width from `grid_divisor()` if the grid must divide it.
  5. Add it to the `_BACKENDS` dict in `src/mpm_jax/backends.py` (`name -> lambda num_grids, autotune: YourBackend()`).
  6. Add a thin `conf/kernel/cuda_vX_inline.yaml` (just `name: cuda_vX_inline`).
  7. Rebuild: `pixi reinstall mpm-cudajax`.
- **Adding a new cuTile-in-JAX kernel:** put the cuTile kernel + `cutile_call` bridge in a dedicated module (see `cutile_p2g.py`), add a `Backend` subclass in `src/mpm_jax/backends.py` and a `_BACKENDS` entry, and a thin `conf/kernel/<name>.yaml`.
- Boundary conditions and constitutive models are registry-based (`REGISTRY` dict in `constitutive.py`, `build_boundary_fns` in `boundary.py`); add a function and a config entry.
- **No `block_until_ready` inside the timed region in benchmark mode.** Both timing modes dispatch all frames back-to-back and sync exactly once after the loop; elapsed/num_frames is the average. Per-stage breakdown comes from `profile=jax` (TensorBoard trace) or `profile_nsight.py`, not from `simulate.py`'s output.
- XLA command-buffer / CUDA-Graph capture is enabled for **all** kernels via `XLA_FLAGS` in the gpu env's `[tool.pixi.feature.gpu.activation.env]` block (`--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE`), so it is always on under `pixi run -e gpu`. There is no per-kernel `cuda_graph` flag anymore.
- Lint with ruff (config in `ruff.toml`); `I` is allowed as a variable name (identity matrix), and `tests/*` skips E402/F401.

## Don't

- Don't run `pip install` — use `pixi add` / `pixi install`.
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `wandb/`, `*.nsys-rep`, `*.sqlite`, or `.pixi/` (`.gitignore` covers these). DO commit `pixi.lock`.
- Don't bypass the solver class for benchmarking; the outer frame is the compiled stepping unit.
- Don't hard-code particle counts, grid sizes, or material params in code — they live in `conf/`.
- Don't reference the old flat `mpm_jax/` layout (no `src/` prefix). All source files live under `src/mpm_jax/`.
