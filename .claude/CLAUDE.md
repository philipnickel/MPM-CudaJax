# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**, progressively optimised with hand-written **CUDA** kernels (via JAX FFI) and, as an alternative, an **NVIDIA cuTile** (tiled programming model) implementation. The project investigates three questions in sequence: where JAX/XLA's automatic GPU compilation is sufficient, where custom CUDA kernels are needed, and whether the tiled programming model (cuTile) can match or beat them.

## Project narrative & analysis method

This is a benchmarking/investigation project; the code is shaped by the following arc. Keep it in mind when adding variants or interpreting results.

**The story (why the code is the way it is):**

1. **Start in JAX.** Implement the full MLS-MPM timestep in pure JAX/XLA (`backend=jax`) — the baseline. It scans over the 27 stencil offsets for **both** P2G and G2P (so neither `(N, 27, *)` intermediate materialises), uses the unified MLS-MPM G2P (the APIC affine `C` doubles as ∇v), and a closed-form StVK elastic stress (SVD-free, no plasticity). **Every other variant reuses this exact JAX G2P (`jax_scan_g2p_mls` / `_g2p_scan_mls`), so across the backend set only the P2G implementation varies.**
2. **Profile to find the bottlenecks.** Use `profile_nsight.py` to locate the *canonical MPM bottlenecks* — chiefly the P2G scatter and the `(N, 27, *)` stencil materialisation.
3. **Optimise the P2G with custom CUDA.** JAX FFI lets us drop hand-written CUDA **P2G** kernels in (`backend=cuda_v1` through `backend=cuda_v4`) while the rest of the timestep — G2P, grid, constitutive — stays the JAX baseline. Holding G2P fixed isolates the P2G question: "where is XLA's scatter enough vs. where does a custom kernel win?"
4. **Try a different programming model — tiled (cuTile).** Finally, investigate whether the **tiled programming model** (NVIDIA cuTile) can reach similar or better P2G performance while keeping the same JAX-owned frame loop. The cuTile Hydra choices are `backend=cutile_v1` for direct scatter and `backend=cutile_v3` for the cleaner home-cell-tiled path. JAX computes stress, updates the grid, and runs the **same JAX baseline G2P**; cuTile owns only the tiled P2G kernel via the cuTile/JAX bridge. (An NVIDIA Warp tiled path was also explored and then dropped — it was slower than both cuTile and the CUDA kernels at this resolution.)

**The analysis method (applied to every variant, in this order):**

1. **`profile_nsight.py`** — automate sweeping metrics across variants *and* particle counts. This is the broad sweep, run across **multiple architectures / systems** (e.g. A100 / H100 / GH200).
2. **`ncu` (Nsight Compute), from the CLI** — same particle count, get roofline / occupancy / memory-throughput for deeper per-kernel analysis.

## Standard benchmark settings

All current performance comparisons use one fixed, well-resolved configuration so numbers are comparable across variants and architectures **and the simulation stays numerically stable** — under-resolution (too few particles per cell) makes MLS-MPM go unstable, which corrupts wall-clock timings (the falling material explodes, particles clamp to the bounds and cluster, and the atomic-scatter P2G becomes contention-bound; see below).

- **Current profiling operating point:** `G=96`, `N=10M`, one frame with 50 substeps. Particles fill `[0.1, 0.9]^3`, which is about 77³ active cells and ~22 particles per active cell.
- **∆x ~= 1.04×10⁻²** (`dx = 1 / sim.num_grids`).
- **Particles uniformly sampled in [0.1, 0.9]³** → `sim.center = [0.5, 0.5, 0.5]`, region side 0.8.
- **APIC transfer** (codebase default) + **StVK elastic jelly** (`material=jelly`).
- Per-particle volume follows the region: `vol = 0.8³ / N` (`MPMParams` derives `vol = prod(sim.size)/n`, so set `sim.size=[0.8,0.8,0.8]`).
- **∆t = 5×10⁻⁵**. Jelly is soft (`E = 10⁴`, wave speed `c = √(E/ρ) ≈ 3.2 m/s`), so with `∆x ≈ 8×10⁻³` the CFL ceiling `∆t_max ≈ ∆x/c ≈ 2.5×10⁻³` sits far above this — `∆t = 5×10⁻⁵` is deliberately conservative for stable, comparable timings (the value is inherited from the original sand benchmark, whose stiffer material needed it). **Run the standard benchmark via `sim=benchmark`** (below), which bakes in all of these settings; do not hand-override `sim.num_grids`/`sim.n_particles` on top of `sim=default`.

**Phase definitions** (kept broad so timing reflects real work, and to attribute the grid step):
- **P2G**: compute force from the deformation gradient F; scatter mass + elastic force + APIC affine momentum to grid nodes; normalize grid velocity and apply gravity. *(The grid normalize+gravity step is grouped under P2G for timing.)*
- **G2P**: gather velocity from the grid; gather velocity gradient (if needed) + the APIC affine velocity field; update F; advect particles.

**Reproduce:**
```bash
# sim=benchmark encodes 10M particles, num_grids=96, ∆t=5e-5, center/size, sticky floor
pixi run python simulate.py -cn config sim=benchmark \
    backend=<k> material=jelly render.enabled=false
```

## Package manager: pixi

**Always use `pixi` to install, sync, and run.** Never invoke `pip`, `pip install`, `python -m pip`, or a bare `python` from the system interpreter — those will miss the project's locked environment.

One environment is defined in `[tool.pixi.environments]` in `pyproject.toml`:

- `default` — Linux only (`linux-64`, `linux-aarch64`) and GPU-first. **JAX runs on CUDA 13** (PyPI `jax[cuda13]`); `cuda-tile[tileiras]` (PyPI) is the cuTile runtime the `cutile` backend requires; `warp-lang==1.14.0` (now used only by the optional Warp OpenGL renderer, not by any P2G kernel) and `nsight-python` are PyPI too. conda-forge supplies the `cuda-nvcc` (**CUDA 13.x**) + `gxx` toolchain that compiles the FFI `.cu` kernels and the nanobind capsule module, plus `nsight-compute`. Runtime environment variables live in Pixi feature activation tables in `pyproject.toml`; the default `gpu` feature owns the CUDA/JAX, single-host NCCL, and compile-cache values. No `module load` required on DTU HPC.

Common patterns:

```bash
pixi install                                      # default GPU env
pixi run python simulate.py ...                   # default GPU env
pixi run test                                     # pytest
pixi run sim                                      # task alias for `python simulate.py`
pixi run ncu-ui                                   # Nsight Compute GUI in default env
pixi run sweep-particles                          # task alias for `simulate.py -cn sweep sweep=particle_count`
pixi run plot-sweeps                              # plot sweep CSVs into figures/sweeps/
pixi add <pkg>                                    # add a runtime dep (edits pyproject.toml)
pixi add --feature gpu <pkg>                      # add to the GPU feature used by default
```

### CUDA kernel build (scikit-build-core + CMake + nanobind)

CUDA kernels in `src/mpm_jax/p2g/cuda/kernels/*.cu` and the tiny capsule binding `p2g_ffi_module.cc` build via `CMakeLists.txt` driven by **scikit-build-core** at `pixi install` time. CMake produces one importable nanobind extension, `mpm_jax.p2g.cuda._p2g_ffi`. `src/mpm_jax/p2g/cuda/p2g_cuda.py` imports that module, obtains PyCapsule handlers for the CUDA FFI symbols, and registers them with JAX FFI (`jax.ffi.register_ffi_target` / `ffi_call`).

Key knobs:

- CUDA is required in the default Pixi environment. If `nvcc` is unavailable, CMake configure fails instead of installing a JAX-only wheel without the native kernels.
- CMake defaults to `native` CUDA architecture autodetection. Cross-build hosts that need a fixed architecture should set that in Pixi task/environment configuration instead of prefixing ad hoc commands.
- `editable.rebuild = true` in `pyproject.toml` means edits to `.cu`, `.cuh`, or binding `.cc` sources trigger a rebuild when the native extension is next imported. Manual rebuild: `pixi reinstall mpm-cudajax`.
- `[build-system].requires` pulls in `scikit-build-core>=0.10`, `cmake>=3.24`, `jax>=0.4.20`, and `nanobind` (jax is needed at build time so CMake can `import jax.ffi` to find the FFI headers).

## Layout

```
simulate.py            Hydra entry point + timing + GIF rendering
profile_nsight.py      Nsight Python profiler for custom CUDA/cuTile P2G kernels
postprocessing/        Analysis tooling (top-level, NOT in the installed mpm_jax package; plotting deps only)
  callbacks.py         ScalingPlotCallback: aggregate sweep results.json -> CSV + scaling plots
pyproject.toml         deps + scikit-build-core build + default Pixi GPU env + tasks
pixi.lock              locked deps (committed)
CMakeLists.txt         CUDA kernel build (called by scikit-build-core)
ruff.toml              lint config
conf/                  Hydra config groups
  config.yaml          top-level defaults (material/sim/backend)
  nsight_profile.yaml  top-level defaults for profile_nsight.py
  material/            jelly.yaml  (constitutive model)
  sim/default.yaml     n_particles, num_grids, dt, ...
  sweep.yaml           pre-baked Hydra multirun sweep entry point
  sweep/               scale-axis config group (particle_count, weak_scaling)
src/mpm_jax/
  types.py             MPMState, MPMParams
  solver.py            RuntimeConfig + MPMSolver + jitted frame stepping + get_particles
  constitutive.py      StVK elastic stress (the jelly material), SVD-free
  g2p_scan.py          JAX baseline G2P: lax.scan over 27 offsets + MLS C=∇v (shared by ALL kernels)
  grid.py              grid_update: momentum normalise + gravity + damping; build_grid_x
  p2g/
    scan.py            JAX baseline P2G: lax.scan over 27 offsets (defines OFFSET_27)
    sort.py            morton_argsort, home_cell_id
    backends/          backend implementations; modules register Hydra choices via hydra-zen
    cutile/            cuTile P2G kernels + cutile_call bridges
      v1.py            cutile_v1: direct scatter comparison backend
      v3.py            cutile_v3: home-cell tiled local reduction
    cuda/
      p2g_cuda.py      imports _p2g_ffi capsules + jax.ffi.register_ffi_target
      kernels/
        p2g_ffi_module.cc    nanobind module exporting FFI handler capsules
        p2g_v1.cu            cuda_v1: one thread/particle, global atomicAdd
        p2g_v2.cu            cuda_v2: home-sorted warp-coalesced atomics
        p2g_v3.cu            cuda_v3: home-cell shared-tile reduction
        p2g_v4.cu            cuda_v4: home-cell structured warp reduction
tests/                 pytest suite
```

## Architecture (one timestep)

Three embarrassingly parallel phases per substep:

1. **P2G** — per-particle: stress (StVK, closed-form) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

### Class-based API

`MPMSolver` (in `src/mpm_jax/solver.py`) is a plain Python class wrapping the functional JAX core:

- Built once from `params`, an `elasticity_fn`, a `Backend`, and `steps_per_frame`. Array state is mutated in place by the driver API; the backend, sticky-floor mask, closures, and the compiled `_frame` are fixed for the solver's lifetime. The solver is never a JAX argument — only `state` (an `MPMState` pytree) is traced — so it needs no pytree machinery.
- `step()` advances one host-driven substep using individually jitted stage
  callables and mutates `self.state`; `_frame` advances one configured frame by
  running a jitted `lax.fori_loop` over the same pure substep.
- `run(capture_frames=...)` drives the configured frame loop, including warmup,
  timing, optional frame capture, and final synchronization.
- The `steps_per_frame` substeps run as a single jitted `lax.fori_loop` over `MPMSolver.step_state`.

### Backend Variants

Kernel selection is a small implementation package, not an if/elif chain. Because only the P2G varies, `src/mpm_jax/p2g/backends/common.py` defines the shared interface/helpers and each implementation module registers its own Hydra backend choice with hydra-zen: `jax.py`, `cuda.py`, and `cutile.py`. A variant overrides `prepare()` when it needs particle ordering and `scatter()` for the P2G kernel. The frame loop calls `backend.prepare()`, `backend.scatter()`, then the shared `g2p_mls()` path; it never contains backend-specific dispatch. Importing `mpm_jax.p2g.backends` loads those modules and commits their config-group entries to Hydra's ConfigStore, so `backend=jax,cuda_v1,cuda_v2,cuda_v3,cuda_v4,cutile_v1,cutile_v3` are Python-backed Hydra choices with no `conf/backend/*.yaml` stubs. Backend constructors own CUDA/cuTile registration. Solver construction is `MPMSolver(hydra.utils.instantiate(cfg.solver))`: the `solver` config node targets `RuntimeConfig`, whose `backend` field is already the instantiated backend object. `MPMSolver` derives params, particles, and initial state from that runtime config; the sticky floor is fixed in the solver frame. There is no availability check — the default Pixi env guarantees the kernels exist.

Current kernel names:

| `backend=` | Class | What it does |
|---|---|---|
| `jax` | MPMSolver | The JAX/XLA baseline. `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), closed-form StVK stress (SVD-free). The shared G2P every other kernel reuses — so only P2G varies |
| `cuda_v1` | MPMSolver | CUDA P2G (one thread/particle, global atomicAdd) + JAX baseline G2P |
| `cuda_v2` | MPMSolver | Home-cell sorted CUDA P2G with one thread/particle and warp-coalesced global atomics + JAX baseline G2P |
| `cuda_v3` | MPMSolver | Home-cell sorted CUDA P2G with one block per home cell, shared 27-node local reduction, and final global flush + JAX baseline G2P |
| `cuda_v4` | MPMSolver | Home-cell sorted CUDA P2G with structured warp reduction over the 27-node local tile before the final global flush + JAX baseline G2P |
| `cutile_v1` | MPMSolver | cuTile direct 27-stencil scatter comparison backend |
| `cutile_v3` | MPMSolver | cuTile home-cell tiled P2G with local 27-node reduction + JAX baseline G2P |

Material baseline:
- `material=jelly` is the only material: StVK elastic stress (closed-form, SVD-free), no plasticity. (The earlier StVK/Drucker-Prager *sand* path and its in-repo Jacobi SVD were removed — jelly never needed them, and the SVD's only consumer was sand's plasticity.)
- The cuTile backend is part of the same JAX loop as the CUDA/JAX variants, so
  ordinary benchmark timing applies. `profile_nsight.py` is only for custom
  CUDA/cuTile P2G scatter kernels; use XProf tracing for `backend=jax`.

## Common commands

```bash
# Default run (renders GIF in the Hydra run directory)
pixi run python simulate.py

# Timing run (no GIF)
pixi run python simulate.py sim=benchmark render.enabled=false

# Switch kernel
pixi run python simulate.py backend=jax                              # JAX/XLA baseline (scan P2G + MLS G2P)
pixi run python simulate.py backend=cuda_v1 material=jelly            # CUDA P2G + JAX G2P
pixi run python simulate.py backend=cuda_v2 material=jelly            # home-sorted warp-coalesced atomics
pixi run python simulate.py backend=cuda_v3 material=jelly            # home-cell shared-tile reduction
pixi run python simulate.py backend=cuda_v4 material=jelly            # home-cell structured warp reduction
pixi run python simulate.py backend=cutile_v3 material=jelly sim=benchmark render.enabled=false  # cuTile tiled P2G

# Override sim params
pixi run python simulate.py sim.n_particles=50000 sim.num_grids=64

# Benchmark P2G prepare/scatter and full solver ms/step at G=96, N=10M.
pixi run python tools/benchmark_p2g_substeps.py

# Nsight Python profiler (direct NCU metrics for warmed custom P2G scatter).
# Use only CUDA/cuTile backends here; use XProf tracing for backend=jax.
# conf/nsight_metrics/{timing,roofline,full}.yaml owns the metric presets;
# override nsight.analyze.metrics=[...] directly for focused collection.
pixi run python profile_nsight.py -cn nsight_profile backend=cutile_v3 sim.n_particles=4096

# Cross-backend analysis: nsight_sweep configs define serial Hydra multiruns
# over custom CUDA/cuTile backends only. Each job appends
# ProfileResults.to_dataframe() rows to sweep-level results.parquet; the
# nsight_plot callback loads that authoritative parquet and renders
# roofline/atomics/occupancy/scheduler (+ roofline_scaling for scale sweeps).
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=single_point

# Scaling roofline trajectories (adds roofline_scaling.png; axis auto-detected):
#   load: fixed grid, growing N (throughput vs problem size, NOT strong scaling)
#         -> nsight_sweep=particle_count
#   weak: fixed ppc, grid+N grow together
#         -> nsight_sweep=weak
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=particle_count
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=weak

# Interactive NCU GUI: launch through Pixi so runtime env vars are inherited.
# simulate.py warms once, then marks the measured jitted frame loop with NVTX.
# sim=benchmark is one frame with 50 substeps for steady benchmark timing.
pixi run ncu-ui
# In the GUI, use app: .pixi/envs/default/bin/python
# args: simulate.py sim=benchmark backend=cutile_v3 render.enabled=false

# Sweeps (Hydra multirun). One entry point + a sweep/ config group axis.
pixi run python simulate.py -cn sweep                       # particle_count (default)
pixi run python simulate.py -cn sweep sweep=particle_count  # constant grid, particle count up
pixi run python simulate.py -cn sweep sweep=weak_scaling    # constant active PPC, particle count up
pixi run sweep                                     # task alias (default = particle_count)
pixi run sweep-particles                           # task alias
pixi run sweep-weak                                # task alias
pixi run plot-sweeps                               # write figures/sweeps/<gpu-kind>/

# Sweep definitions (sweep group choices in conf/sweep/)
# - particle_count: G=96, N=2^18..2^24, 50 substeps        (constant grid, N up)
# - weak_scaling:   active PPC ~= 8.492, G=32..192, 50 substeps  (constant PPC, N up)

# Tests
pixi run test

# Lint
pixi run lint
```

## DTU HPC notes

The default environment is fully self-contained — no `module load` is needed because conda-forge provides `cuda-nvcc`, `gxx`, and the CUDA runtime libs inside the env.

```bash
pixi install    # CMake auto-detects the local GPU arch by default
pixi run sim    # smoke-test
```

**Warp 1.14 note:** `warp-lang==1.14.0` (PyPI) is kept in the default env only for the optional Warp OpenGL renderer (`warp.render`); no P2G kernel uses Warp anymore. `libc = { family = "glibc", version = "2.34" }` in `pyproject.toml` lets both the `manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64 wheel (H100/A100) resolve correctly.

## Conventions

- **Sweeps must use Hydra multirun**, never a bash `for` loop. Either use the pre-baked sweep entry point (`-cn sweep sweep=<axis>`, axes live in `conf/sweep/`) or pass axes inline: `pixi run python simulate.py -m sim.n_particles=5000,50000,200000 backend=jax,cuda_v1,cuda_v2 render.enabled=false`. Add new scale axes under `conf/sweep/<name>.yaml`. Hydra puts each combination in `outputs/sweeps/<gpu-kind>/runs/<date>/<run>/`, and `simulate.py` appends a dataframe-ready row to `outputs/sweeps/<gpu-kind>/results.csv`.
- **Default to short benchmarks.** Steady-state ms/step is stable after the first frame (warmup), so `sim.num_frames=5` (50 substeps) gives reliable timings.
- Single-particle functions vectorise via `jax.vmap` (e.g. `constitutive.stvk_elasticity_jacobi` is `jax.vmap` of a single-3×3 stress). Don't write batched code by hand — vmap is the contract.
- **Adding a new CUDA P2G kernel** (e.g. `cuda_vX`) — only the P2G varies; G2P stays the JAX baseline:
  1. Add `src/mpm_jax/p2g/cuda/kernels/p2g_vX.cu`.
  2. Add the `.cu` source to `P2G_FFI_SOURCES` in `CMakeLists.txt`.
  3. Declare/export the handler capsule in `p2g_ffi_module.cc`, then add a `CudaVXP2G` kernel class in `src/mpm_jax/p2g/cuda/p2g_cuda.py`; constructing the class registers the FFI target.
  4. Add a backend implementation in `src/mpm_jax/p2g/backends/` using that kernel class and overriding `prepare()` if it needs a sort. Decorate the implementation with `hydra_zen.store(name="cuda_vX", group="backend", num_grids="${sim.num_grids}")`.
  5. Include the backend in any sweep configs that should exercise it.
  6. Rebuild the editable package metadata after module/package shape changes: `pixi reinstall mpm-cudajax`.
- **Adding a new cuTile-in-JAX kernel:** put the cuTile kernel + `cutile_call` bridge in a dedicated module under `src/mpm_jax/p2g/cutile/`, add a backend implementation under `src/mpm_jax/p2g/backends/`, and decorate it with `hydra_zen.store(..., group="backend")`.
- Constitutive models are Hydra-instantiated (`material.elasticity._target_`); the sticky floor boundary is fixed in `solver.py`.
- **No `block_until_ready` inside the timed region when `render.enabled=false`.** Timing-only runs dispatch all frames back-to-back and sync exactly once after the loop; elapsed/num_frames is the average. Per-stage breakdown comes from `profile_nsight.py`, not from `simulate.py`'s output.
- `simulate.py` calls `solver.warmup()` before entering its profiled solve range. The measured jitted frame loop is wrapped with NVTX (`mpm_cudajax@<backend>_solve`). `sim=benchmark` is one frame with 50 substeps, and cuTile kernels are named `cutile_v1_p2g_kernel...` / `cutile_v3_p2g_kernel...` in Nsight Compute.
- `profile_nsight.py` profiles the warmed custom CUDA/cuTile P2G scatter path directly:
  solver construction and backend `prepare()` run outside the annotated region,
  then the jitted scatter call is annotated as `<backend>_p2g`.
- Nsight sweeps are serial Hydra multiruns selected with `nsight_sweep=*`.
  Per-job appends to `results.parquet` are the intended aggregation path;
  `NsightPlotCallback` requires that file and does not reconstruct it.
- There is no project-level `XLA_FLAGS` command-buffer override in the default Pixi env. JAX uses the pinned `jaxlib` defaults; XProf tracing uses `jax.profiler.ProfileOptions`, including CUPTI graph tracing, in that same default environment.
- Lint with ruff (config in `ruff.toml`); `I` is allowed as a variable name (identity matrix), and `tests/*` skips E402/F401.

## Don't

- Don't run `pip install` — use `pixi add` / `pixi install`.
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `wandb/`, `*.nsys-rep`, `*.sqlite`, or `.pixi/` (`.gitignore` covers these). DO commit `pixi.lock`.
- Don't bypass the solver class for benchmarking; the outer frame is the compiled stepping unit.
- Don't hard-code particle counts, grid sizes, or material params in code — they live in `conf/`.
- Don't reference the old flat `mpm_jax/` layout (no `src/` prefix). All source files live under `src/mpm_jax/`.
