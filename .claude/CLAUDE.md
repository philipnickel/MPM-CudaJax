# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX** with progressively optimised hand-written **CUDA** P2G scatter kernels. The point of the project is to investigate where JAX/XLA's automatic GPU compilation is sufficient and where custom CUDA kernels are needed.

## Package manager: pixi

**Always use `pixi` to install, sync, and run.** Never invoke `pip`, `pip install`, `python -m pip`, or a bare `python` from the system interpreter — those will miss the project's locked environment.

Three environments, defined in `[tool.pixi.environments]` in `pyproject.toml`:

- `default` — CPU. JAX from conda-forge, no CUDA. Use on macOS / laptop / non-GPU CI.
- `gpu` — Linux only (`linux-64`, `linux-aarch64`). JAX with `*cuda12*` jaxlib build, `cuda-nvcc`, `gxx`, full CUDA 12 toolchain from conda-forge. `warp-lang==1.14.0` from PyPI (conda-forge only carries 1.13, which has an sm_120/Blackwell tile-kernel regression). No `module load` required on DTU HPC — everything ships from conda-forge.
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
- `editable.rebuild = true` in `pyproject.toml` means edits to `.cu` sources trigger a rebuild on the next `import mpm_jax.cuda.p2g_cuda`. Manual rebuild: `pixi run -e gpu rebuild-cuda` (which calls `pixi reinstall mpm-cudajax`).
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
  material/            jelly.yaml, jelly_jacobi.yaml, sand.yaml  (constitutive model)
  sim/default.yaml     n_particles, num_grids, dt, BCs, ...
  kernel/              jax.yaml, jax_v1_5.yaml, cuda_v*.yaml, warp_*.yaml (P2G impl)
  profile/             none.yaml, jax.yaml, warp.yaml
  sweep_*.yaml         pre-baked Hydra multirun sweeps
src/mpm_jax/
  types.py             MPMState, StepIntermediates, MPMParams, make_params, OFFSET_27
  solver.py            MPMSolver, WarpGraphSolver + build_jit_step / build_jit_stages
  registry.py          KERNELS dict, REMOVED_KERNELS dict, build_solver(cfg)
  constitutive.py      5 elasticity + 4 plasticity models
  boundary.py          6 boundary condition types
  callbacks.py         on_frame callback helpers
  p2g_scan.py          jax_v1_5 P2G: lax.scan over 27 offsets, build_jit_stages_scan
  warp_kernels.py      Warp kernels (v1 inline, v2 tile, v3 supercell tile) + JAX wrappers
  warp_graph.py        WarpBonusSimulator: pure-Warp CUDA graph capture/replay engine
  blocks/              Pure-math building blocks (no JIT, no closures)
    weights.py         compute_weights_and_indices: B-spline weights, grid indices
    p2g.py             p2g_compute, p2g_scatter, p2g: Particle-to-Grid (JAX path)
    g2p.py             g2p: Grid-to-Particle gather + APIC update
    grid.py            grid_update: momentum normalise + gravity + damping; build_grid_x
    svd.py             3x3 Jacobi SVD (used by Warp paths)
    sort.py            morton_argsort, _home_super_cell_id
    init.py            get_particles: uniform particle initialisation
  stepping/            Per-variant frame builders (one jit'd frame = N substeps)
    substep.py         step(): one full P2G2P substep (pure fn, safe to JIT)
    jax_frames.py      build_jax_frame, build_jax_v1_5_frame
    cuda_frames.py     build_cuda_v1_frame .. build_cuda_v4_frame
    warp_frames.py     build_warp_v1_frame, build_warp_v3_frame
    warp_graph_frame.py build_warp_graph: constructs a WarpGraphSolver from cfg + particles
  cuda/
    p2g_cuda.py        loads prebuilt .so + jax.ffi.register_ffi_target
    _lib/              prebuilt .so files (gitignored, populated by CMake)
    kernels/
      p2g_inline.cu          cuda_v1: one thread/particle, inline weights, global atomicAdd
      p2g_v2_inline.cu       cuda_v2: warp-shuffle coalescing, inline weights
      p2g_v3_inline.cu       cuda_v3: Morton-sorted particles, inline weights
      p2g_v4_inline.cu       cuda_v4: super-cell-owned grid tile, inline weights
      g2p_fused.cu           fused G2P gather + APIC update (shared by v1–v4)
tests/                 pytest suite
docs/superpowers/      design specs and implementation plans
```

## Architecture (one timestep)

Three embarrassingly parallel phases per substep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

### Class-based API

`MPMSolver` (in `src/mpm_jax/solver.py`) is the stateful shell over the functional JAX core:

- Built once from `params`, an `elasticity_fn`, `plasticity_fn`, boundary functions `pre_fn`/`post_fn`, a frame builder `build_frame`, and `steps_per_frame`. A single JIT-compiled `_frame` function is built at construction time; `self` is never traced.
- `step()` advances one frame (= `steps_per_frame` substeps) by calling `_frame(self.state)`.
- `solve(num_frames, on_frame=None)` loops `step()` with an optional IO callback.
- Default loop inside `build_jax_frame` is `lax.fori_loop`; pass `loop_kind="python"` to unroll instead.

`WarpGraphSolver(MPMSolver)` wraps the pure-Warp `WarpBonusSimulator` capture/replay engine. It overrides `step()` and `solve()` to call `engine.launch_frame()` and does NOT build a JAX `_frame`. `reset()` and `reset_to_initial()` raise `NotImplementedError` (state is owned by the Warp engine).

### Kernel registry

Kernel selection is a registry, not an if/elif chain. `src/mpm_jax/registry.py` defines:

- `KERNELS: dict[str, KernelSpec]` — maps `kernel.name` to a `KernelSpec(solver_cls, build_frame, defaults)`. `build_solver(cfg)` reads this dict, builds particles/params/BCs/constitutive functions, and calls `spec.solver_cls(...)` with the registered frame builder.
- `REMOVED_KERNELS: dict[str, str]` — migration messages for removed/renamed kernels.

Current kernel names:

| `kernel=` | Class | What it does |
|---|---|---|
| `jax` | MPMSolver | Pure JAX/XLA baseline: cuSOLVER SVD, vmap'd compute, `jnp.at[].add()` scatter, `lax.fori_loop` over substeps |
| `jax_v1_5` | MPMSolver | Pure JAX, but P2G uses `lax.scan` over 27 stencil offsets to avoid `(N, 27, *)` intermediates |
| `cuda_v1_inline` | MPMSolver | Inline-weight CUDA P2G (one thread/particle, global atomicAdd) + CUDA G2P |
| `cuda_v2_inline` | MPMSolver | Warp-shuffle coalesced inline CUDA P2G + CUDA G2P; default `loop_kind=fori` |
| `cuda_v3_inline` | MPMSolver | Morton-sorted inline CUDA P2G + CUDA G2P; `cuda_graph=true` enables XLA command-buffer replay |
| `cuda_v4_inline` | MPMSolver | Super-cell-owned grid tile inline CUDA P2G + CUDA G2P |
| `warp_v1_inline` | MPMSolver | P2G authored as a Warp kernel, called from inside JAX JIT via `warp.jax_experimental.jax_kernel` |
| `warp_v3_supercell_tile` | MPMSolver | Super-cell-owned Warp tile P2G: sort by home super-cell, accumulate 4³ shared tile |
| `warp_bonus_graph` | WarpGraphSolver | Pure-Warp CUDA graph: bins by super-cell, runs tiled P2G + grid + G2P without JAX |
| `warp_bonus_v2_graph` | WarpGraphSolver | Pure-Warp graph that sorts particle ids only (avoids copying sorted x/v/C/F buffers) |

Removed/renamed kernels (error message from `build_solver`):
- `cuda_v1`, `cuda_v2`, `cuda_v4` → use the `_inline` variants.
- `cuda_fused` → deprecated; use an inline kernel and `profile=jax`.
- `cuda_v2_fori_inline` → use `kernel=cuda_v2_inline loop_kind=fori` (now the default).
- `cuda_v3_fori_inline` → use `kernel=cuda_v3_inline loop_kind=fori`.
- `cuda_v6_inline` → use `kernel=cuda_v3_inline cuda_graph=true`.

`StepIntermediates` (the namedtuple passed from P2G stage to G2P stage in the per-stage JIT path) is intentionally minimal — only `x_post_bc` and `F_pre_plast`. Carrying weight/dweight/dpos/index would cost ~1148 bytes per particle of inter-stage HBM and OOM the GPU at N=5M. G2P recomputes B-spline weights from `x_post_bc` instead.

## Common commands

```bash
# Default run (renders GIF to ./output)
pixi run -e gpu python simulate.py

# Benchmark mode (timing only, no GIF)
pixi run -e gpu python simulate.py benchmark=true

# Switch kernel
pixi run -e gpu python simulate.py kernel=jax                                          # XLA baseline
pixi run -e gpu python simulate.py kernel=jax_v1_5                                     # scan-over-offsets P2G
pixi run -e gpu python simulate.py kernel=cuda_v1_inline material=jelly_jacobi         # inline CUDA P2G + G2P
pixi run -e gpu python simulate.py kernel=cuda_v2_inline material=jelly_jacobi         # warp-shuffle CUDA (fori loop)
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi         # Morton-sorted CUDA
pixi run -e gpu python simulate.py kernel=cuda_v3_inline cuda_graph=true material=jelly_jacobi  # with XLA CUDA graphs
pixi run -e gpu python simulate.py kernel=warp_v1_inline material=jelly_jacobi
pixi run -e gpu python simulate.py kernel=warp_bonus_graph material=jelly_jacobi benchmark=true

# loop_kind override (python = unrolled, fori = lax.fori_loop; fori is the default)
pixi run -e gpu python simulate.py kernel=cuda_v2_inline kernel.loop_kind=python

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=50000 sim.num_grids=64

# Profilers
pixi run -e gpu python simulate.py profile=jax  benchmark=true     # TensorBoard trace
pixi run -e gpu python simulate.py profile=warp benchmark=true kernel=warp_bonus_graph  # Warp graph timing

# Nsight Python profiler (per-stage kernel analysis)
pixi run -e gpu python profile_nsight.py -cn nsight_profile kernel=jax material=jelly_jacobi nsight.phase=p2g sim.n_particles=4096

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

**Warp 1.14 note:** The `gpu` env pins `warp-lang==1.14.0` from PyPI (conda-forge only carries 1.13, which has a tile-kernel bug on sm_120/Blackwell). `libc = { family = "glibc", version = "2.34" }` in `pyproject.toml` lets both the `manylinux_2_34` aarch64 wheel (GH200) and the `manylinux_2_28` x86_64 wheel (H100/A100) resolve correctly.

## Conventions

- **Sweeps must use Hydra multirun**, never a bash `for` loop. Either use a pre-baked sweep config (`-cn sweep_*`) or pass axes inline: `pixi run -e gpu python simulate.py -m sim.n_particles=5000,50000,200000 kernel=jax,cuda_v1_inline,cuda_v2_inline benchmark=true`. Add new sweep configs under `conf/sweep_<name>.yaml`. Hydra puts each combination in its own `multirun/<date>/<run>/` subdir.
- **Default to short benchmarks.** Steady-state ms/step is stable after the first frame (warmup), so `sim.num_frames=5` (50 substeps) gives reliable timings.
- Single-particle functions live in `src/mpm_jax/blocks/`; vectorise via `jax.vmap`. Don't write batched code by hand — vmap is the contract.
- **Adding a new inline CUDA P2G kernel** (e.g. `cuda_vX_inline`):
  1. Add `src/mpm_jax/cuda/kernels/p2g_vX_inline.cu` (and `g2p_fused.cu` if needed).
  2. Add the kernel name to the `KERNELS` list in `CMakeLists.txt`.
  3. Add `_register_vX_inline()` + `cuda_p2g_vX_inline()` wrapper in `src/mpm_jax/cuda/p2g_cuda.py`.
  4. Add `build_cuda_vX_frame()` in `src/mpm_jax/stepping/cuda_frames.py`.
  5. Register it in `src/mpm_jax/registry.py` `KERNELS` dict as `KernelSpec(MPMSolver, build_cuda_vX_frame)`.
  6. Add `conf/kernel/cuda_vX_inline.yaml`.
  7. Rebuild: `pixi run -e gpu rebuild-cuda` (or `pixi reinstall mpm-cudajax`).
- **Adding a new pure-Warp kernel:** follow the `build_warp_graph` pattern in `src/mpm_jax/stepping/warp_graph_frame.py` and register with `KernelSpec(WarpGraphSolver, build_warp_graph, defaults)`.
- Boundary conditions and constitutive models are registry-based (`REGISTRY` dict in `constitutive.py`, `build_boundary_fns` in `boundary.py`); add a function and a config entry.
- **No `block_until_ready` inside the timed region in benchmark mode.** Both timing modes dispatch all frames back-to-back and sync exactly once after the loop; elapsed/num_frames is the average. Per-stage breakdown comes from `profile=jax` (TensorBoard trace) or `profile_nsight.py`, not from `simulate.py`'s output.
- `simulate.py` enables XLA CUDA graph capture for `kernel=cuda_v3_inline cuda_graph=true` by setting `XLA_FLAGS` before JAX is imported. This must happen before any `import jax` in the process.
- Lint with ruff (config in `ruff.toml`); `I` is allowed as a variable name (identity matrix), and `tests/*` skips E402/F401.

## Don't

- Don't run `pip install` — use `pixi add` / `pixi install`.
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `wandb/`, `*.nsys-rep`, `*.sqlite`, or `.pixi/` (`.gitignore` covers these). DO commit `pixi.lock`.
- Don't bypass the solver class for benchmarking — `simulate_frame` exists only for unjitted per-stage profiling.
- Don't hard-code particle counts, grid sizes, or material params in code — they live in `conf/`.
- Don't reference the old flat `mpm_jax/` layout (no `src/` prefix). All source files live under `src/mpm_jax/`.
