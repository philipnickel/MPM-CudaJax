# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels and an **NVIDIA cuTile** (tiled
programming model) kernel. Investigates where JAX/XLA's automatic GPU
compilation is sufficient and where custom kernels win.

The solver is **constructed from config** by Hydra-instantiating `cfg.solver`
into a `RuntimeConfig` and passing it to `MPMSolver`: the `backend` config
targets the backend class directly, and `MPMSolver` builds params, particles,
boundaries, and initial state. `solver.step()` advances one staged substep;
`solver.run(capture_frames=...)` drives configured jitted frames by default.

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
Prints `total_steps`, `elapsed_s`, `steps_per_sec`, average `ms/step`,
high-level `particles_per_sec`, and the detected `gpu_type`. No GIF, no
per-frame state capture — just wall-clock timing.

Outputs:
- Single-run GIF renders, `results.json`, Hydra logs, config snapshots → `outputs/runs/<gpu-kind>/<date>/<run>/`
- Multirun job outputs → `outputs/sweeps/<gpu-kind>/runs/<date>/<run>/`
- Dataframe-ready sweep table → `outputs/sweeps/<gpu-kind>/results.csv`
- Native CUDA extension → `mpm_jax.p2g.cuda._p2g_ffi` (rebuilds on native-source edit via `editable.rebuild=true`)

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
`mpm_jax.p2g.cuda._p2g_ffi`. `p2g_cuda.py` imports that module, gets PyCapsule
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
pixi run python simulate.py backend=cutile_v3 material=jelly sim=benchmark render.enabled=false  # cuTile (tiled model)

# Override sim params
pixi run python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

## Kernel variants

| `backend=` | What it does |
|---|---|
| `jax` | The JAX/XLA baseline: `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), closed-form StVK stress. Every other kernel reuses this G2P, so only the P2G varies. |
| `cuda_v1` | CUDA P2G (one thread/particle, global `atomicAdd`) + JAX baseline G2P. |
| `cuda_v2` | CUDA warp-shuffle coalesced P2G + JAX baseline G2P. |
| `cuda_v3` | CUDA Morton-sorted P2G + JAX baseline G2P. (XLA command-buffer / CUDA-Graph capture is on for every kernel via the default env's `XLA_FLAGS`.) |
| `cuda_v4` | CUDA super-cell-owned grid tile P2G + JAX baseline G2P. |
| `cutile_v1` | cuTile direct 27-stencil scatter comparison backend. |
| `cutile_v3` | cuTile home-cell tiled P2G with local 27-node reduction + JAX baseline G2P. Requires `cuda-tile`. |

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (StVK) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

The solver is class-based:

- **`MPMSolver`** is a plain Python class. Particle/grid state is mutated in
  place by the driver API; the backend, constitutive closure, sticky-floor mask,
  staged callables, and compiled `_frame` are fixed for the solver's lifetime.
  `step()` advances one substep with individually jitted stages; `_frame`
  advances `steps_per_frame` substeps as a single XLA program via a `lax.fori_loop`
  over the same pure substep, and `run()` uses `_frame` by default.

Construction (`RuntimeConfig` + `MPMSolver` in `src/mpm_jax/solver.py`):
- Hydra instantiates `cfg.solver` into `RuntimeConfig`; backend choices are Python-backed hydra-zen registrations in `src/mpm_jax/p2g/backends/`, with each backend passing `num_grids` for validation. `simulate.py` / `profile_nsight.py` import `mpm_jax.p2g.backends` before composition, then call `MPMSolver(hydra.utils.instantiate(cfg.solver))`.
- `MPMSolver` reads the runtime config and builds params (with derived dx/vol/p_mass), particles, and initial state. The backend object is already instantiated by Hydra and owns CUDA/cuTile registration and grid-divisibility validation; the sticky floor is fixed in the solver frame.
- `src/mpm_jax/p2g/backends/` is a small P2G backend hierarchy. Variants override `prepare()` when they need ordering and `scatter()` for the P2G kernel. The implementation modules register the user-facing Hydra choices (`jax`, `cuda_v1`, etc.) directly via hydra-zen. The solver substep calls `backend.prepare()`, `backend.scatter()`, then the shared `g2p_mls()` path.

All solver variants now run through the same JAX-owned frame loop. The pure-JAX path compiles the entire frame (multiple substeps) as one XLA program. The CUDA variants (`cuda_v*`) move per-particle stencil work into CUDA kernels so the `(N, 27, *)` intermediate tensors never materialize in HBM. The cuTile variant (`cutile`) launches a tiled-programming-model P2G kernel from inside that same JAX frame via the cuTile/JAX bridge.

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
pixi run python simulate.py -cn sweep_all
pixi run python simulate.py -cn sweep_particle_count
pixi run python simulate.py -cn sweep_particle_density
pixi run python simulate.py -cn sweep_weak_scaling
```

The benchmark preset uses one frame with 50 substeps. `sweep_particle_count`
uses fixed `G=96` and particle counts `2^18..2^24`. `sweep_particle_density`
fixes `N=10M` and compares `G=32,64,96,128,160,192`.
`sweep_weak_scaling` keeps active-cell PPC near the benchmark density
(`particles_per_active_cell ~= 8.492`) while scaling both `G` and `N`.

Each combination gets its own
`outputs/sweeps/<gpu-kind>/runs/<date>/<time>/<job>_<override-dirname>/` subdir
with a flat `results.json`, `metrics.jsonl`, and single-row `metrics.csv`. The
same record is appended to `outputs/sweeps/<gpu-kind>/results.csv`, so pandas can
load a completed sweep with one call. The record includes `n_particles`,
`num_grids`, total `grid_cells`, `particles_per_grid_cell`,
`particles_per_active_cell`, throughput, timing, GPU type, and the Hydra
override string.
Runs with `render.enabled=true` also place `render.gif` in that same run
directory and record its path as `render_path`.

For an ad-hoc sweep:
`pixi run python simulate.py -m sim.n_particles=5000,50000,200000 sim.num_grids=32,64,96 backend=jax,cuda_v2 render.enabled=false`.

To load all sweep rows for one GPU:

```bash
pixi run python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("outputs/sweeps/<gpu-kind>")
df = pd.read_csv(root / "results.csv")
print(df.sort_values(["kernel", "num_grids", "n_particles"]))
PY
```

For all GPU folders at once, concatenate each `outputs/sweeps/*/results.csv`.

To generate figures:

```bash
pixi run python tools/plot_sweeps.py
# or
pixi run plot-sweeps
```

Plots and per-plot summary CSVs are written to `figures/sweeps/<gpu-kind>/`.

## Profiling

### Nsight Compute

Use the dedicated Nsight Python entrypoint for NCU metrics on one target
(`frame`, `p2g`, `prepare`, or `scatter`):

```bash
pixi run python profile_nsight.py -cn nsight_profile \
    backend=cutile_v3 nsight.target=scatter sim.n_particles=4096
```

**Metric presets** select what NCU collects (`nsight.analyze.metric_preset`):
`time`, `speed_of_light`, `roofline`, `atomics`, `memory_locality`, `occupancy`,
`scheduler`, or `full` (everything in one pass-set). Or pass any NCU metric list
directly via `nsight.analyze.metrics=[...]`. The `roofline` preset needs
`replay_mode=kernel` (the default) — its `.peak_sustained` ceilings are derived
by NCU from the running chip, so they are architecture-correct on A100/H100/GH200
with no datasheet constants.

**Cross-backend analysis figures** (hierarchical fp32 roofline, atomic-scatter,
occupancy, warp-stall breakdown) for the custom P2G kernels — one NCU sweep, then
plot:

```bash
pixi run python profile_nsight.py -cn nsight_profile nsight.target=scatter \
    nsight.analyze.metric_preset=full nsight.analyze.derive_metric=full \
    'nsight.sweep.kernels=[cuda_v1,cuda_v2,cuda_v3,cuda_v4,cutile_v1,cutile_v3]'
pixi run python postprocessing/nsight_plots.py <run_dir>/nsight_results.json -o nsight_figs/
```

For interactive Nsight Compute, launch `simulate.py` directly with the Pixi env
Python. `simulate.py` warms once, then wraps only the measured solve loop in
an NVTX range. In the GUI, enable NVTX support, leave CPU call stack off, set
cache control to "Flush All" for a first reproducible report, keep Import SASS
enabled, and set Import Source to yes when you want source pages:

```text
Application Executable: /root/MPM-CudaJax/.pixi/envs/default/bin/python
Working Directory:      /root/MPM-CudaJax
Arguments:              simulate.py sim=benchmark backend=cutile_v3 render.enabled=false profile.step_mode=staged
```

The `sim=benchmark` preset is one frame with one substep, so the measured solve
range stays small. `profile.step_mode=staged` makes the live solver run one
substep through individually jitted stages with NVTX ranges around
`elasticity`, `prepare`, `scatter`, `grid_update`, and `g2p`.
In the API Stream, use **Run to Next Range Start** to land on the
`cutile_v3_solve` / `cutile_v3_scatter` NVTX ranges, then **Run to Next Kernel**
and **Profile Kernel**. The cuTile kernel names show up as
`cutile_v1_p2g_kernel...` or `cutile_v3_p2g_kernel...`; earlier kernels in the
same scatter range are JAX/XLA helper kernels.
For fuller JAX/XLA source-location metadata while profiling, paste this into the
GUI environment editor, one variable per row:

```text
JAX_PLATFORMS=cuda
JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_FLAGS=--xla_gpu_enable_command_buffer=
JAX_COMPILATION_CACHE_DIR=/root/MPM-CudaJax/.jax_cache
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
```

If the GUI shows a single text field, paste the same environment as a
semicolon-separated list:

```text
JAX_PLATFORMS=cuda;JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1;XLA_PYTHON_CLIENT_PREALLOCATE=false;XLA_FLAGS=--xla_gpu_enable_command_buffer=;JAX_COMPILATION_CACHE_DIR=/root/MPM-CudaJax/.jax_cache;JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0;JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0;
```

For graph-captured benchmark behavior instead, use
`XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE`.

For a cleaner Nsight Compute API stream, use staged solver mode. It builds the
same Hydra-selected solver, warms the per-stage JITs, then runs
`solver.step(profile=True)` through `simulate.py` with NVTX ranges around
`elasticity`, `prepare`, `scatter`, `grid_update`, and `g2p`:

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
pixi run python simulate.py \
    sim=benchmark backend=cutile_v3 render.enabled=false \
    profile.step_mode=staged
```

### JAX / XProf

The JAX profiler captures the trace; **[XProf](https://openxla.org/xprof)** is
the viewer. There is a baked-in `conf/trace.yaml`: the standard `sim=benchmark`
preset shortened to 5 substeps × 2 frames, with profiling on and rendering off.

```bash
pixi run python simulate.py -cn trace backend=cutile_v3        # one backend
pixi run python simulate.py -cn trace -m backend=jax,cuda_v3   # several
```

Or enable profiling on any run with `profile.enabled=true`. The capture includes
CUDA streams + graphs, HLO graph/op stats, and memory; warmup (JIT compilation)
runs outside the trace, and each frame is a `StepTraceAnnotation` step. Traces
land in `traces/<label>/` (one run per backend, `label` defaults to the backend
name) so the viewer lists them side by side. `gpu_enable_cupti_activity_graph_trace`
(on by default) traces kernels *inside* the CUDA graphs.

There's also `conf/trace_substep.yaml` — a single isolated substep
(`label=jax_p2g_substep`) for inspecting the P2G fusions/HLO in the Graph Viewer.

**Viewing a remote run via SSH tunnel:**

```bash
ssh -L 6006:localhost:6006 <host>                              # DTU HPC: add -J <login> <compute>
cd ~/MPM-CudaJax && pixi run xprof --logdir traces --port 6006  # on the remote
# then open http://localhost:6006 locally
```

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `jelly` (default) | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `backend` | `jax` (default), `cuda_v1`, `cuda_v2`, `cuda_v3`, `cuda_v4`, `cutile_v1`, `cutile_v3` | P2G implementation (G2P shared) |

Top-level fields: `tag`, `render`. All overridable from CLI:

```bash
pixi run python simulate.py sim.n_particles=100000 backend=cuda_v3 render.enabled=false
```

XLA command-buffer / CUDA-Graph capture is always on via `XLA_FLAGS` in the
default env — no per-kernel flag. For profiling runs where richer XLA
annotations matter more than exact graph-captured timing, temporarily override
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`.

## Tests

```bash
pixi run test
```

Run focused GPU checks:

```bash
pixi run pytest tests/test_cuda_ffi_loader.py tests/test_p2g_scan.py \
    tests/test_cuda_v2_matches_v1.py -q
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
        └── p2g/
            ├── scan.py      # JAX scan P2G
            ├── sort.py      # morton_argsort, home_super_cell_id, home_cell_id
            ├── backends/    # backend implementations + hydra-zen registrations
            ├── cutile/      # cuTile P2G kernels + jax bridges
            └── cuda/
                ├── p2g_cuda.py  # FFI capsule registration + kernel objects
                └── kernels/     # p2g_ffi_module.cc plus p2g_v1.cu,
                                 # p2g_v2.cu, p2g_v3.cu, p2g_v4.cu
```

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
