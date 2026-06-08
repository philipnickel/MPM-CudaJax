# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels and an **NVIDIA cuTile** (tiled
programming model) kernel. Investigates where JAX/XLA's automatic GPU
compilation is sufficient and where custom kernels win.

The solver is **constructed from config** by Hydra-instantiating `cfg.solver`
into a `RuntimeConfig` and passing it to `MPMSolver`: the `backend` config
targets the backend class directly, and `MPMSolver` builds params, particles,
boundaries, and initial state. `solver.run(capture_frames=...)` drives the
configured jitted frame loop.

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
    sim=benchmark
```
Prints `total_steps`, `elapsed_s`, `steps_per_sec`, average `ms/step`,
high-level `particles_per_sec`, and the detected `gpu_type`. The benchmark
preset disables GIF rendering and per-frame state capture by default.

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

CMake defaults to `native` CUDA architecture autodetection during `pixi install`.
Cross-build hosts that need a fixed architecture should set that in Pixi
task/environment configuration, not by prefixing ad hoc run commands.

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
pixi run python simulate.py sim=benchmark

# Pick a kernel
pixi run python simulate.py backend=jax                              # JAX/XLA baseline (scan P2G + MLS G2P)
pixi run python simulate.py backend=cuda_v1 material=jelly
pixi run python simulate.py backend=cuda_v2 material=jelly            # Morton-sorted warp-shuffle coalescing
pixi run python simulate.py backend=cuda_v3 material=jelly            # super-cell grid tile
pixi run python simulate.py backend=CuTile material=jelly sim=benchmark  # cuTile (tiled model)

# Override sim params
pixi run python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

## Kernel variants

| `backend=` | What it does |
|---|---|
| `jax` | The JAX/XLA baseline: `lax.scan` over the 27 offsets for **both** P2G and G2P, unified MLS-MPM G2P (APIC affine `C` reused as ∇v), closed-form StVK stress. Every other kernel reuses this G2P, so only the P2G varies. |
| `cuda_v1` | CUDA P2G (one thread/particle, global `atomicAdd`) + JAX baseline G2P. |
| `cuda_v2` | CUDA Morton-sorted warp-shuffle coalesced P2G + JAX baseline G2P. |
| `cuda_v3` | CUDA super-cell-owned grid tile P2G + JAX baseline G2P. |
| `CuTile` | cuTile home-cell tiled P2G with local 27-node reduction + JAX baseline G2P. Requires `cuda-tile`. |

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (StVK) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

The solver is class-based:

- **`MPMSolver`** is a plain Python class. Particle/grid state is mutated in
  place by the driver API; the backend, constitutive closure, sticky-floor mask,
  and compiled `_frame` are fixed for the solver's lifetime. `_frame` advances
  `steps_per_frame` substeps as a single XLA program via a `lax.fori_loop` over
  the pure substep, and `run()` drives that compiled frame loop.

Construction (`RuntimeConfig` + `MPMSolver` in `src/mpm_jax/solver.py`):
- Hydra instantiates `cfg.solver` into `RuntimeConfig`; backend choices are Python-backed hydra-zen registrations in `src/mpm_jax/p2g/backends/`, with each backend passing `num_grids` for validation. `simulate.py` / `profile_nsight.py` import `mpm_jax.p2g.backends` before composition, then call `MPMSolver(hydra.utils.instantiate(cfg.solver))`.
- `MPMSolver` reads the runtime config and builds params (with derived dx/vol/p_mass), particles, and initial state. The backend object is already instantiated by Hydra and owns CUDA/cuTile registration and grid-divisibility validation; the sticky floor is fixed in the solver frame.
- `src/mpm_jax/p2g/backends/` is a small P2G backend hierarchy. Variants override `prepare()` when they need ordering and `scatter()` for the P2G kernel. The implementation modules register the user-facing Hydra choices (`jax`, `cuda_v1`, etc.) directly via hydra-zen. The solver substep calls `backend.prepare()`, `backend.scatter()`, then the shared `g2p_mls()` path.

All solver variants now run through the same JAX-owned frame loop. The pure-JAX path compiles the entire frame (multiple substeps) as one XLA program. The CUDA variants (`cuda_v*`) move P2G stencil work into CUDA kernels so the `(N, 27, *)` intermediate tensors never materialize in HBM; v2 sorts particles by Morton code for warp-shuffle coalescing, while v3 uses a super-cell-owned shared-memory tile. The cuTile variant (`cutile`) launches a tiled-programming-model P2G kernel from inside that same JAX frame via the cuTile/JAX bridge.

To time P2G `prepare`, P2G `scatter`, `prepare+scatter`, and full solver
`ms/step` at the benchmark operating point (`G=128`, `N=10M`), run:

```bash
pixi run python tools/benchmark_p2g_substeps.py
```

The tool writes `p2g_substep_timings.parquet` in the Hydra output directory.

## Sweeps

Pre-baked Hydra multirun sweeps share one entry point `conf/sweep.yaml`. It
holds the shared bits (multirun mode, every backend, `sim=benchmark`, render
off) and selects a scaling axis from the `conf/sweep/` config group:

```bash
pixi run python simulate.py -cn sweep                       # particle_count (default)
pixi run python simulate.py -cn sweep sweep=particle_count  # constant grid, particle count up
pixi run python simulate.py -cn sweep sweep=weak_scaling    # constant active PPC, particle count up
pixi run sweep-sm                                           # CUDA MPS active-thread % sweep
```

The benchmark preset uses one frame with 10 substeps. `particle_count` uses
fixed `G=128` and the configured particle-count axis. `weak_scaling` keeps
active-cell PPC near the benchmark density (`particles_per_active_cell ~=
9.31`) while scaling both `G` and `N`.

The MPS SM-scaling sweep is the special process-level axis: CUDA reads
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` when JAX creates its CUDA context, so
`run_sm_scaling.sh` loops over percentages and launches normal
`simulate.py -cn sweep sweep=sm_scaling` multiruns. That config writes to the
static `outputs/sweeps/<gpu-kind>/sm_scaling` directory so each percentage
aggregates into one plot set.

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

Use the dedicated Nsight Python entrypoint for NCU metrics on the custom
CUDA/cuTile P2G scatter kernels. Do not use this path for `backend=jax`; the JAX
baseline lowers to several XLA-generated kernels and belongs in the XProf trace
workflow below. The profiler prepares inputs outside the annotation, warms the
jitted scatter once, then profiles the warmed P2G scatter call:

```bash
pixi run python profile_nsight.py -cn nsight_profile \
    backend=CuTile sim.n_particles=4096
```

`conf/nsight_metrics/` owns the direct NCU metric presets passed to
`nsight.analyze.metrics`: `timing`, `roofline`, and `full` (the default). You
can still override `nsight.analyze.metrics=[...]` directly for focused runs.
The `full` preset includes the NCU counters needed by the roofline, atomics,
occupancy, and scheduler plots.

**Cross-backend analysis figures** (hierarchical fp32 roofline, atomic-scatter,
occupancy, warp-stall breakdown) for the custom P2G kernels. One Hydra job
profiles one backend variant; the `conf/nsight_sweep/` group defines serial
Hydra multiruns over the custom backends. Each job calls
`ProfileResults.to_dataframe()`, writes the processed `nsight-python` metric-row
DataFrame to `nsight_metrics.parquet` in its Hydra output dir, and appends those
rows to sweep-level `results.parquet` under
`outputs/nsight/<gpu>/sweeps/<date>/<time>/`. The `.ncu-rep` report from Nsight
Compute is saved in the same per-job Hydra output dir. At multirun end,
`conf/nsight_plot/standard.yaml` installs `NsightPlotCallback`, which loads that
authoritative `results.parquet` with pandas and renders the figures:

The MPS SM-percentage Nsight sweep is intentionally static at
`outputs/nsight/<gpu>/sm_scaling/`, because each percentage is a separate Python
process and all percentages need to append into one trajectory aggregate.

```bash
# Single operating point (all custom P2G kernels at the benchmark resolution):
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=single_point
```

The default `nsight_profile.yaml` uses `replay_mode=kernel`, drops known tiny XLA
wrapper kernels around cuTile scatter, and has no default kernel combiner. Nsight
Python will fail if more than one non-ignored kernel remains unless
`nsight.analyze.combine_kernel_metrics` is explicitly overridden.

**Scaling roofline trajectories.** Add a scale axis to the multirun and
`nsight_plots.py` also emits `roofline_scaling.png` — a hierarchical roofline where
each backend is a connected L1/L2/HBM trajectory. The axis is auto-detected
from what varies:

```bash
# Load sweep (throughput vs problem size): fixed grid, growing particle count
# (raises ppc/density). Resources are constant, so this is NOT strong scaling.
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=particle_count

# Weak scaling: fixed particles-per-cell (~9.31), grid + N grow together.
pixi run python profile_nsight.py -cn nsight_profile nsight_sweep=weak

# MPS SM-percentage scaling: fixed benchmark point, fresh process per percent.
# For a roofline-only collection, add: nsight_metrics=roofline nsight_plot=roofline_only
pixi run nsight-sweep-sm

# Manual re-render from an existing aggregate:
pixi run python postprocessing/nsight_plots.py \
    outputs/nsight/<gpu>/sweeps/<date>/<time>/results.parquet -o figures/nsight/<gpu>/
```

For interactive Nsight Compute, launch the GUI through Pixi so the profiled
process inherits the default runtime environment.
`simulate.py` warms once, then wraps only the measured solve loop in an NVTX
range. In the GUI, enable NVTX support, leave CPU call stack off, set cache
control to "Flush All" for a first reproducible report, keep Import SASS
enabled, and set Import Source to yes when you want source pages:

```text
Application Executable: /root/MPM-CudaJax/.pixi/envs/default/bin/python
Working Directory:      /root/MPM-CudaJax
Arguments:              simulate.py sim=benchmark backend=CuTile
```

The `sim=benchmark` preset is one frame with 10 substeps, so the measured solve
range is the jitted frame containing the configured substep loop. In the API
Stream, use **Run to Next Range Start** to land on the `CuTile_solve` NVTX
range, then **Run to Next Kernel** and **Profile Kernel**. The cuTile kernel names show up as
`cutile_p2g_kernel...`; earlier kernels in the
same solve range are JAX/XLA helper kernels.
The GUI environment editor can stay empty; Pixi owns the runtime environment.

### JAX / XProf

The JAX profiler captures the trace; **[XProf](https://openxla.org/xprof)** is
the viewer. There is a baked-in `conf/trace.yaml`: the standard `sim=benchmark`
preset shortened to 3 substeps x 2 frames, with profiling on and rendering off.

```bash
pixi run python simulate.py -cn trace backend=CuTile        # one backend
pixi run python simulate.py -cn trace -m backend=jax,cuda_v3   # several
```

Or enable profiling on any run with `profile.enabled=true`. The capture includes
CUDA streams, HLO graph/op stats, and memory; warmup (JIT compilation) runs
outside the trace, and each frame is a `StepTraceAnnotation` step. Traces land in
`traces/<label>/` (one run per backend, `label` defaults to the backend name) so
the viewer lists them side by side. Trace runs disable XLA command buffers before
JAX initializes so XProf can show kernels and named scopes inside the compiled
frame; opt out with `profile.disable_command_buffers=false`.

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
| `backend` | `jax` (default), `cuda_v1`, `cuda_v2`, `cuda_v3`, `CuTile` | P2G implementation (G2P shared) |

Top-level fields: `tag`, `render`. All overridable from CLI:

```bash
pixi run python simulate.py sim.n_particles=100000 backend=cuda_v3 render.enabled=false
```

Runtime environment variables live in `pyproject.toml`: CUDA/JAX activation,
single-host multi-device NCCL tuning, and the persistent compile-cache settings
are in the default `gpu` feature. XProf trace collection uses
`jax.profiler.ProfileOptions`, and `simulate.py` adds the trace-only
`XLA_FLAGS` command-buffer override before importing JAX.

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
│   ├── sweep.yaml          # sweep entry point + shared bits
│   └── sweep/              # scale-axis config group: all, particle_count, ...
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
                                 # p2g_v2.cu, p2g_v3.cu
```

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
