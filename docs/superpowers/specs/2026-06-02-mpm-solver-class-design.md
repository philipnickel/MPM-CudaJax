# MPMSolver class + blocks/stepping/registry reorg — design

**Date:** 2026-06-02
**Status:** Approved (pending written-spec review)
**Author:** brainstormed with Claude

## Problem

The codebase works but the dispatch layer is un-Pythonic and hard to extend:

- `simulate.run_jax` is a ~150-line `if/elif kernel_name` chain (carrying a literal
  `#TODO: the below is a mess` comment) that prints a per-kernel blurb, selects one of
  eight near-duplicate `build_jit_frame_*` factories, maintains a hand-kept
  warmup-exclusion set, and runs the timed loop.
- Each kernel variant has its own `build_jit_frame_*` function (4 in `cuda/p2g_cuda.py`,
  3 in `warp_p2g.py`, 1 in `p2g_scan.py`, 1 default in `solver.py`). They share almost the
  same scaffold (elasticity → pre-BC → weights → P2G → grid update → post-BC → G2P →
  plasticity) and differ only in the P2G/G2P core, an optional per-substep sort, or a
  graph wrap — but the scaffold is duplicated in each.
- The pure-Warp graph path (`warp_bonus_graph`/`warp_bonus_v2_graph`) is already a class
  (`WarpBonusSimulator`) living in its own world, reached through a separate
  `run_warp_bonus` dispatch branch.

There is no single object that owns "a simulation": geometry, parameters, state, and the
compiled step all get re-assembled inline in `simulate.py` for every run.

## Goals

1. A class-based `MPMSolver` that, at construction, builds the grid/particles/params/BCs
   and initial state, and exposes `step()` and `solve()`.
2. "Which kernel" becomes data (a registry key), not control flow.
3. Both execution models — JAX-functional and pure-Warp graph — present the **same**
   `step()`/`solve()` interface.
4. Kill the duplication: shared math lives in small pure modules; each variant is a thin,
   readable composition.
5. Preserve current behavior: numerical parity, whole-frame JIT (one XLA program per
   frame), and the benchmark timing contract (no `block_until_ready` in the timed region).

## Non-goals

- No new kernel variants or numerical changes. This is a structural refactor only.
- No performance regression and no intended performance *gain* — parity is success.
- Not changing the Hydra config-group concept, the constitutive/boundary registries, or
  the `.cu` sources.

## Key decisions (from brainstorming)

| # | Decision |
|---|----------|
| 1 | One `step()`/`solve()` interface spans both backends. |
| 2 | Base `MPMSolver` = JAX functional core wrapped in a stateful shell. `WarpGraphSolver` **subclasses** it and overrides the build + loop for the mutable capture/replay model. |
| 3 | Solver owns the invariant scaffold + loop + state; a per-variant `build_frame` function (composed from shared blocks) owns what varies. No heavyweight polymorphic "Kernel" object — the variants differ structurally (sort / fused FFI / graph wrap) and a rigid method signature would leak. |
| 4 | Full `src/` layout. |
| 5 | Big-bang migration: all callers (simulate.py, tests, configs, docs) updated in one coordinated change; suite green at the end. |
| 6 | Collapse flag-only variants: `loop_kind` and `cuda_graph` become parameters; `cuda_v2_fori_inline`, `cuda_v3_fori_inline`, `cuda_v6_inline` are removed as `kernel=` names. |
| 7 | The `src/` move + build rewiring is an **isolated, verified-green first step** before any logic refactor. |

## Execution model: substep / frame / step / solve

- **`_substep(state) → state`** — one MPM substep, composed from `blocks/`. Pure. The atomic unit.
- **frame** — a chunk of `steps_per_frame` substeps; the compiled boundary and the IO cadence.
  This is the **only** place backends differ:
  - **JAX base:** a `@jax.jit` `lax.fori_loop` over `_substep` (rolled, not Python-unrolled →
    faster compile). This makes today's `loop_kind='fori'` the default; `'python'` (unrolled)
    stays selectable for variants/debugging.
  - **Warp subclass:** the native scoped CUDA-graph capture, replayed.
- **`step()`** — advance one frame-chunk; reassigns and returns `self.state`.
- **`solve(num_frames, on_frame=None)`** — loop `step()`. The `on_frame(frame_idx, state)` hook
  is where per-frame IO lives (GIF capture, metric reads, DLPack handoff to the Warp HashGrid).
  Benchmark mode passes no hook and syncs once at the end, preserving the
  "no `block_until_ready` in the timed region" contract.

### JIT discipline (the load-bearing constraint)

JAX jits *functions*, not objects. `self` is never traced. `build_frame` captures static
config (params, closures, `steps_per_frame`, `loop_kind`, `cuda_graph`) in a closure and
returns a `@jax.jit` **pure** function, stored once as `self._frame`. `step()`/`solve()`
only *call* it. This is exactly what `build_jit_frame` does today — moved into a class
attribute. The whole frame remains a single XLA program.

## Target module structure

```
src/mpm_jax/
  types.py            MPMState, MPMParams, StepIntermediates, make_params   (from solver.py)
  blocks/             pure functions, no orchestration, no jit
    weights.py        _single_particle_weights, compute_weights_and_indices, OFFSET_27
    p2g.py            _single_particle_p2g, p2g_compute, p2g_scatter, p2g (jax default)
    g2p.py            _single_particle_g2p, g2p
    grid.py           grid_update, build_grid_x
    svd.py            jacobi_svd                                            (from jacobi_svd.py)
    sort.py           morton sort, cell sort                               (from morton.py)
  stepping/           build_*_frame(params, fns, steps_per_frame, **opts) -> jit'd pure fn
    jax_frames.py     build_jax_frame, build_jax_v1_5_frame                (absorbs p2g_scan.py)
    cuda_frames.py    build_cuda_v1/v2/v3/v4_frame  (v2/v3 take loop_kind, cuda_graph)
    warp_frames.py    build_warp_v1/v2_tile/v3_supercell_frame
  cuda/
    ffi.py            FFI registration + op wrappers                       (from p2g_cuda.py)
    kernels/*.cu      unchanged
    _lib/             built .so (CMake output path updated to src/...)
  warp_kernels.py     Warp @wp.kernel defs                                 (from warp_p2g.py)
  warp_graph.py       pure-Warp capture/replay engine                      (from warp_bonus.py)
  constitutive.py     unchanged (REGISTRY)
  boundary.py         unchanged (build_boundary_fns)
  solver.py           MPMSolver (base), WarpGraphSolver (subclass)
  registry.py         KERNELS: name -> KernelSpec(solver_cls, build_frame, defaults)
```

`stepping/jax_frames.py` avoids shadowing the `jax` package. Names are provisional.

## Solver classes

```python
class MPMSolver:                       # base = JAX functional core, stateful shell
    def __init__(self, params, *, elasticity_fn, plasticity_fn,
                 pre_fn, post_fn, build_frame, steps_per_frame, init_state,
                 **frame_opts):                           # loop_kind / cuda_graph / etc
        self.params = params
        self.state  = init_state                          # MPMState (mutable attribute)
        self._frame = build_frame(params, elasticity_fn, plasticity_fn,
                                  pre_fn, post_fn, steps_per_frame,
                                  **frame_opts)            # jit'd pure fn
    def step(self):
        self.state = self._frame(self.state); return self.state
    def solve(self, num_frames, on_frame=None):
        for f in range(num_frames):
            self.step()
            if on_frame: on_frame(f, self.state)
        return self.state
    def reset(self, init_state):
        self.state = init_state

class WarpGraphSolver(MPMSolver):      # pure-Warp graph backend (warp_bonus_*)
    # reuses base geometry/params/state setup; builds a warp_graph capture engine
    # instead of a jit'd _frame; overrides step()/solve() for capture/replay.
```

`WarpGraphSolver` subclasses `MPMSolver` so the shared `__init__` setup (params, grid,
particles, BCs, initial state) is reused; only the build step and `step()`/`solve()` diverge.
The shared surface is `step()` / `solve()` / `self.state`.

## Registry + config

```python
@dataclass(frozen=True)
class KernelSpec:
    solver_cls: type
    build_frame: Callable
    defaults: dict = field(default_factory=dict)   # e.g. loop_kind, cuda_graph, indexed_sort

# build_solver(cfg) lives in registry.py: looks up KERNELS[cfg.kernel.name], builds
# params/grid/particles/BCs/init_state from cfg, merges spec.defaults with kernel-cfg
# overrides into frame_opts (each build_frame ignores opts it doesn't use), and returns
# spec.solver_cls(params, ..., build_frame=spec.build_frame, **frame_opts).

KERNELS = {
  "jax":                    KernelSpec(MPMSolver,      build_jax_frame),
  "jax_v1_5":               KernelSpec(MPMSolver,      build_jax_v1_5_frame),
  "cuda_v1_inline":         KernelSpec(MPMSolver,      build_cuda_v1_frame),
  "cuda_v2_inline":         KernelSpec(MPMSolver,      build_cuda_v2_frame),   # loop_kind from cfg
  "cuda_v3_inline":         KernelSpec(MPMSolver,      build_cuda_v3_frame),   # loop_kind + cuda_graph from cfg
  "cuda_v4_inline":         KernelSpec(MPMSolver,      build_cuda_v4_frame),
  "warp_v1_inline":         KernelSpec(MPMSolver,      build_warp_v1_frame),
  "warp_v2_tile":           KernelSpec(MPMSolver,      build_warp_v2_tile_frame),
  "warp_v3_supercell_tile": KernelSpec(MPMSolver,      build_warp_v3_frame),
  "warp_bonus_graph":       KernelSpec(WarpGraphSolver, build_warp_graph),
  "warp_bonus_v2_graph":    KernelSpec(WarpGraphSolver, build_warp_graph, {"indexed_sort": True}),
}
```

- `loop_kind: python|fori` and `cuda_graph: bool` move into `conf/kernel/cuda_v2_inline.yaml`
  and `conf/kernel/cuda_v3_inline.yaml`.
- Removed `kernel=` names: `cuda_v2_fori_inline`, `cuda_v3_fori_inline`, `cuda_v6_inline`
  (their `conf/kernel/*.yaml` deleted). The deprecated-error branches (`cuda_v1`, `cuda_v2`,
  `cuda_v4`, `cuda_fused`) become a small "removed kernels" lookup that raises with guidance.

## `simulate.py` after the refactor

The `run_jax`/`run_warp_bonus` split, the `if/elif`, the print blurbs, the warmup-exclusion
set, and `_maybe_enable_cuda_graphs` collapse to roughly:

```python
solver = build_solver(cfg)             # registry.py: params, grid, particles, BCs, init state, frame
solver.step()                          # warmup (one frame) + block_until_ready
# benchmark: timed loop of solver.step(); one sync at end
# render:    solver.solve(num_frames, on_frame=capture_and_metrics)
```

`build_solver(cfg)` is a classmethod/factory that reads the Hydra config and constructs the
right solver via the registry. Per the existing TODO, the manual per-kernel prints are
dropped — Hydra/OmegaConf already prints the resolved config. The XLA-flag toggling for
CUDA graphs moves behind the `cuda_graph` parameter (must still run before `import jax`).

## Migration plan (big-bang, but src move isolated first)

**Step 0 — isolated, verified green:** move the package to `src/mpm_jax/` with **no logic
changes**. Update `[tool.scikit-build] wheel.packages`, the editable pypi path in
`pyproject.toml`, and CMake `KERNEL_OUT_DIR` → `src/mpm_jax/cuda/_lib`. The FFI loader uses
`Path(__file__).parent`, so it follows the move. **Gate:** `pixi install -e gpu` rebuilds the
`.so` files, `import mpm_jax.cuda.p2g_cuda` (pre-rename) loads them, and the full test suite
passes. Nothing else proceeds until this is green.

**Step 1+ — the refactor** on the moved tree: introduce `blocks/`, `stepping/`, `registry.py`,
the solver classes; repoint `simulate.py`; update tests and configs; delete the old
`build_jit_frame_*` functions and removed kernel configs; update CLAUDE.md + README.

## Testing

- Equivalence tests (`test_cuda_v2_inline_matches_v1`, `test_warp_v2_tile_matches_v1`,
  `test_warp_v3_supercell_tile_matches_v1`, `test_jax_v1_5`, etc.) repoint to the relocated
  builders and keep asserting numerical parity. Parity holds because block math is moved
  verbatim, not rewritten.
- `test_cuda_ffi_loader.py` validates the `.so` load path after the `src/` move.
- Add a small `test_solver_api` covering: registry lookup builds the right class; `step()`
  advances state; `solve(n)` == `n × step()`; `on_frame` fires `n` times; a JAX kernel and a
  Warp-graph kernel both satisfy the interface.
- Smoke: one short `simulate.py` run per representative kernel (`jax`, `cuda_v3_inline`,
  `warp_v1_inline`, `warp_bonus_graph`) in benchmark mode.

## Risks

1. **`src/` + scikit-build editable + CUDA `.so` path** — the main risk; mitigated by making
   Step 0 isolated and gated on a clean rebuild + import + green suite.
2. **`WarpGraphSolver` fit** — mapping `WarpBonusSimulator`'s `capture`/`run_frames` onto
   `step()`/`solve()` without contorting the base. If subclassing proves leaky, fall back to a
   shared ABC with two siblings (revisit, don't force).
3. **`lax.fori_loop` as frame default** — must accommodate FFI custom calls, per-substep
   Morton sort, and Warp tile calls in the body. Already supported via the existing
   `loop_kind='fori'`, so low risk; `loop_kind='python'` remains the escape hatch.
4. **Benchmark contract** — `solve()`'s default (no `on_frame`) must not introduce any
   intra-loop sync; verify the timed path matches today's numbers.

## Success criteria

- `simulate.run_jax`'s `if/elif` chain and the eight `build_jit_frame_*` functions are gone;
  kernel selection is a registry lookup.
- `MPMSolver(...).solve(n)` reproduces today's GIF output; benchmark ms/step is within noise
  of the current numbers for `jax`, `cuda_v3_inline`, `warp_v1_inline`, `warp_bonus_graph`.
- Full suite green; CLAUDE.md + README reflect the new layout and collapsed kernel names.
