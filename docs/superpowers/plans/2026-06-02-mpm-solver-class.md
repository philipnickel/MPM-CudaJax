# MPMSolver class + blocks/stepping/registry reorg — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `simulate.run_jax` `if/elif` dispatch and the eight near-duplicate `build_jit_frame_*` factories with a class-based `MPMSolver` (+ `WarpGraphSolver` subclass), kernel selection via a registry, shared math in `blocks/`, and per-variant frame builders in `stepping/` — all under a `src/` layout, with no behavior change.

**Architecture:** A stateful `MPMSolver` holds `self.state` and a jit'd pure frame function built once at construction. `step()` advances one frame (a `lax.fori_loop` chunk of `steps_per_frame` substeps); `solve(num_frames, on_frame=None)` loops `step()` with an IO hook. `WarpGraphSolver` subclasses it and overrides the build/loop for pure-Warp CUDA-graph capture-replay. A `registry.py` maps each `kernel=` name to `(solver_cls, build_frame, defaults)`.

**Tech Stack:** Python 3.10+, JAX (CUDA 12 jaxlib), NVIDIA Warp, scikit-build-core + CMake (CUDA FFI kernels), Hydra/OmegaConf, pixi, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-02-mpm-solver-class-design.md`

**Conventions for every task below:**
- All commands run through pixi. CPU-only checks: `pixi run test`. GPU checks: `pixi run -e gpu pytest ...`. Build: `pixi install -e gpu`.
- "Suite green" means `pixi run test` passes (CPU-safe tests) and, on a GPU host, `pixi run -e gpu pytest tests/ -q` passes. GPU-only tests skip cleanly on CPU via `is_available()` guards — note in the run output which were skipped.
- Commit after every task. Never leave the tree red between tasks.

---

## Phase 0 — Isolated `src/` move (NO logic changes)

This phase is gated: nothing in Phase 1+ starts until the suite is green on the moved tree. The only risk being de-risked here is the scikit-build + CUDA `.so` path change.

### Task 0.1: Move the package under `src/`

**Files:**
- Move: `mpm_jax/` → `src/mpm_jax/` (entire tree, including `cuda/kernels/*.cu`)
- Modify: `pyproject.toml`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Move the package with git (preserves history + content verbatim)**

```bash
mkdir -p src
git mv mpm_jax src/mpm_jax
```

- [ ] **Step 2: Point scikit-build wheel packaging at the new path**

In `pyproject.toml`, change line 23:

```diff
- wheel.packages = ["mpm_jax"]
+ wheel.packages = ["src/mpm_jax"]
```

- [ ] **Step 3: Point CMake kernel source + output dirs at `src/`**

In `CMakeLists.txt`, update the two directory variables (currently lines 40 and 46):

```diff
- set(KERNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpm_jax/cuda/kernels)
+ set(KERNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mpm_jax/cuda/kernels)
```

```diff
- set(KERNEL_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mpm_jax/cuda/_lib)
+ set(KERNEL_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mpm_jax/cuda/_lib)
```

Also update the `install(... DESTINATION ...)` (line 63) to keep the wheel layout correct:

```diff
-    install(TARGETS ${name} LIBRARY DESTINATION mpm_jax/cuda/_lib)
+    install(TARGETS ${name} LIBRARY DESTINATION src/mpm_jax/cuda/_lib)
```

Note: the FFI loader in `src/mpm_jax/cuda/p2g_cuda.py` resolves `_lib` via `Path(__file__).resolve().parent / "_lib"` and `importlib.util.find_spec("mpm_jax.cuda._lib.<mod>")`, so it follows the move with no code change.

- [ ] **Step 4: Update `.gitignore` if it pins the old `_lib` path**

Run: `grep -n "_lib\|build" .gitignore`
If `mpm_jax/cuda/_lib` is referenced explicitly, change it to `src/mpm_jax/cuda/_lib` (a bare `_lib/` or `*.so` pattern needs no change).

- [ ] **Step 5: Rebuild and verify import + FFI load (GPU host)**

Run:
```bash
pixi install -e gpu
pixi run -e gpu python -c "import mpm_jax; from mpm_jax.cuda import p2g_cuda; print('available:', p2g_cuda.is_available())"
```
Expected: prints `available: True` (on a GPU host with kernels built). On CPU: `import mpm_jax` succeeds and `is_available()` is `False` without error.

- [ ] **Step 6: Run the full suite**

Run: `pixi run -e gpu pytest tests/ -q` (or `pixi run test` on CPU).
Expected: same pass/skip result as before the move. No import errors.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: move package under src/ (no logic changes)"
```

### Task 0.2: Smoke-test the runtime entry point after the move

**Files:** none (verification only)

- [ ] **Step 1: Run a tiny simulation through each backend family**

Run:
```bash
pixi run -e gpu python simulate.py kernel=jax sim.n_particles=4096 sim.num_frames=2 benchmark=true
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi sim.n_particles=4096 sim.num_frames=2 benchmark=true
pixi run -e gpu python simulate.py kernel=warp_bonus_graph material=jelly_jacobi sim.num_grids=32 sim.n_particles=4096 sim.num_frames=2 benchmark=true
```
Expected: each prints timing summary, no import/FFI errors.

- [ ] **Step 2: Commit (no-op marker if nothing changed)**

If Step 1 surfaced a path bug, fix it and commit `fix: <desc> after src move`. Otherwise no commit.

---

## Phase 1 — Extract shared building blocks

Goal: pull the pure math out of `solver.py` into `blocks/` and put state types in `types.py`, leaving thin re-exports so the suite stays green at each step. The old public names (`build_jit_frame`, `step`, `grid_update`, `MPMState`, ...) keep working until Phase 5 deletes them.

### Task 1.1: Create `types.py`

**Files:**
- Create: `src/mpm_jax/types.py`
- Modify: `src/mpm_jax/solver.py`

- [ ] **Step 1: Move the type/param definitions into `types.py`**

Create `src/mpm_jax/types.py` containing, moved verbatim from `solver.py`: `MPMState`, `StepIntermediates`, `MPMParams`, `OFFSET_27`, and `make_params` (current `solver.py:6-68` and `38-42`). Keep the imports they need (`NamedTuple`, `jax`, `jax.numpy as jnp`, `numpy as np`).

- [ ] **Step 2: Re-export from `solver.py` for backward compat**

At the top of `src/mpm_jax/solver.py`, replace the moved definitions with:

```python
from mpm_jax.types import (
    MPMState, StepIntermediates, MPMParams, OFFSET_27, make_params,
)
```

- [ ] **Step 3: Run the suite**

Run: `pixi run test`
Expected: PASS (CPU tests). `test_solver.py` imports `MPMState, MPMParams, make_params` — must still resolve via the re-export.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: extract state/param types into types.py"
```

### Task 1.2: Create `blocks/weights.py`, `blocks/p2g.py`, `blocks/g2p.py`, `blocks/grid.py`

**Files:**
- Create: `src/mpm_jax/blocks/__init__.py` (empty)
- Create: `src/mpm_jax/blocks/weights.py`, `blocks/p2g.py`, `blocks/g2p.py`, `blocks/grid.py`
- Modify: `src/mpm_jax/solver.py`

- [ ] **Step 1: Move pure functions into block modules (verbatim bodies)**

- `blocks/weights.py`: `_single_particle_weights`, `compute_weights_and_indices` (`solver.py:75-187`). Imports `OFFSET_27` from `mpm_jax.types`.
- `blocks/p2g.py`: `_single_particle_p2g`, `p2g_compute`, `p2g_scatter`, `p2g` (`solver.py:124-223`).
- `blocks/g2p.py`: `_single_particle_g2p`, `g2p` (`solver.py:149-176`, `238-243`).
- `blocks/grid.py`: `grid_update` (`solver.py:226-235`) and a new `build_grid_x(num_grids)` helper extracted from `simulate.run_jax:274-276`:

```python
import jax.numpy as jnp

def build_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
```

- [ ] **Step 2: Re-export from `solver.py`**

Add to `solver.py`:

```python
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.p2g import p2g_compute, p2g_scatter, p2g
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update
```

Keep `step`, `build_jit_step`, `build_jit_frame`, `simulate_frame`, `build_jit_stages` in `solver.py` for now (they orchestrate the blocks and are still imported by tests).

- [ ] **Step 3: Run the suite**

Run: `pixi run test`
Expected: PASS. `test_solver.py` imports `grid_update, step` from `mpm_jax.solver` — must resolve via re-export.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: extract weights/p2g/g2p/grid blocks"
```

### Task 1.3: Move SVD and sort helpers into `blocks/`

**Files:**
- Move: `src/mpm_jax/jacobi_svd.py` → `src/mpm_jax/blocks/svd.py`
- Move: `src/mpm_jax/morton.py` → `src/mpm_jax/blocks/sort.py`
- Modify: importers of those modules

- [ ] **Step 1: Find importers**

Run: `grep -rn "jacobi_svd\|import morton\|from mpm_jax.morton\|from mpm_jax.jacobi_svd" src/ tests/ simulate.py profile_nsight.py`

- [ ] **Step 2: Move with git**

```bash
git mv src/mpm_jax/jacobi_svd.py src/mpm_jax/blocks/svd.py
git mv src/mpm_jax/morton.py src/mpm_jax/blocks/sort.py
```

- [ ] **Step 3: Update every importer**

For each hit from Step 1, rewrite the import: `from mpm_jax.jacobi_svd import X` → `from mpm_jax.blocks.svd import X`; `from mpm_jax.morton import Y` → `from mpm_jax.blocks.sort import Y`. (These are used by the CUDA/Warp stepping code, e.g. `p2g_cuda.py` for the Morton sort path.)

- [ ] **Step 4: Run the suite**

Run: `pixi run -e gpu pytest tests/ -q` (GPU — the sort/svd consumers are in CUDA paths). On CPU: `pixi run test`.
Expected: PASS / same skips.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: move svd + sort helpers into blocks/"
```

---

## Phase 2 — Stepping layer

Goal: relocate the per-variant frame builders into `stepping/`, standardize their signature to accept `**frame_opts` (so the solver can forward `loop_kind` / `cuda_graph` uniformly), and make `lax.fori_loop` the default frame loop. The deprecated `build_jit_frame_*` names get re-exported until Phase 5.

### Task 2.1: Create `stepping/jax_frames.py`

**Files:**
- Create: `src/mpm_jax/stepping/__init__.py` (empty)
- Create: `src/mpm_jax/stepping/substep.py`
- Create: `src/mpm_jax/stepping/jax_frames.py`
- Modify: `src/mpm_jax/solver.py`, `src/mpm_jax/p2g_scan.py`

> **Avoid a circular import.** The substep orchestrator `step` is used by both the new `jax_frames.py` *and* the legacy functions left in `solver.py` (`build_jit_step`, `simulate_frame`, `build_jit_stages`). If `jax_frames` imports `step` from `solver` while `solver` imports `build_jax_frame` from `jax_frames`, that's a cycle. Fix: move `step` into `stepping/substep.py` (which imports only from `blocks/` + `types`), and have both `jax_frames.py` and `solver.py` import `step` from there. All imports point "down"; no cycle.

- [ ] **Step 1a: Move `step` into `stepping/substep.py`**

Move the `step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=None)` orchestrator (`solver.py:250-287`) verbatim into `src/mpm_jax/stepping/substep.py`. Its imports become: `import jax`, `from mpm_jax.blocks.weights import compute_weights_and_indices`, `from mpm_jax.blocks.p2g import p2g`, `from mpm_jax.blocks.g2p import g2p`, `from mpm_jax.blocks.grid import grid_update`, `from mpm_jax.types import MPMState`. In `solver.py`, replace the moved function with `from mpm_jax.stepping.substep import step` (keeps the `mpm_jax.solver.step` name that tests import). The remaining `solver.py` functions (`build_jit_step`, `simulate_frame`, `build_jit_stages`) now use that imported `step`.

- [ ] **Step 1b: Define the standard frame signature and the JAX builder**

In `stepping/jax_frames.py`:

```python
import jax
from mpm_jax.stepping.substep import step  # neutral home — no import back into solver


def build_jax_frame(params, elasticity_fn, plasticity_fn,
                    pre_fn, post_fn, steps_per_frame,
                    *, loop_kind="fori", **_ignored):
    """Default JAX frame: a chunk of `steps_per_frame` substeps as one XLA program."""
    def substep(state):
        with jax.named_scope("elasticity"):
            stress = elasticity_fn(state.F)
        with jax.named_scope("substep"):
            state = step(params, state, stress, pre_fn, post_fn, 0.0)
        with jax.named_scope("plasticity"):
            return state._replace(F=plasticity_fn(state.F))

    @jax.jit
    def jit_frame(state):
        if loop_kind == "fori":
            return jax.lax.fori_loop(0, steps_per_frame, lambda _, s: substep(s), state)
        for _ in range(steps_per_frame):  # 'python' = unrolled
            state = substep(state)
        return state

    return jit_frame
```

- [ ] **Step 2: Move the `jax_v1_5` (scan-over-offsets) builder**

Move `build_jit_frame_scan` from `p2g_scan.py` into `stepping/jax_frames.py`, renamed `build_jax_v1_5_frame`, with the same standardized signature (`*, loop_kind="fori", **_ignored`). Keep its scan-over-27-offsets P2G body. Leave `build_jit_stages_scan` in `p2g_scan.py` (tests still import it) and have `p2g_scan.build_jit_frame_scan` re-export from the new location.

- [ ] **Step 3: Re-export old name from `solver.py`**

In `solver.py`, replace the body of `build_jit_frame` with a thin shim, or add:

```python
from mpm_jax.stepping.jax_frames import build_jax_frame as build_jit_frame  # compat alias
```

Keep `build_jit_step`, `build_jit_stages`, `simulate_frame`, `step` defined in `solver.py` (tests import them).

- [ ] **Step 4: Run the suite**

Run: `pixi run test` then (GPU) `pixi run -e gpu pytest tests/test_jax_v1_5.py -q`
Expected: PASS. `test_warp_bonus_matches_jax.py` imports `build_jit_frame` from `mpm_jax.solver` — alias must work.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: add stepping/jax_frames.py (fori default), alias old names"
```

### Task 2.2: Create `stepping/cuda_frames.py`

**Files:**
- Create: `src/mpm_jax/stepping/cuda_frames.py`
- Modify: `src/mpm_jax/cuda/p2g_cuda.py`

- [ ] **Step 1: Relocate the CUDA frame builders with the standard signature**

Move `build_jit_frame_inline`, `build_jit_frame_v2_inline`, `build_jit_frame_v3_inline`, `build_jit_frame_v4_inline` (from `cuda/p2g_cuda.py:224-...`) into `stepping/cuda_frames.py`, renamed and re-signatured:

- `build_cuda_v1_frame(params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame, **_ignored)`
- `build_cuda_v2_frame(..., *, loop_kind="fori", **_ignored)` — passes `loop_kind` through to the existing `loop_kind`-aware body.
- `build_cuda_v3_frame(..., *, loop_kind="fori", cuda_graph=False, **_ignored)` — `cuda_graph` selects the XLA-flag/graph-capture path that was `cuda_v6_inline`.
- `build_cuda_v4_frame(..., **_ignored)`

Keep the FFI registration helpers (`_register_*`, `is_available`, op wrappers, `make_fused_stages`) in `cuda/p2g_cuda.py`; the stepping builders import the registered ops from there.

- [ ] **Step 2: Re-export old names from `cuda/p2g_cuda.py`**

```python
from mpm_jax.stepping.cuda_frames import (
    build_cuda_v1_frame as build_jit_frame_inline,
    build_cuda_v2_frame as build_jit_frame_v2_inline,
    build_cuda_v3_frame as build_jit_frame_v3_inline,
    build_cuda_v4_frame as build_jit_frame_v4_inline,
)
```

(`test_warp_v1_inline_matches_cuda.py` imports `build_jit_frame_inline` from `mpm_jax.cuda.p2g_cuda` — alias keeps it working.)

- [ ] **Step 3: Run GPU equivalence tests**

Run: `pixi run -e gpu pytest tests/test_cuda_v2_inline_matches_v1.py tests/test_warp_v1_inline_matches_cuda.py -q`
Expected: PASS on GPU (skip on CPU).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: add stepping/cuda_frames.py with loop_kind/cuda_graph opts"
```

### Task 2.3: Create `stepping/warp_frames.py`; relocate Warp kernel + graph modules

**Files:**
- Create: `src/mpm_jax/stepping/warp_frames.py`
- Move: `src/mpm_jax/warp_p2g.py` → `src/mpm_jax/warp_kernels.py`
- Move: `src/mpm_jax/warp_bonus.py` → `src/mpm_jax/warp_graph.py`
- Modify: importers

- [ ] **Step 1: Move the Warp modules**

```bash
git mv src/mpm_jax/warp_p2g.py src/mpm_jax/warp_kernels.py
git mv src/mpm_jax/warp_bonus.py src/mpm_jax/warp_graph.py
```

- [ ] **Step 2: Relocate the Warp frame builders**

Move `build_jit_frame_warp_inline`, `build_jit_frame_warp_tile`, `build_jit_frame_warp_supercell_tile` (now in `warp_kernels.py`) into `stepping/warp_frames.py`, renamed `build_warp_v1_frame`, `build_warp_v2_tile_frame`, `build_warp_v3_frame`, each with `(..., steps_per_frame, **_ignored)`. They import the `@wp.kernel` defs and `TILE_SIZE` from `warp_kernels.py`.

- [ ] **Step 3: Re-export old names + update importers**

In `warp_kernels.py`, re-export:
```python
from mpm_jax.stepping.warp_frames import (
    build_warp_v1_frame as build_jit_frame_warp_inline,
    build_warp_v2_tile_frame as build_jit_frame_warp_tile,
    build_warp_v3_frame as build_jit_frame_warp_supercell_tile,
)
```
Update test imports that point at `mpm_jax.warp_p2g` / `mpm_jax.warp_bonus`:
- `tests/test_warp_v2_tile_matches_v1.py`, `tests/test_warp_v3_supercell_tile_matches_v1.py`, `tests/test_warp_v1_inline_matches_cuda.py`: `from mpm_jax.warp_p2g import ...` → `from mpm_jax.warp_kernels import ...`
- `tests/test_warp_bonus_matches_jax.py`: `from mpm_jax.warp_bonus import WarpBonusSimulator` → `from mpm_jax.warp_graph import WarpBonusSimulator`

- [ ] **Step 4: Update `simulate.py` import sites for the moved modules**

Run: `grep -n "warp_p2g\|warp_bonus" simulate.py profile_nsight.py`
Rewrite each to `warp_kernels` / `warp_graph` respectively. (These are temporary — `simulate.py` is rewritten in Phase 5, but it must stay runnable now.)

- [ ] **Step 5: Run the suite**

Run: `pixi run -e gpu pytest tests/test_warp_v2_tile_matches_v1.py tests/test_warp_v3_supercell_tile_matches_v1.py tests/test_warp_bonus_matches_jax.py -q`
Expected: PASS on GPU (skip on CPU).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor: stepping/warp_frames.py; rename warp modules"
```

---

## Phase 3 — Registry

### Task 3.1: Create `registry.py` with `KernelSpec`, `KERNELS`, `build_solver`

**Files:**
- Create: `src/mpm_jax/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
from mpm_jax.registry import KERNELS, REMOVED_KERNELS, KernelSpec


def test_every_kernel_has_a_spec():
    expected = {
        "jax", "jax_v1_5",
        "cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline",
        "warp_v1_inline", "warp_v2_tile", "warp_v3_supercell_tile",
        "warp_bonus_graph", "warp_bonus_v2_graph",
    }
    assert set(KERNELS) == expected
    for spec in KERNELS.values():
        assert isinstance(spec, KernelSpec)
        assert spec.solver_cls is not None
        assert callable(spec.build_frame)


def test_removed_kernels_listed():
    for name in ("cuda_v1", "cuda_v2", "cuda_v4", "cuda_fused",
                 "cuda_v2_fori_inline", "cuda_v3_fori_inline", "cuda_v6_inline"):
        assert name in REMOVED_KERNELS
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run pytest tests/test_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: mpm_jax.registry`.

- [ ] **Step 3: Implement `registry.py`**

```python
from dataclasses import dataclass, field
from typing import Callable

from mpm_jax.solver import MPMSolver, WarpGraphSolver
from mpm_jax.stepping.jax_frames import build_jax_frame, build_jax_v1_5_frame
from mpm_jax.stepping.cuda_frames import (
    build_cuda_v1_frame, build_cuda_v2_frame, build_cuda_v3_frame, build_cuda_v4_frame,
)
from mpm_jax.stepping.warp_frames import (
    build_warp_v1_frame, build_warp_v2_tile_frame, build_warp_v3_frame,
)
from mpm_jax.stepping.warp_graph_frame import build_warp_graph


@dataclass(frozen=True)
class KernelSpec:
    solver_cls: type
    build_frame: Callable
    defaults: dict = field(default_factory=dict)


KERNELS = {
    "jax":                    KernelSpec(MPMSolver, build_jax_frame),
    "jax_v1_5":               KernelSpec(MPMSolver, build_jax_v1_5_frame),
    "cuda_v1_inline":         KernelSpec(MPMSolver, build_cuda_v1_frame),
    "cuda_v2_inline":         KernelSpec(MPMSolver, build_cuda_v2_frame,
                                         {"loop_kind": "fori"}),
    "cuda_v3_inline":         KernelSpec(MPMSolver, build_cuda_v3_frame,
                                         {"loop_kind": "fori", "cuda_graph": False}),
    "cuda_v4_inline":         KernelSpec(MPMSolver, build_cuda_v4_frame),
    "warp_v1_inline":         KernelSpec(MPMSolver, build_warp_v1_frame),
    "warp_v2_tile":           KernelSpec(MPMSolver, build_warp_v2_tile_frame),
    "warp_v3_supercell_tile": KernelSpec(MPMSolver, build_warp_v3_frame),
    "warp_bonus_graph":       KernelSpec(WarpGraphSolver, build_warp_graph),
    "warp_bonus_v2_graph":    KernelSpec(WarpGraphSolver, build_warp_graph,
                                         {"indexed_sort": True}),
}

REMOVED_KERNELS = {
    "cuda_v1": "Use cuda_v1_inline (scatter-only variant removed).",
    "cuda_v2": "Use cuda_v2_inline.",
    "cuda_v4": "Use cuda_v4_inline.",
    "cuda_fused": "Deprecated; use an inline kernel and profile=jax.",
    "cuda_v2_fori_inline": "Use kernel=cuda_v2_inline with loop_kind=fori.",
    "cuda_v3_fori_inline": "Use kernel=cuda_v3_inline with loop_kind=fori.",
    "cuda_v6_inline": "Use kernel=cuda_v3_inline with cuda_graph=true.",
}
```

> NOTE: `build_solver(cfg)` and `build_warp_graph` are added in Tasks 4.3 and 4.2 respectively. This task's test only checks the spec table, so it passes once the classes/builders it imports exist. If Task 4 is not yet done, temporarily import the solver classes from their stub (Task 4.1 lands first — see ordering note below). **Order: do Task 4.1 (classes) before 3.1, or stub the imports.** Recommended: implement 4.1 → 4.2 → 4.3, then 3.1.

- [ ] **Step 4: Run the test (after 4.1–4.2 exist)**

Run: `pixi run pytest tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: kernel registry (KERNELS, REMOVED_KERNELS, KernelSpec)"
```

---

## Phase 4 — Solver classes

> **Ordering:** Implement 4.1 and 4.2 before Task 3.1's test runs, since `registry.py` imports these names.

### Task 4.1: `MPMSolver` base class (TDD)

**Files:**
- Modify: `src/mpm_jax/solver.py` (append the class)
- Test: `tests/test_solver_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_solver_api.py`:

```python
import jax.numpy as jnp
from mpm_jax.types import MPMState, make_params
from mpm_jax.solver import MPMSolver, step as _substep_orch  # noqa: F401
from mpm_jax.stepping.jax_frames import build_jax_frame
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.blocks.grid import build_grid_x


def _make_solver(steps_per_frame=2, n=64, G=16):
    params = make_params(n_particles=n, num_grids=G, dt=3e-4)
    grid_x = build_grid_x(G)
    x = jnp.array([[0.5, 0.5, 0.5]] * n, dtype=jnp.float32)
    pre_fn, post_fn = build_boundary_fns([], grid_x, params.dx, x, params.dt, params.p_mass)
    elasticity = get_constitutive({"name": "CorotatedElasticity"})
    plasticity = get_constitutive({"name": "IdentityPlasticity"})
    init = MPMState(x=x, v=jnp.zeros((n, 3)), C=jnp.zeros((n, 3, 3)),
                    F=jnp.broadcast_to(jnp.eye(3), (n, 3, 3)).copy())
    return MPMSolver(params, elasticity_fn=elasticity, plasticity_fn=plasticity,
                     pre_fn=pre_fn, post_fn=post_fn, build_frame=build_jax_frame,
                     steps_per_frame=steps_per_frame, init_state=init)


def test_step_returns_and_mutates_state():
    s = _make_solver()
    x0 = s.state.x
    out = s.step()
    assert out is s.state
    assert s.state.x.shape == x0.shape


def test_solve_equals_n_steps_and_fires_hook():
    s = _make_solver()
    calls = []
    s.solve(3, on_frame=lambda i, st: calls.append(i))
    assert calls == [0, 1, 2]


def test_reset_restores_state():
    s = _make_solver()
    init = s.state
    s.step()
    s.reset(init)
    assert s.state is init
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run pytest tests/test_solver_api.py -q`
Expected: FAIL with `ImportError: cannot import name 'MPMSolver'`.

- [ ] **Step 3: Implement `MPMSolver` in `solver.py`**

Append to `src/mpm_jax/solver.py`:

```python
class MPMSolver:
    """Stateful shell over the functional JAX core.

    Builds one jit'd pure frame function at construction; step()/solve()
    only call it. `self` is never traced.
    """

    def __init__(self, params, *, elasticity_fn, plasticity_fn,
                 pre_fn, post_fn, build_frame, steps_per_frame, init_state,
                 **frame_opts):
        self.params = params
        self.steps_per_frame = steps_per_frame
        self.state = init_state
        self._frame = build_frame(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
            steps_per_frame, **frame_opts,
        )

    def step(self):
        self.state = self._frame(self.state)
        return self.state

    def solve(self, num_frames, on_frame=None):
        for f in range(num_frames):
            self.step()
            if on_frame is not None:
                on_frame(f, self.state)
        return self.state

    def reset(self, init_state):
        self.state = init_state
        return self.state
```

- [ ] **Step 4: Run the test**

Run: `pixi run pytest tests/test_solver_api.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: MPMSolver base class (stateful shell, functional core)"
```

### Task 4.2: `WarpGraphSolver` subclass + `build_warp_graph`

**Files:**
- Create: `src/mpm_jax/stepping/warp_graph_frame.py`
- Modify: `src/mpm_jax/solver.py`
- Test: `tests/test_solver_api.py` (add a GPU-guarded case)

- [ ] **Step 1: Write the failing test (GPU-guarded)**

Append to `tests/test_solver_api.py`:

```python
import pytest


def _warp_available():
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


@pytest.mark.skipif(not _warp_available(), reason="needs Warp + CUDA")
def test_warp_graph_solver_interface():
    from omegaconf import OmegaConf
    from mpm_jax.solver import WarpGraphSolver
    from mpm_jax.stepping.warp_graph_frame import build_warp_graph
    import numpy as np
    cfg = OmegaConf.create({
        "sim": {"n_particles": 4096, "num_grids": 32, "dt": 3e-4,
                "steps_per_frame": 2, "clip_bound": 0.5, "damping": 1.0,
                "gravity": [0, 0, -9.8], "rho": 1000.0, "size": [0.5, 0.5, 0.5],
                "initial_velocity": [0, 0, 0], "center": [0.5, 0.5, 0.5]},
        "material": {"elasticity": {"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4},
                     "plasticity": {"name": "IdentityPlasticity"}},
    })
    particles = np.random.RandomState(0).uniform(0.3, 0.7, size=(4096, 3)).astype(np.float32)
    solver = build_warp_graph(cfg, particles=particles)
    assert isinstance(solver, WarpGraphSolver)
    solver.step()
    solver.solve(2)
```

- [ ] **Step 2: Run it (skips on CPU)**

Run: `pixi run -e gpu pytest tests/test_solver_api.py::test_warp_graph_solver_interface -q`
Expected: FAIL with import error (or SKIP on CPU).

- [ ] **Step 3: Implement `WarpGraphSolver` and `build_warp_graph`**

Append to `solver.py`:

```python
class WarpGraphSolver(MPMSolver):
    """Pure-Warp graph backend. Wraps the capture/replay engine; no JAX frame."""

    def __init__(self, engine):
        # Do NOT call super().__init__: there is no jit'd _frame here.
        self._engine = engine            # WarpBonusSimulator (captured graphs)
        self.steps_per_frame = engine.steps_per_frame

    def step(self):
        self._engine.run_frames(1)
        return self.state

    def solve(self, num_frames, on_frame=None):
        if on_frame is None:
            self._engine.run_frames(num_frames)
            return self.state
        for f in range(num_frames):
            self._engine.run_frames(1)
            on_frame(f, self.state)
        return self.state

    @property
    def state(self):
        return self._engine            # positions live in engine.x (wp.array)

    @state.setter
    def state(self, _value):
        pass                            # state is owned by the Warp engine
```

Create `src/mpm_jax/stepping/warp_graph_frame.py`:

```python
from mpm_jax.warp_graph import WarpBonusSimulator
from mpm_jax.solver import WarpGraphSolver


def build_warp_graph(cfg, *, particles, indexed_sort=False, **_ignored):
    """Construct a pure-Warp capture/replay solver.

    `cfg` is the resolved Hydra config; `particles` is the (N,3) numpy init.
    """
    engine = WarpBonusSimulator(particles, cfg, indexed_sort=indexed_sort)
    engine.capture()
    return WarpGraphSolver(engine)
```

> The `WarpGraphSolver.state` property exposing the engine is the pragmatic seam noted as Risk #2 in the spec. If callers need `MPMState`-shaped reads, add an `engine.as_mpm_state()` adapter; not required for parity tests, which read `engine.x`.

- [ ] **Step 4: Run the test (GPU)**

Run: `pixi run -e gpu pytest tests/test_solver_api.py::test_warp_graph_solver_interface -q`
Expected: PASS on GPU; SKIP on CPU.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: WarpGraphSolver + build_warp_graph (capture/replay)"
```

### Task 4.3: `build_solver(cfg)` factory in `registry.py`

**Files:**
- Modify: `src/mpm_jax/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_build_solver_dispatches_jax_kernel():
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import MPMSolver
    cfg = OmegaConf.create({
        "kernel": {"name": "jax"},
        "sim": {"n_particles": 64, "num_grids": 16, "dt": 3e-4, "steps_per_frame": 2,
                "clip_bound": 0.5, "damping": 1.0, "gravity": [0, 0, -9.8], "rho": 1000.0,
                "size": [0.5, 0.5, 0.5], "initial_velocity": [0, 0, 0],
                "center": [0.5, 0.5, 0.5], "boundary_conditions": []},
        "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                     "plasticity": {"name": "IdentityPlasticity"}},
    })
    solver = build_solver(cfg)
    assert isinstance(solver, MPMSolver)
    solver.step()


def test_build_solver_rejects_removed_kernel():
    import pytest
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    cfg = OmegaConf.create({"kernel": {"name": "cuda_v6_inline"}, "sim": {}, "material": {}})
    with pytest.raises(ValueError, match="cuda_v3_inline"):
        build_solver(cfg)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run pytest tests/test_registry.py::test_build_solver_dispatches_jax_kernel -q`
Expected: FAIL (`build_solver` undefined).

- [ ] **Step 3: Implement `build_solver`**

Add to `registry.py` (imports for the helpers at top):

```python
import numpy as np
import jax.numpy as jnp
from mpm_jax.types import MPMState, make_params
from mpm_jax.blocks.grid import build_grid_x
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.stepping.warp_graph_frame import build_warp_graph


def _initial_particles(cfg):
    # Mirrors simulate.get_particles: a cube of n points centered at sim.center.
    from simulate import get_particles
    return get_particles(int(cfg.sim.n_particles),
                         center=list(cfg.sim.center), size=[0.5, 0.5, 0.5])


def build_solver(cfg):
    name = cfg.kernel.name
    if name in REMOVED_KERNELS:
        raise ValueError(f"kernel={name} removed. {REMOVED_KERNELS[name]}")
    spec = KERNELS[name]
    particles_np = _initial_particles(cfg)

    if spec.solver_cls is WarpGraphSolver:
        opts = {**spec.defaults}
        return build_warp_graph(cfg, particles=particles_np, **opts)

    sim, mat = cfg.sim, cfg.material
    params = make_params(
        n_particles=int(sim.n_particles), num_grids=int(sim.num_grids), dt=float(sim.dt),
        gravity=list(sim.gravity), rho=float(sim.rho), clip_bound=float(sim.clip_bound),
        damping=float(sim.damping), center=list(sim.center), size=list(sim.size),
    )
    particles = jnp.array(particles_np, dtype=jnp.float32)
    grid_x = build_grid_x(params.num_grids)
    pre_fn, post_fn = build_boundary_fns(
        list(sim.boundary_conditions), grid_x, params.dx, particles, params.dt, params.p_mass)
    elasticity_fn = get_constitutive(mat.elasticity)
    plasticity_fn = get_constitutive(mat.plasticity)
    init = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (int(sim.n_particles), 3)).copy(),
        C=jnp.zeros((int(sim.n_particles), 3, 3)),
        F=jnp.tile(jnp.eye(3), (int(sim.n_particles), 1, 1)),
    )
    # merge spec defaults with optional kernel-cfg overrides (loop_kind, cuda_graph)
    frame_opts = {**spec.defaults}
    for k in ("loop_kind", "cuda_graph"):
        if k in cfg.kernel:
            frame_opts[k] = cfg.kernel[k]
    return spec.solver_cls(
        params, elasticity_fn=elasticity_fn, plasticity_fn=plasticity_fn,
        pre_fn=pre_fn, post_fn=post_fn, build_frame=spec.build_frame,
        steps_per_frame=int(sim.steps_per_frame), init_state=init, **frame_opts,
    )
```

> `get_particles` currently lives in `simulate.py`. To avoid a circular import (`registry` ← `simulate`), Task 5.1 moves `get_particles` into `src/mpm_jax/blocks/init.py` and `registry._initial_particles` imports it from there. Update the import accordingly when 5.1 lands.

- [ ] **Step 4: Run the tests**

Run: `pixi run pytest tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: build_solver(cfg) registry factory"
```

---

## Phase 5 — Rewire `simulate.py`

### Task 5.1: Move `get_particles` into a block and repoint registry

**Files:**
- Create: `src/mpm_jax/blocks/init.py`
- Modify: `simulate.py`, `src/mpm_jax/registry.py`

- [ ] **Step 1: Move `get_particles`**

Run: `grep -n "def get_particles" simulate.py` and move that function verbatim into `src/mpm_jax/blocks/init.py`. In `simulate.py`, add `from mpm_jax.blocks.init import get_particles`.

- [ ] **Step 2: Repoint registry**

In `registry.py`, change `_initial_particles` to `from mpm_jax.blocks.init import get_particles` (top-level import; removes the deferred `simulate` import and the circular-import risk).

- [ ] **Step 3: Run tests**

Run: `pixi run pytest tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: move get_particles into blocks/init.py"
```

### Task 5.2: Replace `run_jax` / `run_warp_bonus` dispatch with `build_solver`

**Files:**
- Modify: `simulate.py`

- [ ] **Step 1: Rewrite the run path**

Replace the bodies of `run_jax` (the `if/elif kernel_name` chain, the print blurbs, `_maybe_enable_cuda_graphs`, the `build_jit_frame` selection block, and the warmup-exclusion set) and `run_warp_bonus` with a single path built on `build_solver`. Concretely, `run_jax` becomes:

```python
def run_jax(cfg):
    from mpm_jax.registry import build_solver, KERNELS, REMOVED_KERNELS
    import jax, jax.numpy as jnp
    kernel_name = cfg.kernel.name
    # CUDA-graph XLA flags must be set before importing jax — handled in main() now.
    solver = build_solver(cfg)
    bench = cfg.get("benchmark", False)

    with jax.profiler.TraceAnnotation("warmup", kernel=kernel_name):
        solver.step()
        jax.block_until_ready(solver.state.x)
        solver.reset(solver_initial_state(cfg))  # re-init for the timed region

    frames, frame_metrics = [], []
    if bench:
        with jax.profiler.TraceAnnotation("benchmark", kernel=kernel_name):
            t0 = time.perf_counter()
            for f in tqdm(range(cfg.sim.num_frames), desc="MPM"):
                with jax.profiler.StepTraceAnnotation("frame", step_num=f):
                    solver.step()
            jax.block_until_ready(solver.state.x)
            elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        def on_frame(f, st):
            frames.append(np.array(st.x))
            frame_metrics.append({"x_mean_z": float(st.x[:, 2].mean()),
                                  "v_max": float(jnp.abs(st.v).max())})
        solver.solve(cfg.sim.num_frames, on_frame=on_frame)
        jax.block_until_ready(solver.state.x)
        elapsed = time.perf_counter() - t0

    total_steps = cfg.sim.num_frames * cfg.sim.steps_per_frame
    summary = {"timestep": {"mean_ms": elapsed / cfg.sim.num_frames * 1000,
                            "std_ms": 0.0, "total_ms": elapsed * 1000,
                            "count": cfg.sim.num_frames}}
    return frames, elapsed, total_steps, summary, frame_metrics
```

Add a small `solver_initial_state(cfg)` helper (or expose `build_solver` returning the init state) so warmup can reset cleanly; simplest is to have `MPMSolver` capture its own initial state and add `reset_to_initial()`. Implement whichever is cleaner — if adding `reset_to_initial()`, store `self._init_state = init_state` in `MPMSolver.__init__` and add:

```python
def reset_to_initial(self):
    self.state = self._init_state
    return self.state
```

and call `solver.reset_to_initial()` instead of `solver.reset(solver_initial_state(cfg))`.

For the `WarpGraphSolver` branch, the timed loop is the same `solver.solve(...)` shape, but three things differ and must be branched on `isinstance(solver, WarpGraphSolver)`:
1. **Sync:** use `import warp as wp; wp.synchronize()` instead of `jax.block_until_ready(...)` (the engine state is Warp arrays, not JAX).
2. **State reads:** `solver.state` is the `WarpBonusSimulator` engine; read positions via `solver.state.x.numpy()` (a `wp.array`) for frames/metrics, not `st.x[:, 2].mean()`.
3. **No `make_state`/`reset_to_initial`:** the engine owns and re-initialises its own buffers on `capture()`; for warmup, run one `solver.step()` then re-`capture()` if a clean timed region is needed, or simply time from the first replayed frame (matches the pre-refactor `WarpBonusSimulator` benchmark).

Keep the existing Warp `HashGrid` bookkeeping block from the old `run_jax` non-benchmark path (the DLPack `wp.from_dlpack(state.x)` handoff) only for the JAX-solver GIF path; the Warp-graph GIF path reads `engine.x.numpy()` directly. Drop the separate `run_warp_bonus` function and route both solver types through one `run(cfg)`.

- [ ] **Step 2: Move CUDA-graph XLA-flag toggling into `main()`**

The old `_maybe_enable_cuda_graphs(kernel_name)` set `XLA_FLAGS` before `import jax`. Preserve that ordering: in `main()`, before any `import jax`, check `cfg.kernel.name == "cuda_v3_inline"` and `cfg.kernel.get("cuda_graph", False)` and set the flag. Keep the existing `_MPM_INSIDE_PROFILER` / nsys-relaunch logic untouched.

- [ ] **Step 3: Run a representative matrix (GPU)**

Run:
```bash
pixi run -e gpu python simulate.py kernel=jax sim.n_particles=4096 sim.num_frames=3 benchmark=true
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi sim.n_particles=4096 sim.num_frames=3 benchmark=true
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi cuda_graph=true sim.n_particles=4096 sim.num_frames=3 benchmark=true
pixi run -e gpu python simulate.py kernel=warp_v1_inline material=jelly_jacobi sim.n_particles=4096 sim.num_frames=3 benchmark=true
pixi run -e gpu python simulate.py kernel=warp_bonus_graph material=jelly_jacobi sim.num_grids=32 sim.n_particles=4096 sim.num_frames=3 benchmark=true
pixi run -e gpu python simulate.py kernel=jax sim.n_particles=4096 sim.num_frames=3   # GIF path (on_frame)
```
Expected: all print timing; the last writes `output/jelly_jax.gif`. ms/step within noise of pre-refactor numbers for `jax` and `cuda_v3_inline`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: simulate.py uses build_solver; remove if/elif dispatch"
```

---

## Phase 6 — Collapse flag-only kernel configs

### Task 6.1: Update kernel configs; delete removed-variant configs

**Files:**
- Modify: `conf/kernel/cuda_v2_inline.yaml`, `conf/kernel/cuda_v3_inline.yaml`
- Delete: `conf/kernel/cuda_v2_fori_inline.yaml`, `conf/kernel/cuda_v3_fori_inline.yaml`, `conf/kernel/cuda_v6_inline.yaml`
- Modify: any `conf/sweep_*.yaml` referencing the deleted names

- [ ] **Step 1: Add the new params to the base configs**

Append to `conf/kernel/cuda_v2_inline.yaml`:
```yaml
loop_kind: fori   # python | fori
```
Append to `conf/kernel/cuda_v3_inline.yaml`:
```yaml
loop_kind: fori     # python | fori
cuda_graph: false   # true == former cuda_v6_inline
```

- [ ] **Step 2: Delete the collapsed variant configs**

```bash
git rm conf/kernel/cuda_v2_fori_inline.yaml conf/kernel/cuda_v3_fori_inline.yaml conf/kernel/cuda_v6_inline.yaml
```

- [ ] **Step 3: Repoint sweeps**

Run: `grep -rn "cuda_v2_fori_inline\|cuda_v3_fori_inline\|cuda_v6_inline" conf/`
For each hit, replace with the base name + override, e.g. `kernel=cuda_v3_inline` and add `kernel.loop_kind=fori` / `kernel.cuda_graph=true` to the sweep axis.

- [ ] **Step 4: Verify configs load**

Run:
```bash
pixi run -e gpu python simulate.py kernel=cuda_v2_inline kernel.loop_kind=python material=jelly_jacobi sim.n_particles=4096 sim.num_frames=2 benchmark=true
pixi run -e gpu python simulate.py -cn sweep_scaling --cfg job >/dev/null && echo "sweep config OK"
```
Expected: runs/configs resolve with no Hydra errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: collapse fori/v6 kernels into loop_kind/cuda_graph params"
```

---

## Phase 7 — Delete shims, migrate tests, update docs

### Task 7.1: Repoint equivalence tests to new module paths

**Files:**
- Modify: `tests/test_cuda_equivalence.py`, `tests/test_cuda_v2_inline_matches_v1.py`, `tests/test_jax_v1_5.py`, `tests/test_warp_v1_inline_matches_cuda.py`, `tests/test_warp_v2_tile_matches_v1.py`, `tests/test_warp_v3_supercell_tile_matches_v1.py`, `tests/test_warp_bonus_matches_jax.py`, `tests/test_solver.py`

- [ ] **Step 1: Rewrite imports to canonical (non-shim) paths**

Apply these import rewrites (the targets exist after Phases 1–4):
- `from mpm_jax.solver import build_jit_frame` → `from mpm_jax.stepping.jax_frames import build_jax_frame as build_jit_frame`
- `from mpm_jax.p2g_scan import build_jit_stages_scan` → keep (still in `p2g_scan.py`)
- `from mpm_jax.cuda.p2g_cuda import build_jit_frame_inline` → `from mpm_jax.stepping.cuda_frames import build_cuda_v1_frame as build_jit_frame_inline`
- `from mpm_jax.warp_kernels import build_jit_frame_warp_inline` → `from mpm_jax.stepping.warp_frames import build_warp_v1_frame as build_jit_frame_warp_inline` (and the tile/supercell equivalents)
- `mpm_jax.solver import grid_update, step, simulate_frame` → `grid_update` from `mpm_jax.blocks.grid`; keep `step`, `simulate_frame`, `build_jit_stages` from `mpm_jax.solver`.

- [ ] **Step 2: Run the full suite (GPU)**

Run: `pixi run -e gpu pytest tests/ -q`
Expected: PASS / clean skips.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: repoint imports to blocks/stepping module paths"
```

### Task 7.2: Remove backward-compat shims

**Files:**
- Modify: `src/mpm_jax/solver.py`, `src/mpm_jax/cuda/p2g_cuda.py`, `src/mpm_jax/warp_kernels.py`, `src/mpm_jax/p2g_scan.py`

- [ ] **Step 1: Delete the compat aliases**

Remove the re-export lines added in Tasks 1.1–2.3 (`build_jit_frame as ...` aliases, the `grid_update`/`p2g` re-exports from `solver.py`, etc.). Keep only what is still imported by canonical paths: `solver.py` retains `step`, `build_jit_step`, `build_jit_stages`, `simulate_frame`, `MPMSolver`, `WarpGraphSolver`, and the `from mpm_jax.types import ...` / `from mpm_jax.blocks... import ...` that those functions *use internally*.

- [ ] **Step 2: Grep for dangling references**

Run: `grep -rn "build_jit_frame_inline\|build_jit_frame_warp\|build_jit_frame_v[234]\|build_jit_frame_scan" src/ tests/ simulate.py profile_nsight.py`
Expected: no hits in `src/` or `tests/` (only canonical `build_*_frame` names remain). Fix any stragglers.

- [ ] **Step 3: Run the full suite**

Run: `pixi run -e gpu pytest tests/ -q`
Expected: PASS / clean skips.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: remove backward-compat shims"
```

### Task 7.3: Update `profile_nsight.py` and docs

**Files:**
- Modify: `profile_nsight.py`
- Modify: `.claude/CLAUDE.md`, `README.md`

- [ ] **Step 1: Repoint `profile_nsight.py`**

Run: `grep -n "warp_p2g\|warp_bonus\|build_jit_frame\|run_jax\|p2g_scan\|jacobi_svd\|morton" profile_nsight.py`
Rewrite imports to the new module paths; if it duplicated `run_jax`'s dispatch, route it through `build_solver(cfg)` too.

- [ ] **Step 2: Run an Nsight smoke test (GPU)**

Run: `pixi run -e gpu python profile_nsight.py -cn nsight_profile kernel=jax material=jelly_jacobi nsight.phase=p2g sim.n_particles=4096`
Expected: completes without import errors.

- [ ] **Step 3: Update CLAUDE.md + README**

Update both to reflect: `src/mpm_jax/` layout; `blocks/`, `stepping/`, `registry.py`, `solver.py` (MPMSolver/WarpGraphSolver); the registry replacing the if/elif; `loop_kind`/`cuda_graph` params; removed `kernel=` names (`cuda_v2_fori_inline`, `cuda_v3_fori_inline`, `cuda_v6_inline`). Update the "Adding a new inline CUDA kernel" checklist to: add `.cu`, add to `CMakeLists.txt` KERNELS, add an FFI wrapper in `cuda/p2g_cuda.py`, add a `build_*_frame` in `stepping/cuda_frames.py`, register it in `registry.py`, add `conf/kernel/<name>.yaml`.

- [ ] **Step 4: Final full suite + commit**

Run: `pixi run -e gpu pytest tests/ -q`
Expected: PASS / clean skips.

```bash
git add -A && git commit -m "docs: update CLAUDE.md + README for src/ + solver/registry layout"
```

---

## Final verification checklist (run after Phase 7)

- [ ] `grep -rn "if kernel_name ==" simulate.py` → no hits (dispatch gone).
- [ ] `grep -rn "build_jit_frame_" src/ tests/` → no hits (zoo gone).
- [ ] `pixi install -e gpu` rebuilds `.so` into `src/mpm_jax/cuda/_lib/` and imports cleanly.
- [ ] `pixi run -e gpu pytest tests/ -q` green (note skips).
- [ ] Benchmark ms/step for `jax` and `cuda_v3_inline` within noise of pre-refactor numbers (record both).
- [ ] GIF path (`simulate.py kernel=jax` without `benchmark=true`) produces `output/jelly_jax.gif`.
