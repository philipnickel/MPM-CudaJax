"""Backend classes for the shared JAX-owned MPM frame loop.

Only the P2G (and the particle ordering it needs) varies across kernels, so the
backends form a small template-method class hierarchy:

    Backend            jax_baseline: identity order + JAX scan P2G + shared G2P
    ├─ CudaInline      one CUDA inline-scatter launch (kind picks the .cu kernel)
    │   ├─ CudaV1/V2   no reordering
    │   └─ CudaV3      + Morton sort (overrides prepare)
    ├─ CudaV4          super-cell sort + super-cell-tiled CUDA scatter
    └─ CutileV6        super-cell sort + cuTile arena scatter (per-GPU autotuned)

A variant overrides ``prepare()`` (the "sort") and ``p2g()`` (the scatter after
sorting); ``g2p()`` lives on the base and is shared by all. The frame loop calls
``backend.step()`` — which orders the particles then scatters — and
``backend.g2p()``; it never sees the sort. Backends are plain (identity-hashable)
classes so they slot into the solver's static field unchanged; they hold no array
state. Availability is not checked — the ``gpu`` pixi env guarantees the CUDA
``.so`` kernels and the cuTile runtime are present.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mpm_jax.blocks.grid import grid_update
from mpm_jax.blocks.sort import home_super_cell_id, morton_argsort
from mpm_jax.g2p_scan import _g2p_scan_mls
from mpm_jax.p2g_scan import _p2g_scan
from mpm_jax.types import MPMState

# NOTE: the pure-JAX p2g/g2p/sort modules are imported eagerly (at import time,
# outside any trace). They build module-level stencil constants (e.g.
# g2p_scan.OFFSET_27_INT); importing them lazily inside a Backend method would
# run that module body *during* a jit trace and leak the constant as a tracer.
# CUDA/cuTile imports stay lazy (below) so the CPU env can import this module.


class PreparedSubstep(NamedTuple):
    x: Any
    v: Any
    C: Any
    F: Any
    stress: Any
    cell_start: Any = None   # set by the sorted variants (CSR start offsets)


# ============================================================================
# P2G / G2P math (module functions; the Backend methods are thin wrappers)
# ============================================================================
def _jax_scan_p2g(params, prepared):
    return _p2g_scan(
        prepared.x, prepared.v, prepared.C, prepared.stress,
        params.dt, params.vol, params.p_mass,
        params.dx, params.inv_dx, params.num_grids,
    )


def _jax_scan_g2p_mls(params, prepared, grid_v):
    return _g2p_scan_mls(
        grid_v, prepared.x, prepared.F,
        params.dt, params.inv_dx, params.dx, params.num_grids, params.clip_bound,
    )


_CUDA_INLINE_FNS = {
    "inline": "cuda_p2g_inline",
    "v2_inline": "cuda_p2g_v2_inline",
    "v3_inline": "cuda_p2g_v3_inline",
}


def _cuda_inline_p2g(kind, params, prepared):
    from mpm_jax.cuda import p2g_cuda  # pylint: disable=import-outside-toplevel

    fn = getattr(p2g_cuda, _CUDA_INLINE_FNS[kind])
    return fn(
        prepared.x, prepared.v, prepared.C, prepared.stress,
        params.num_grids, params.dt, params.vol, params.p_mass,
        params.inv_dx, params.dx,
    )


def _cuda_v4_p2g(params, prepared):
    from mpm_jax.cuda.p2g_cuda import cuda_p2g_v4_inline  # pylint: disable=import-outside-toplevel

    return cuda_p2g_v4_inline(
        prepared.x, prepared.v, prepared.C, prepared.stress, prepared.cell_start,
        params.num_grids, params.dt, params.vol, params.p_mass,
        params.inv_dx, params.dx,
    )


def _cutile_atomic_tile_p2g(params, prepared, kernel):
    from mpm_jax.cutile_p2g import cutile_p2g_atomic_tile  # pylint: disable=import-outside-toplevel

    return cutile_p2g_atomic_tile(
        prepared.x, prepared.v, prepared.C, prepared.stress,
        prepared.cell_start, params.num_grids, params.dt, params.vol, params.p_mass,
        params.inv_dx, params.dx, kernel=kernel,
    )


# ============================================================================
# Particle ordering ("sort") helpers used by prepare()
# ============================================================================
def _identity_order(state, stress):
    return PreparedSubstep(state.x, state.v, state.C, state.F, stress)


def _morton_order(params, state, stress):
    order = morton_argsort(state.x, params.inv_dx, params.num_grids)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order], stress[order]
    )


def _supercell_boundaries(params, super_cell_width):
    Gs = params.num_grids // super_cell_width
    return jnp.arange(Gs ** 3 + 1, dtype=jnp.int32)


def _supercell_order(params, state, stress, super_cell):
    """Sort by home super-cell of width ``super_cell`` and build the CSR
    ``cell_start`` of length (G/super_cell)**3 + 1."""
    super_id = home_super_cell_id(state.x, params.inv_dx, params.num_grids, super_cell)
    order = jnp.argsort(super_id)
    cell_start = jnp.searchsorted(
        super_id[order], _supercell_boundaries(params, super_cell)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order],
        stress[order], cell_start=cell_start,
    )


def _v4_super_cell():
    from mpm_jax.cuda.p2g_cuda import V4_SUPER_CELL_WIDTH  # pylint: disable=import-outside-toplevel
    return V4_SUPER_CELL_WIDTH


def _arena_super_cell():
    from mpm_jax.cutile_p2g import ARENA_SC  # pylint: disable=import-outside-toplevel
    return ARENA_SC


# ============================================================================
# Backend hierarchy
# ============================================================================
class Backend:
    """jax_baseline: identity particle order, JAX scan P2G, shared MLS-MPM G2P.

    Variants override ``prepare`` (the sort) and ``p2g`` (the scatter); ``g2p``
    is shared and never overridden. The frame loop calls ``step`` (order + p2g)
    and ``g2p``.
    """

    name = "jax_baseline"

    def prepare(self, params, state, stress):
        """Particle ordering hook ("sort"); default is identity (no reorder)."""
        return _identity_order(state, stress)

    def p2g(self, params, prepared):
        """Scatter the ordered particles to the grid ("step after sort")."""
        return _jax_scan_p2g(params, prepared)

    def step(self, params, state, stress):
        """The P2G unit the frame loop calls: order, then scatter.

        Returns ``(prepared, grid_mv, grid_m)`` — the ordered particles (the G2P
        gathers from them, keeping the sort's memory locality) and the grid.
        """
        prepared = self.prepare(params, state, stress)
        grid_mv, grid_m = self.p2g(params, prepared)
        return prepared, grid_mv, grid_m

    def g2p(self, params, prepared, grid_v):
        """Gather grid -> particles + APIC/F update. Shared by every variant."""
        return _jax_scan_g2p_mls(params, prepared, grid_v)

    def grid_divisor(self):
        """num_grids must be divisible by this (None = no constraint)."""
        return None


# Register the FFI/cuTile kernels at backend-construction time (outside any
# trace) so the handlers exist before the frame compiles. A persistent
# compile-cache *hit* skips tracing, so registering only inside the traced
# wrapper would be too late -- execution would fail with "No FFI handler
# registered". ``is_available(kind)`` is what performs the FFI registration.
def _register_cuda_kernel(kind):
    from mpm_jax.cuda.p2g_cuda import is_available  # pylint: disable=import-outside-toplevel
    is_available(kind)


def _load_cutile_kernels():
    import mpm_jax.cutile_p2g  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import


class CudaInline(Backend):
    """One CUDA inline-scatter launch; ``kind`` selects the .cu kernel."""

    kind = "inline"

    def __init__(self):
        _register_cuda_kernel(self.kind)

    def p2g(self, params, prepared):
        return _cuda_inline_p2g(self.kind, params, prepared)


class CudaV1(CudaInline):
    name = "cuda_v1_inline"
    kind = "inline"


class CudaV2(CudaInline):
    name = "cuda_v2_inline"
    kind = "v2_inline"


class CudaV3(CudaInline):
    name = "cuda_v3_inline"
    kind = "v3_inline"

    def prepare(self, params, state, stress):
        return _morton_order(params, state, stress)


class CudaV4(Backend):
    name = "cuda_v4_inline"

    def __init__(self):
        _register_cuda_kernel("v4_inline")

    def prepare(self, params, state, stress):
        return _supercell_order(params, state, stress, _v4_super_cell())

    def p2g(self, params, prepared):
        return _cuda_v4_p2g(params, prepared)

    def grid_divisor(self):
        return _v4_super_cell()


class CutileV6(Backend):
    """cuTile arena scatter; occupancy autotuned per-GPU and cached."""

    name = "cutile_v6_atomic_tile"

    def __init__(self, autotune=True):
        _load_cutile_kernels()
        kernel = None
        if autotune:
            from mpm_jax.cutile_autotune import tuned_atomic_tile_kernel  # pylint: disable=import-outside-toplevel
            kernel = tuned_atomic_tile_kernel()
        self.kernel = kernel

    def prepare(self, params, state, stress):
        return _supercell_order(params, state, stress, _arena_super_cell())

    def p2g(self, params, prepared):
        return _cutile_atomic_tile_p2g(params, prepared, self.kernel)

    def grid_divisor(self):
        return _arena_super_cell()


_BACKENDS = {
    "jax_baseline": lambda num_grids, autotune: Backend(),
    "cuda_v1_inline": lambda num_grids, autotune: CudaV1(),
    "cuda_v2_inline": lambda num_grids, autotune: CudaV2(),
    "cuda_v3_inline": lambda num_grids, autotune: CudaV3(),
    "cuda_v4_inline": lambda num_grids, autotune: CudaV4(),
    "cutile_v6_atomic_tile": lambda num_grids, autotune: CutileV6(autotune=autotune),
}

KERNEL_NAMES = tuple(_BACKENDS)


def build_backend(name, num_grids, *, autotune=True):
    """Construct and validate the backend for a P2G variant.

    The only check is the super-cell grid-divisibility rule (raised here, at
    backend init, before any compile). Kernel availability is not checked — the
    ``gpu`` pixi env guarantees the CUDA ``.so`` kernels and cuTile runtime.
    """
    try:
        backend = _BACKENDS[name](num_grids, autotune)
    except KeyError:
        raise KeyError(
            f"Unknown P2G kernel {name!r}. Available: {', '.join(_BACKENDS)}."
        ) from None
    divisor = backend.grid_divisor()
    if divisor is not None and num_grids is not None and num_grids % divisor != 0:
        raise RuntimeError(
            f"{name} requires num_grids ({num_grids}) divisible by "
            f"super-cell width ({divisor})."
        )
    return backend


def jax_baseline_backend():
    """The JAX/XLA baseline backend (identity order, scan P2G + shared MLS G2P).

    A zero-arg convenience used as the reference in tests; equivalent to
    ``build_backend("jax_baseline", num_grids)``.
    """
    return Backend()


def build_backend_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, backend, steps_per_frame):
    """Build one JIT-compiled frame from a backend object.

    The frame owns the common MPM control flow (boundary conditions, elasticity,
    grid update, plasticity, the substep loop). The backend owns only the
    P2G — ``backend.step`` orders the particles and scatters — and ``g2p``.
    The ``steps_per_frame`` substeps run as a single ``lax.fori_loop``.
    """
    @jax.jit
    def jit_frame(state):
        def step_body(state):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            state = state._replace(x=x, v=v)

            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope(f"{backend.name}_p2g"):
                prepared, grid_mv, grid_m = backend.step(params, state, stress)

            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope(f"{backend.name}_g2p"):
                new_x, new_v, new_C, new_F = backend.g2p(params, prepared, grid_v)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        return jax.lax.fori_loop(0, steps_per_frame, lambda _, s: step_body(s), state)

    return jit_frame
