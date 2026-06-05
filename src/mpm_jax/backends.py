"""Backend classes for the shared JAX-owned MPM frame loop.

Only the P2G (and the particle ordering it needs) varies across kernels, so the
backends form a small template-method class hierarchy:

    Backend            jax_baseline: identity order + JAX scan P2G + shared G2P
    ├─ CudaInline      one CUDA inline-scatter launch (kind picks the .cu kernel)
    │   ├─ CudaV1/V2   no reordering
    │   └─ CudaV3      + Morton sort (overrides prepare)
    ├─ CudaV4          super-cell sort + super-cell-tiled CUDA scatter
    └─ CutileV6        super-cell sort + cuTile arena scatter

A variant overrides ``prepare()`` (the "sort") and ``p2g()`` (the scatter after
sorting); ``g2p()`` lives on the base and is shared by all. The frame loop calls
``backend.step()`` — which orders the particles then scatters — and
``backend.g2p()``; it never sees the sort. Backends are plain (identity-hashable)
classes so they slot into the solver's static field unchanged; they hold no array
state. Availability is not checked — the ``gpu`` pixi env guarantees the CUDA
``.so`` kernels and the cuTile runtime are present.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp

from mpm_jax.blocks.sort import home_super_cell_id, morton_argsort
from mpm_jax.cuda import p2g_cuda
from mpm_jax.cuda.p2g_cuda import (
    SUPPORTED_SC,
    V4_SUPER_CELL_WIDTH,
    cuda_p2g_v4_inline,
    is_available,
)
from mpm_jax.cutile_p2g import ARENA_SC, cutile_p2g_atomic_tile
from mpm_jax.g2p_scan import _g2p_scan_mls
from mpm_jax.p2g_scan import _p2g_scan

# NOTE: the pure-JAX p2g/g2p/sort modules are imported eagerly (at import time,
# outside any trace). They build module-level stencil constants (e.g.
# p2g_scan.OFFSET_27); importing them lazily inside a Backend method would
# run that module body *during* a jit trace and leak the constant as a tracer.


class PreparedSubstep(NamedTuple):
    x: Any
    v: Any
    C: Any
    F: Any
    stress: Any
    cell_start: Any = None  # set by the sorted variants (CSR start offsets)


# ============================================================================
# P2G / G2P math (module functions; the Backend methods are thin wrappers)
# ============================================================================
def _jax_scan_p2g(params, prepared):
    return _p2g_scan(
        prepared.x,
        prepared.v,
        prepared.C,
        prepared.stress,
        params.dt,
        params.vol,
        params.p_mass,
        params.dx,
        params.inv_dx,
        params.num_grids,
    )


def _jax_scan_g2p_mls(params, prepared, grid_v):
    return _g2p_scan_mls(
        grid_v,
        prepared.x,
        prepared.F,
        params.dt,
        params.inv_dx,
        params.dx,
        params.num_grids,
        params.clip_bound,
    )


def _cuda_inline_p2g(kind, params, prepared):
    # CudaV1/V2/V3 set ``kind`` to inline / v2_inline / v3_inline; the matching
    # FFI wrapper is cuda_p2g_<kind>.
    fn = getattr(p2g_cuda, f"cuda_p2g_{kind}")
    return fn(
        prepared.x,
        prepared.v,
        prepared.C,
        prepared.stress,
        params.num_grids,
        params.dt,
        params.vol,
        params.p_mass,
        params.inv_dx,
        params.dx,
    )


def _cuda_v4_p2g(params, prepared, super_cell):
    return cuda_p2g_v4_inline(
        prepared.x,
        prepared.v,
        prepared.C,
        prepared.stress,
        prepared.cell_start,
        params.num_grids,
        params.dt,
        params.vol,
        params.p_mass,
        params.inv_dx,
        params.dx,
        super_cell=super_cell,
    )


def _cutile_atomic_tile_p2g(params, prepared):
    return cutile_p2g_atomic_tile(
        prepared.x,
        prepared.v,
        prepared.C,
        prepared.stress,
        prepared.cell_start,
        params.num_grids,
        params.dt,
        params.vol,
        params.p_mass,
        params.inv_dx,
        params.dx,
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
    return jnp.arange(Gs**3 + 1, dtype=jnp.int32)


def _supercell_order(params, state, stress, super_cell):
    """Sort by home super-cell of width ``super_cell`` and build the CSR
    ``cell_start`` of length (G/super_cell)**3 + 1."""
    super_id = home_super_cell_id(state.x, params.inv_dx, params.num_grids, super_cell)
    order = jnp.argsort(super_id)
    cell_start = jnp.searchsorted(
        super_id[order], _supercell_boundaries(params, super_cell)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order],
        state.v[order],
        state.C[order],
        state.F[order],
        stress[order],
        cell_start=cell_start,
    )


def _arena_super_cell():
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

    def __init__(self, num_grids=None):
        self.validate_num_grids(num_grids)

    def validate_num_grids(self, num_grids):
        divisor = self.grid_divisor()
        if divisor is not None and num_grids is not None and num_grids % divisor != 0:
            raise RuntimeError(
                f"{self.name} requires num_grids ({num_grids}) divisible by "
                f"super-cell width ({divisor})."
            )

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
    is_available(kind)


def _load_cutile_kernels():
    return None


class CudaInline(Backend):
    """One CUDA inline-scatter launch; ``kind`` selects the .cu kernel."""

    kind = "inline"

    def __init__(self, num_grids=None):
        _register_cuda_kernel(self.kind)
        super().__init__(num_grids=num_grids)

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

    def __init__(self, num_grids=None, super_cell_width=None):
        _register_cuda_kernel("v4_inline")
        sc = V4_SUPER_CELL_WIDTH if super_cell_width is None else int(super_cell_width)
        if sc not in SUPPORTED_SC:
            raise ValueError(
                f"cuda_v4_inline super_cell_width={sc} is not a compiled "
                f"instantiation; the kernel is built for {SUPPORTED_SC}."
            )
        self.super_cell = sc
        # validate_num_grids() reads grid_divisor() -> self.super_cell, so set
        # it before the base __init__ runs the divisibility check.
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        return _supercell_order(params, state, stress, self.super_cell)

    def p2g(self, params, prepared):
        return _cuda_v4_p2g(params, prepared, self.super_cell)

    def grid_divisor(self):
        return self.super_cell


class CutileV6(Backend):
    """cuTile arena scatter; occupancy left to the cuTile compiler default."""

    name = "cutile_v6_atomic_tile"

    def __init__(self, num_grids=None):
        _load_cutile_kernels()
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        return _supercell_order(params, state, stress, _arena_super_cell())

    def p2g(self, params, prepared):
        return _cutile_atomic_tile_p2g(params, prepared)

    def grid_divisor(self):
        return _arena_super_cell()


# Every P2G backend; each class owns its ``name`` (and Hydra ``_target_``), so
# this tuple is the single source of truth — add a variant by adding it here.
_BACKENDS = (Backend, CudaV1, CudaV2, CudaV3, CudaV4, CutileV6)
_BACKEND_CLASSES = {cls.name: cls for cls in _BACKENDS}
KERNEL_NAMES = tuple(_BACKEND_CLASSES)
BACKEND_TARGETS = {
    name: f"{cls.__module__}.{cls.__qualname__}"
    for name, cls in _BACKEND_CLASSES.items()
}


# this also seems hacky
def build_backend(name, num_grids):
    """Construct and validate the backend for a P2G variant.

    The only check is the super-cell grid-divisibility rule (raised here, at
    backend init, before any compile). Kernel availability is not checked — the
    ``gpu`` pixi env guarantees the CUDA ``.so`` kernels and cuTile runtime.
    """
    try:
        backend = _BACKEND_CLASSES[name](num_grids=num_grids)
    except KeyError:
        raise KeyError(
            f"Unknown P2G kernel {name!r}. Available: {', '.join(_BACKEND_CLASSES)}."
        ) from None
    return backend
