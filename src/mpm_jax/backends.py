"""Backend interface for the shared JAX-owned MPM frame loop."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from mpm_jax.blocks.grid import grid_update
from mpm_jax.types import MPMState


class PreparedSubstep(NamedTuple):
    x: Any
    v: Any
    C: Any
    F: Any
    stress: Any
    weight: Any = None
    dweight: Any = None
    dpos: Any = None
    index: Any = None
    cell_start: Any = None


@dataclass(frozen=True)
class Backend:
    """Static backend operations used by the shared frame loop."""

    name: str
    p2g: Callable
    g2p: Callable
    prepare: Callable = lambda params, state, stress: PreparedSubstep(
        state.x, state.v, state.C, state.F, stress
    )
    loop_kind: str = "fori"
    defaults: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


def build_backend_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, backend, steps_per_frame,
                        *, loop_kind=None, phase_barriers=False, **_ignored):
    """Build one JIT-compiled frame from a backend object.

    The frame owns the common MPM control flow. Backends only provide particle
    ordering plus P2G/G2P implementations.
    """
    loop_kind = loop_kind or backend.loop_kind

    @jax.jit
    def jit_frame(state):
        def step_body(state):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            state = state._replace(x=x, v=v)

            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope(f"{backend.name}_prepare"):
                prepared = backend.prepare(params, state, stress)

            with jax.named_scope(f"{backend.name}_p2g"):
                grid_mv, grid_m = backend.p2g(params, prepared)

            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope(f"{backend.name}_g2p"):
                new_x, new_v, new_C, new_F = backend.g2p(params, prepared, grid_v)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        def step_body_phased(state):
            # Profiling mode: collapse the timestep into the 3 classical MPM
            # phases (P2G / Grid / G2P) and insert an optimization_barrier at
            # each phase boundary so XLA cannot fuse across them. The barrier is
            # an identity, so results are bit-for-bit identical to step_body —
            # it only forces 3 separately-labelled kernels so a profiler can
            # attribute device time per phase. Weights (`prepare`) are placed in
            # G2P, where they are consumed (the P2G scan recomputes them inline).
            # NOTE: jax-baseline tool — p2g gets a minimal Prepared (x,v,C,stress),
            # which the CUDA/Warp backends' p2g does not accept.
            with jax.named_scope("P2G"):
                x, v = pre_fn(state.x, state.v, 0.0)
                state = state._replace(x=x, v=v)
                stress = elasticity_fn(state.F)
                grid_mv, grid_m = backend.p2g(
                    params,
                    PreparedSubstep(state.x, state.v, state.C, state.F, stress),
                )
            grid_mv, grid_m, stress, sx, sv, sC, sF = jax.lax.optimization_barrier(
                (grid_mv, grid_m, stress, state.x, state.v, state.C, state.F))
            state = state._replace(x=sx, v=sv, C=sC, F=sF)

            with jax.named_scope("Grid"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)
            (grid_v,) = jax.lax.optimization_barrier((grid_v,))

            with jax.named_scope("G2P"):
                prepared = backend.prepare(params, state, stress)
                new_x, new_v, new_C, new_F = backend.g2p(params, prepared, grid_v)
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        body = step_body_phased if phase_barriers else step_body
        if loop_kind == "fori":
            return jax.lax.fori_loop(0, steps_per_frame, lambda _, s: body(s), state)
        for _ in range(steps_per_frame):
            state = body(state)
        return state

    return jit_frame


def _make_jax_scan_p2g():
    from mpm_jax.p2g_scan import _p2g_scan  # pylint: disable=import-outside-toplevel

    def p2g(params, prepared):
        return _p2g_scan(
            prepared.x, prepared.v, prepared.C, prepared.stress,
            params.dt, params.vol, params.p_mass,
            params.dx, params.inv_dx, params.num_grids,
        )

    return p2g


def _make_jax_scan_g2p_mls():
    from mpm_jax.g2p_scan import _g2p_scan_mls  # pylint: disable=import-outside-toplevel

    def g2p(params, prepared, grid_v):
        return _g2p_scan_mls(
            grid_v, prepared.x, prepared.F,
            params.dt, params.inv_dx, params.dx, params.num_grids, params.clip_bound,
        )

    return g2p


def jax_baseline_backend(**_opts):
    """The JAX/XLA baseline. lax.scan over the 27 offsets for both P2G and G2P;
    the unified MLS-MPM G2P (the APIC affine matrix C is reused as the velocity
    gradient for the F-update, so G2P builds ONE (N, 3, 3) accumulator instead
    of two); scatter-free Jacobi SVD for the stress and plasticity. Every other
    kernel reuses this exact G2P (``_make_jax_scan_g2p_mls``), so across the
    registry only the P2G implementation varies.
    """
    return Backend(
        name="jax_baseline",
        p2g=_make_jax_scan_p2g(),
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )


def _require_cuda(*kernels):
    from mpm_jax.cuda.p2g_cuda import is_available  # pylint: disable=import-outside-toplevel

    for kernel in kernels:
        if not is_available(kernel):
            raise RuntimeError(
                f"CUDA kernel {kernel!r} not registered (missing .so?). "
                "Run `pixi install -e gpu` to build."
            )


def _require_cutile():
    try:
        from mpm_jax.cutile_p2g import is_available  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "cuTile is not available. Run `pixi install -e gpu` after adding "
            "`cuda-tile[tileiras]` to the GPU environment."
        ) from exc
    if not is_available():
        raise RuntimeError("cuTile is not available in this environment.")


def _cuda_inline_p2g(kind):
    def p2g(params, prepared):
        from mpm_jax.cuda import p2g_cuda  # pylint: disable=import-outside-toplevel

        fn = {
            "inline": p2g_cuda.cuda_p2g_inline,
            "v2_inline": p2g_cuda.cuda_p2g_v2_inline,
            "v3_inline": p2g_cuda.cuda_p2g_v3_inline,
        }[kind]
        return fn(
            prepared.x, prepared.v, prepared.C, prepared.stress,
            params.num_grids, params.dt, params.vol, params.p_mass,
            params.inv_dx, params.dx,
        )

    return p2g


def cuda_v1_backend(**_opts):
    _require_cuda("inline")
    return Backend(
        name="cuda_v1_inline",
        p2g=_cuda_inline_p2g("inline"),
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )


def cuda_v2_backend(**_opts):
    _require_cuda("v2_inline")
    return Backend(
        name="cuda_v2_inline",
        p2g=_cuda_inline_p2g("v2_inline"),
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )


def _morton_prepare(params, state, stress):
    from mpm_jax.blocks.sort import morton_argsort  # pylint: disable=import-outside-toplevel

    order = morton_argsort(state.x, params.inv_dx, params.num_grids)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order], stress[order]
    )


def cuda_v3_backend(**_opts):
    _require_cuda("v3_inline")
    return Backend(
        name="cuda_v3_inline",
        prepare=_morton_prepare,
        p2g=_cuda_inline_p2g("v3_inline"),
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )


def _supercell_boundaries(params, super_cell_width):
    Gs = params.num_grids // super_cell_width
    return jnp.arange(Gs ** 3 + 1, dtype=jnp.int32)


def _cuda_v4_prepare(params, state, stress):
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        V4_SUPER_CELL_WIDTH,
    )
    from mpm_jax.blocks.sort import home_super_cell_id  # pylint: disable=import-outside-toplevel

    super_id = home_super_cell_id(
        state.x, params.inv_dx, params.num_grids, V4_SUPER_CELL_WIDTH)
    order = jnp.argsort(super_id)
    super_id_sorted = super_id[order]
    cell_start = jnp.searchsorted(
        super_id_sorted, _supercell_boundaries(params, V4_SUPER_CELL_WIDTH)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order],
        stress[order], cell_start=cell_start,
    )


def _cuda_v4_p2g(params, prepared):
    from mpm_jax.cuda.p2g_cuda import cuda_p2g_v4_inline  # pylint: disable=import-outside-toplevel

    return cuda_p2g_v4_inline(
        prepared.x, prepared.v, prepared.C, prepared.stress, prepared.cell_start,
        params.num_grids, params.dt, params.vol, params.p_mass,
        params.inv_dx, params.dx,
    )


def cuda_v4_backend(**_opts):
    from mpm_jax.cuda.p2g_cuda import V4_SUPER_CELL_WIDTH  # pylint: disable=import-outside-toplevel

    if _opts.get("num_grids") is not None and _opts["num_grids"] % V4_SUPER_CELL_WIDTH != 0:
        raise RuntimeError(
            f"cuda_v4_inline requires num_grids ({_opts['num_grids']}) divisible by "
            f"super-cell width ({V4_SUPER_CELL_WIDTH})."
        )
    _require_cuda("v4_inline")
    return Backend(
        name="cuda_v4_inline",
        prepare=_cuda_v4_prepare,
        p2g=_cuda_v4_p2g,
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )


# --- cuTile (tiled programming model) ---------------------------------------
def _cutile_arena_prepare(params, state, stress):
    """Sort particles by home super-cell at the arena width (SC=2) and build the
    CSR ``cell_start`` over (G/SC)**3 super-cells."""
    from mpm_jax.cutile_p2g import ARENA_SC  # pylint: disable=import-outside-toplevel
    from mpm_jax.blocks.sort import home_super_cell_id  # pylint: disable=import-outside-toplevel

    super_id = home_super_cell_id(state.x, params.inv_dx, params.num_grids, ARENA_SC)
    order = jnp.argsort(super_id)
    super_id_sorted = super_id[order]
    cell_start = jnp.searchsorted(
        super_id_sorted, _supercell_boundaries(params, ARENA_SC)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order],
        stress[order], cell_start=cell_start,
    )


def _make_cutile_atomic_tile_p2g(kernel):
    def p2g(params, prepared):
        from mpm_jax.cutile_p2g import cutile_p2g_atomic_tile  # pylint: disable=import-outside-toplevel

        return cutile_p2g_atomic_tile(
            prepared.x, prepared.v, prepared.C, prepared.stress,
            prepared.cell_start, params.num_grids, params.dt, params.vol, params.p_mass,
            params.inv_dx, params.dx, kernel=kernel,
        )

    return p2g


def cutile_v6_backend(*, num_grids=None, autotune=True, **_opts):
    """Arena P2G whose write-back is a single tile-coalesced atomic_store_add per
    block (no coloring, one launch): sort by SC=2 home super-cell, reduce each
    super-cell into a 4**3 arena, then one atomic_store_add per arena. The
    occupancy hint is autotuned per-GPU and cached (``autotune=False`` skips it ->
    compiler default; occupancy does not affect results, only speed)."""
    from mpm_jax.cutile_p2g import ARENA_SC  # pylint: disable=import-outside-toplevel

    if num_grids is not None and num_grids % ARENA_SC != 0:
        raise RuntimeError(
            f"cutile_v6_atomic_tile requires num_grids ({num_grids}) divisible by "
            f"arena_super_cell ({ARENA_SC})."
        )
    _require_cutile()
    kernel = None
    if autotune:
        from mpm_jax.cutile_autotune import tuned_atomic_tile_kernel  # pylint: disable=import-outside-toplevel

        kernel = tuned_atomic_tile_kernel()
    return Backend(
        name="cutile_v6_atomic_tile",
        prepare=_cutile_arena_prepare,
        p2g=_make_cutile_atomic_tile_p2g(kernel),
        g2p=_make_jax_scan_g2p_mls(),
        loop_kind="fori",
    )
