"""Backend interface for the shared JAX-owned MPM frame loop."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update
from mpm_jax.blocks.weights import compute_weights_and_indices
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
                        *, loop_kind=None, **_ignored):
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

        if loop_kind == "fori":
            return jax.lax.fori_loop(0, steps_per_frame, lambda _, s: step_body(s), state)
        for _ in range(steps_per_frame):
            state = step_body(state)
        return state

    return jit_frame


def _jax_scan_prepare(params, state, stress):
    weight, dweight, dpos, index = compute_weights_and_indices(
        state.x, params.inv_dx, params.dx, params.num_grids)
    return PreparedSubstep(
        state.x, state.v, state.C, state.F, stress,
        weight=weight, dweight=dweight, dpos=dpos, index=index,
    )


def _make_jax_scan_p2g():
    from mpm_jax.p2g_scan import _p2g_scan  # pylint: disable=import-outside-toplevel

    def p2g(params, prepared):
        return _p2g_scan(
            prepared.x, prepared.v, prepared.C, prepared.stress,
            params.dt, params.vol, params.p_mass,
            params.dx, params.inv_dx, params.num_grids,
        )

    return p2g


def _jax_g2p(params, prepared, grid_v):
    return g2p(
        grid_v, prepared.weight, prepared.dweight, prepared.dpos, prepared.index,
        prepared.F, prepared.x, params.dt, params.inv_dx, params.clip_bound,
    )


def jax_v1_5_backend(**_opts):
    return Backend(
        name="jax_v1_5",
        prepare=_jax_scan_prepare,
        p2g=_make_jax_scan_p2g(),
        g2p=_jax_g2p,
        loop_kind="fori",
    )


def _cuda_g2p(params, prepared, grid_v):
    from mpm_jax.cuda.p2g_cuda import cuda_g2p_fused  # pylint: disable=import-outside-toplevel

    return cuda_g2p_fused(
        prepared.x, prepared.F, grid_v,
        params.num_grids, params.dt,
        params.inv_dx, params.dx, params.clip_bound,
    )


def _require_cuda(*kernels):
    from mpm_jax.cuda.p2g_cuda import is_available  # pylint: disable=import-outside-toplevel

    for kernel in kernels:
        if not is_available(kernel):
            raise RuntimeError(
                f"CUDA kernel {kernel!r} not registered (missing .so?). "
                "Run `pixi install -e gpu` to build."
            )


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
    _require_cuda("inline", "g2p_fused")
    return Backend(
        name="cuda_v1_inline",
        p2g=_cuda_inline_p2g("inline"),
        g2p=_cuda_g2p,
        loop_kind="fori",
    )


def cuda_v2_backend(**_opts):
    _require_cuda("v2_inline", "g2p_fused")
    return Backend(
        name="cuda_v2_inline",
        p2g=_cuda_inline_p2g("v2_inline"),
        g2p=_cuda_g2p,
        loop_kind="fori",
    )


def _morton_prepare(params, state, stress):
    from mpm_jax.blocks.sort import morton_argsort  # pylint: disable=import-outside-toplevel

    order = morton_argsort(state.x, params.inv_dx, params.num_grids)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order], stress[order]
    )


def cuda_v3_backend(**_opts):
    _require_cuda("v3_inline", "g2p_fused")
    return Backend(
        name="cuda_v3_inline",
        prepare=_morton_prepare,
        p2g=_cuda_inline_p2g("v3_inline"),
        g2p=_cuda_g2p,
        loop_kind="fori",
    )


def _supercell_boundaries(params, super_cell_width):
    Gs = params.num_grids // super_cell_width
    return jnp.arange(Gs ** 3 + 1, dtype=jnp.int32)


def _cuda_v4_prepare(params, state, stress):
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        V4_SUPER_CELL_WIDTH,
        _home_super_cell_id,
    )

    super_id = _home_super_cell_id(
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
    _require_cuda("v4_inline", "g2p_fused")
    return Backend(
        name="cuda_v4_inline",
        prepare=_cuda_v4_prepare,
        p2g=_cuda_v4_p2g,
        g2p=_cuda_g2p,
        loop_kind="fori",
    )


def _warp_prepare(params, state, stress):
    from mpm_jax.stepping.warp_hybrid_frame import (  # pylint: disable=import-outside-toplevel
        SUPER_CELL_WIDTH,
        _home_super_cell_id,
    )

    super_id = _home_super_cell_id(state.x, params.inv_dx, params.num_grids, SUPER_CELL_WIDTH)
    order = jnp.argsort(super_id)
    super_id_sorted = super_id[order]
    cell_start = jnp.searchsorted(
        super_id_sorted, _supercell_boundaries(params, SUPER_CELL_WIDTH)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order],
        stress[order], cell_start=cell_start,
    )


def _warp_p2g(jax_p2g):
    def p2g(params, prepared):
        from mpm_jax.stepping.warp_hybrid_frame import (  # pylint: disable=import-outside-toplevel
            warp_p2g_supercell_tile,
        )

        return warp_p2g_supercell_tile(
            jax_p2g, prepared.x, prepared.v, prepared.C, prepared.stress,
            prepared.cell_start, params.num_grids, params.dt, params.vol,
            params.p_mass, params.inv_dx, params.dx,
        )

    return p2g


def warp_v3_supercell_backend(*, graph_mode="jax", num_grids=None, **_opts):
    from mpm_jax.stepping.warp_hybrid_frame import (  # pylint: disable=import-outside-toplevel
        SUPER_CELL_WIDTH,
        _make_jax_p2g_supercell_tile,
    )

    if num_grids is not None and num_grids % SUPER_CELL_WIDTH != 0:
        raise RuntimeError(
            f"warp_v3_supercell_tile requires num_grids ({num_grids}) "
            f"divisible by super-cell width ({SUPER_CELL_WIDTH})."
        )
    _require_cuda("g2p_fused")
    return Backend(
        name="warp_v3_supercell_tile",
        prepare=_warp_prepare,
        p2g=_warp_p2g(_make_jax_p2g_supercell_tile(graph_mode)),
        g2p=_cuda_g2p,
        loop_kind="fori",
    )
