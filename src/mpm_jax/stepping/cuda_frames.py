"""CUDA-backed per-frame JIT builders.

These builders mirror ``build_jax_frame`` in signature but route P2G (and
optionally G2P) through hand-written CUDA kernels registered via JAX FFI.

FFI helpers stay in ``mpm_jax.cuda.p2g_cuda``; this module only assembles
the per-frame scan/loop around them.  Imports from ``p2g_cuda`` are deferred
to function call time so that ``p2g_cuda`` can re-export these names at its
end without creating a load-time cycle.
"""

import jax
import jax.numpy as jnp

from mpm_jax.types import MPMState
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update


def build_cuda_v1_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame, **_ignored):
    """Per-frame JIT'd function using the cuda_v1_inline P2G kernel.

    Mirrors ``build_jax_frame`` but routes P2G through one CUDA kernel call
    (inline weights + 27-stencil atomic scatter per particle, no ``(N, 27, *)``
    momentum tensor in HBM).  G2P also uses a CUDA kernel (``g2p_fused.cu``)
    so neither scatter/gather stage materialises ``(N, 27, *)`` tensors.

    Result is one ``@jax.jit`` + Python-unrolled loop over ``steps_per_frame``
    — a single XLA program per frame.  Stress and plasticity stay in JAX
    (model-agnostic); only the two scatter/gather kernels are CUDA.
    """
    # Deferred to avoid load-time cycle: p2g_cuda re-exports this function at
    # its bottom, so a top-level import here would create a circular dependency.
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        is_available,
        cuda_p2g_inline,
        cuda_g2p_fused,
    )

    if not is_available('inline'):
        raise RuntimeError(
            "cuda_v1_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("p2g_inline"):
                grid_mv, grid_m = cuda_p2g_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                new_x, new_v, new_C, new_F = cuda_g2p_fused(
                    x, state.F, grid_v,
                    params.num_grids, params.dt,
                    params.inv_dx, params.dx, params.clip_bound,
                )

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def build_cuda_v2_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame,
                        *, loop_kind="fori", **_ignored):
    """Per-frame JIT'd function using the cuda_v2_inline P2G kernel.

    Identical structure to ``build_cuda_v1_frame``; only the P2G FFI call is
    swapped for the warp-reduction variant.  ``loop_kind`` controls whether
    substeps are unrolled at trace time (``"python"``) or lowered to an
    ``lax.fori_loop`` (``"fori"``).
    """
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        is_available,
        cuda_p2g_v2_inline,
        cuda_g2p_fused,
    )

    if not is_available('v2_inline'):
        raise RuntimeError(
            "cuda_v2_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    @jax.jit
    def jit_frame(state):
        def step_body(state):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("p2g_v2_inline"):
                grid_mv, grid_m = cuda_p2g_v2_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                new_x, new_v, new_C, new_F = cuda_g2p_fused(
                    x, state.F, grid_v,
                    params.num_grids, params.dt,
                    params.inv_dx, params.dx, params.clip_bound,
                )

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        def fori_body(_, state):
            return step_body(state)

        if loop_kind == "fori":
            state = jax.lax.fori_loop(0, steps_per_frame, fori_body, state)
        else:
            for _ in range(steps_per_frame):
                state = step_body(state)
        return state

    return jit_frame


def build_cuda_v3_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame,
                        *, loop_kind="fori", **_ignored):
    """Per-frame JIT'd function using cuda_v3_inline (Morton sort + warp shuffle).

    Each substep sorts particles by Morton (Z-order) code, then runs the
    inline + warp-shuffle P2G kernel + CUDA G2P.  State persists in sorted
    order across substeps (re-sorted each substep on the new positions).

    ``loop_kind`` controls whether substeps are Python-unrolled or lowered to
    an ``lax.fori_loop``.

    Note: the former ``cuda_v6_inline`` variant was ``cuda_v3`` plus an XLA
    CUDA-graph flag set in ``simulate.py`` *before* ``import jax`` — there is
    nothing graph-specific inside this builder.  A ``cuda_graph`` kwarg passed
    by the registry is silently absorbed by ``**_ignored``.
    """
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        is_available,
        cuda_p2g_v3_inline,
        cuda_g2p_fused,
    )
    from mpm_jax.blocks.sort import morton_argsort  # pylint: disable=import-outside-toplevel

    if not is_available('v3_inline'):
        raise RuntimeError(
            "cuda_v3_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    @jax.jit
    def jit_frame(state):
        def step_body(state):
            with jax.named_scope("morton_sort"):
                order = morton_argsort(state.x, params.inv_dx, params.num_grids)
                x_sorted = state.x[order]
                v_sorted = state.v[order]
                C_sorted = state.C[order]
                F_sorted = state.F[order]

            with jax.named_scope("pre_particle"):
                x, v = pre_fn(x_sorted, v_sorted, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(F_sorted)
            with jax.named_scope("p2g_v3_inline"):
                grid_mv, grid_m = cuda_p2g_v3_inline(
                    x, v, C_sorted, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                new_x, new_v, new_C, new_F = cuda_g2p_fused(
                    x, F_sorted, grid_v,
                    params.num_grids, params.dt,
                    params.inv_dx, params.dx, params.clip_bound,
                )

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        def fori_body(_, state):
            return step_body(state)

        if loop_kind == "fori":
            state = jax.lax.fori_loop(0, steps_per_frame, fori_body, state)
        else:
            for _ in range(steps_per_frame):
                state = step_body(state)
        return state

    return jit_frame


def build_cuda_v4_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame, **_ignored):
    """Per-frame JIT'd function using the cuda_v4_inline P2G kernel.

    Each substep argsorts particles by home super-cell, builds a CSR
    ``cell_start`` array, and runs the cell-major + smem-tile P2G kernel.
    State persists in sorted order across substeps.
    """
    from mpm_jax.cuda.p2g_cuda import (  # pylint: disable=import-outside-toplevel
        is_available,
        cuda_p2g_v4_inline,
        cuda_g2p_fused,
        V4_SUPER_CELL_WIDTH,
        _home_super_cell_id,
    )

    if not is_available('v4_inline'):
        raise RuntimeError(
            "cuda_v4_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    G = params.num_grids
    sc = V4_SUPER_CELL_WIDTH
    if G % sc != 0:
        raise RuntimeError(
            f"cuda_v4_inline requires num_grids ({G}) divisible by "
            f"super-cell width ({sc})."
        )
    Gs = G // sc
    Gs3 = Gs ** 3
    super_boundaries = jnp.arange(Gs3 + 1, dtype=jnp.int32)

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope("super_cell_sort"):
                super_id = _home_super_cell_id(x, params.inv_dx, G, sc)
                order = jnp.argsort(super_id)

                x_s = x[order]
                v_s = v[order]
                C_s = state.C[order]
                stress_s = stress[order]
                F_s = state.F[order]

                super_id_sorted = super_id[order]
                cell_start = jnp.searchsorted(
                    super_id_sorted, super_boundaries
                ).astype(jnp.int32)

            with jax.named_scope("p2g_v4_inline"):
                grid_mv, grid_m = cuda_p2g_v4_inline(
                    x_s, v_s, C_s, stress_s, cell_start,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )

            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                new_x, new_v, new_C, new_F = cuda_g2p_fused(
                    x_s, F_s, grid_v,
                    params.num_grids, params.dt,
                    params.inv_dx, params.dx, params.clip_bound,
                )

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame
