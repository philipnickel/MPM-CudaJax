"""Warp-backed per-frame JIT builders.

These builders mirror ``build_jax_frame`` in signature but route P2G (and
optionally G2P) through Warp kernels embedded in JAX via jax_callable /
jax_kernel.

Warp kernel defs and helpers stay in ``mpm_jax.warp_kernels``; this module
only assembles the per-frame scan/loop around them.  Imports from
``warp_kernels`` (and from ``mpm_jax.solver``) are deferred to function
call time because Warp is a heavy, optional GPU dependency — importing it
at module load time would break CPU-only installs.
"""

import jax
import jax.numpy as jnp


def build_warp_v1_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame, **_ignored):
    """Build a fully JIT'd frame using Warp P2G via JAX FFI."""
    # Deferred: warp_kernels is a GPU-only dependency; defer so CPU installs
    # work and Warp is not imported until this builder is actually called.
    from mpm_jax.solver import (  # pylint: disable=import-outside-toplevel
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )
    from mpm_jax.warp_kernels import warp_p2g_inline  # pylint: disable=import-outside-toplevel

    use_cuda_g2p = _ignored.get("use_cuda_g2p", True)

    if use_cuda_g2p:
        from mpm_jax.cuda.p2g_cuda import cuda_g2p_fused, is_available  # pylint: disable=import-outside-toplevel

        if not is_available("g2p_fused"):
            raise RuntimeError(
                "cuda g2p kernel not registered (missing .so?). "
                "Pass use_cuda_g2p=False to fall back to the JAX G2P."
            )
    else:
        cuda_g2p_fused = None

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("warp_p2g_inline"):
                grid_mv, grid_m = warp_p2g_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, state.F, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        state.F, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def build_warp_v2_tile_frame(params, elasticity_fn, plasticity_fn,
                             pre_fn, post_fn, steps_per_frame, **_ignored):
    """Build a fully JIT'd frame using tiled Warp P2G via JAX FFI."""
    # Deferred: warp_kernels is a GPU-only dependency; defer so CPU installs
    # work and Warp is not imported until this builder is actually called.
    from mpm_jax.solver import (  # pylint: disable=import-outside-toplevel
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )
    from mpm_jax.warp_kernels import warp_p2g_inline_tile  # pylint: disable=import-outside-toplevel

    use_cuda_g2p = _ignored.get("use_cuda_g2p", True)

    if use_cuda_g2p:
        from mpm_jax.cuda.p2g_cuda import cuda_g2p_fused, is_available  # pylint: disable=import-outside-toplevel

        if not is_available("g2p_fused"):
            raise RuntimeError(
                "cuda g2p kernel not registered (missing .so?). "
                "Pass use_cuda_g2p=False to fall back to the JAX G2P."
            )
    else:
        cuda_g2p_fused = None

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("warp_p2g_tile"):
                grid_mv, grid_m = warp_p2g_inline_tile(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, state.F, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        state.F, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def build_warp_v3_frame(params, elasticity_fn, plasticity_fn,
                        pre_fn, post_fn, steps_per_frame, **_ignored):
    """Build a frame using a super-cell-owned Warp tile P2G kernel."""
    # Deferred: warp_kernels is a GPU-only dependency; defer so CPU installs
    # work and Warp is not imported until this builder is actually called.
    from mpm_jax.solver import (  # pylint: disable=import-outside-toplevel
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )
    from mpm_jax.warp_kernels import (  # pylint: disable=import-outside-toplevel
        SUPER_CELL_WIDTH,
        warp_p2g_supercell_tile,
        _home_super_cell_id,
    )

    use_cuda_g2p = _ignored.get("use_cuda_g2p", True)

    if params.num_grids % SUPER_CELL_WIDTH != 0:
        raise RuntimeError(
            f"warp_v3_supercell_tile requires num_grids ({params.num_grids}) "
            f"divisible by super-cell width ({SUPER_CELL_WIDTH})."
        )

    if use_cuda_g2p:
        from mpm_jax.cuda.p2g_cuda import cuda_g2p_fused, is_available  # pylint: disable=import-outside-toplevel

        if not is_available("g2p_fused"):
            raise RuntimeError(
                "cuda g2p kernel not registered (missing .so?). "
                "Pass use_cuda_g2p=False to fall back to the JAX G2P."
            )
    else:
        cuda_g2p_fused = None

    G = params.num_grids
    Gs = G // SUPER_CELL_WIDTH
    Gs3 = Gs ** 3
    super_boundaries = jnp.arange(Gs3 + 1, dtype=jnp.int32)

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope("warp_supercell_sort"):
                super_id = _home_super_cell_id(x, params.inv_dx, G, SUPER_CELL_WIDTH)
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

            with jax.named_scope("warp_p2g_supercell_tile"):
                grid_mv, grid_m = warp_p2g_supercell_tile(
                    x_s, v_s, C_s, stress_s, cell_start,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )

            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x_s, F_s, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x_s, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        F_s, x_s, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame
