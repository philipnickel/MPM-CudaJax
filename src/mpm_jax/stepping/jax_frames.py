import jax

from mpm_jax.stepping.substep import step  # neutral home — no import back into solver
from mpm_jax.types import MPMState
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update


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


def build_jax_v1_5_frame(params, elasticity_fn, plasticity_fn,
                         pre_particle_fn, post_grid_fn, steps_per_frame,
                         *, loop_kind="fori", **_ignored):
    """JAX frame using lax.scan over 27 stencil offsets for P2G.

    Avoids materialising the (N, 27, *) HBM intermediates of the default P2G
    path by scanning over the 27 offsets one at a time.
    """
    # jax_v1_5 always Python-unrolls the stencil-scan substep; loop_kind is
    # accepted only for signature uniformity with other frame builders and is ignored.
    # Deferred import: p2g_scan imports from mpm_jax.types and mpm_jax.blocks,
    # but not from jax_frames — no actual cycle.  The import stays deferred so
    # that p2g_scan (a less commonly used path) is only loaded when this builder
    # is actually called.
    from mpm_jax.p2g_scan import _p2g_scan  # pylint: disable=import-outside-toplevel

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("p2g_scan"):
                grid_mv, grid_m = _p2g_scan(
                    x, v, state.C, stress,
                    params.dt, params.vol, params.p_mass,
                    params.dx, params.inv_dx, params.num_grids,
                )
            with jax.named_scope("grid_update"):
                grid_mv_normalized = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv_normalized, grid_m, 0.0)
            with jax.named_scope("g2p"):
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
