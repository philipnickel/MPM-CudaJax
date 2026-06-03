import jax

from mpm_jax.types import MPMState
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.p2g import p2g
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update


def step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=None):
    """One full P2G2P step.

    Pure function — safe to JIT when closures are captured.
    Optionally accepts a custom p2g_fn to swap the scatter implementation.
    """
    p2g_fn = p2g_fn or p2g

    # Pre-particle BCs
    with jax.named_scope("pre_particle"):
        x, v = pre_particle_fn(state.x, state.v, time)

    # Weights (vmap over particles)
    with jax.named_scope("weights"):
        weight, dweight, dpos, index = compute_weights_and_indices(
            x, params.inv_dx, params.dx, params.num_grids)

    # P2G: compute + scatter
    with jax.named_scope("p2g"):
        grid_mv, grid_m = p2g_fn(
            v, state.C, stress, weight, dweight, dpos, index,
            params.dt, params.vol, params.p_mass, params.num_grids)

    # Grid update
    with jax.named_scope("grid_update"):
        grid_mv = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)

    # Post-grid BCs
    with jax.named_scope("post_grid"):
        grid_mv = post_grid_fn(grid_mv, grid_m, time)

    # G2P (vmap over particles)
    with jax.named_scope("g2p"):
        new_x, new_v, new_C, new_F = g2p(
            grid_mv, weight, dweight, dpos, index,
            state.F, x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
