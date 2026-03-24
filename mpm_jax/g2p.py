"""G2P — MLS-MPM formulation (Hu et al. 2018)."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27
from mpm_jax.weights import compute_weights_single


def _single_particle_g2p(grid_v, F_p, x_p, dt, inv_dx, clip_bound):
    weight, dpos_grid, index = compute_weights_single(x_p, inv_dx, jnp.int32(inv_dx))
    gv = grid_v[index]  # (27, 3)

    new_v = (weight[:, None] * gv).sum(axis=0)
    new_C = 4.0 * inv_dx * (weight[:, None, None] * gv[:, :, None] * dpos_grid[:, None, :]).sum(axis=0)
    new_F = jnp.clip((jnp.eye(3) + dt * new_C) @ F_p, -2.0, 2.0)
    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)

    return new_x, new_v, new_C, new_F


@jax.jit
def g2p(state: MPMState, grid_v: jnp.ndarray, params: MPMParams) -> MPMState:
    new_x, new_v, new_C, new_F = jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, None, None, None),
    )(grid_v, state.F, state.x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
