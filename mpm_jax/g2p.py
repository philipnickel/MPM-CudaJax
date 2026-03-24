"""Grid-to-particle transfer (MLS-MPM, Hu et al. 2018).

G2P (no dweight needed):
    v_new = sum(weight * v_grid)
    C_new = 4 * inv_dx * sum(weight * v_grid ⊗ dpos_grid)
    F_new = (I + dt * C_new) @ F
    x_new = x + dt * v_new

C is computed with grid-space dpos and 4*inv_dx.
F update uses C directly as velocity gradient (MLS-MPM key insight).
"""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27


def _single_particle_g2p(grid_v, F_p, x_p, dt, inv_dx, clip_bound):
    """G2P for one particle."""
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)
    num_grids = jnp.int32(inv_dx)

    w = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])

    offsets = OFFSET_27.astype(int)
    weight = w[offsets[:, 0], 0] * w[offsets[:, 1], 1] * w[offsets[:, 2], 2]

    # Grid-space dpos (unitless)
    dpos_grid = OFFSET_27 - fx[None, :]

    idx_3d = base[None, :] + offsets
    index = idx_3d[:, 0] * num_grids * num_grids + idx_3d[:, 1] * num_grids + idx_3d[:, 2]
    index = jnp.clip(index, 0, num_grids ** 3 - 1)

    gv = grid_v[index]  # (27, 3)

    # Velocity
    new_v = (weight[:, None] * gv).sum(axis=0)

    # APIC C with grid-space dpos: C = 4*inv_dx * sum(w * gv ⊗ dpos_grid)
    new_C = 4.0 * inv_dx * (
        weight[:, None, None] * jnp.einsum('ij,ik->ijk', gv, dpos_grid)
    ).sum(axis=0)

    # MLS-MPM F update: C is the velocity gradient
    new_F = jnp.clip((jnp.eye(3) + dt * new_C) @ F_p, -2.0, 2.0)

    # Position
    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)

    return new_x, new_v, new_C, new_F


@jax.jit
def g2p(state: MPMState, grid_v: jnp.ndarray, params: MPMParams) -> MPMState:
    """Grid-to-particle: gather velocities and update particle state."""
    new_x, new_v, new_C, new_F = jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, None, None, None),
    )(grid_v, state.F, state.x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
