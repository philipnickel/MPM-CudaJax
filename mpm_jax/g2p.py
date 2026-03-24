"""Grid-to-particle transfer: gather velocities, update particle state."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27


def _single_particle_g2p(grid_v, F_p, x_p, dt, inv_dx, clip_bound):
    """G2P for one particle. Computes weight, dweight, dpos internally."""
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)
    dx = 1.0 / inv_dx
    num_grids = jnp.int32(inv_dx)

    w = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])

    dw = jnp.stack([
        fx - 1.5,
        -2.0 * (fx - 1.0),
        fx - 0.5,
    ])

    offsets = OFFSET_27.astype(int)
    weight = w[offsets[:, 0], 0] * w[offsets[:, 1], 1] * w[offsets[:, 2], 2]

    dweight = inv_dx * jnp.stack([
        dw[offsets[:, 0], 0] *  w[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] * dw[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] *  w[offsets[:, 1], 1] * dw[offsets[:, 2], 2],
    ], axis=-1)

    dpos = (OFFSET_27 - fx[None, :]) * dx

    idx_3d = base[None, :] + offsets
    index = idx_3d[:, 0] * num_grids * num_grids + idx_3d[:, 1] * num_grids + idx_3d[:, 2]
    index = jnp.clip(index, 0, num_grids ** 3 - 1)

    # Gather grid velocities
    gv = grid_v[index]  # (27, 3)

    # Velocity: weighted average
    new_v = (weight[:, None] * gv).sum(axis=0)

    # APIC affine matrix
    new_C = 4.0 * inv_dx * inv_dx * (
        weight[:, None, None] * jnp.einsum('ij,ik->ijk', gv, dpos)
    ).sum(axis=0)

    # Velocity gradient via dweight (for F update)
    grad_v = jnp.einsum('ij,ik->ijk', gv, dweight).sum(axis=0)

    # Update F and position
    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F_p + dt * grad_v @ F_p, -2.0, 2.0)

    return new_x, new_v, new_C, new_F


@jax.jit
def g2p(state: MPMState, grid_v: jnp.ndarray, params: MPMParams) -> MPMState:
    """Grid-to-particle: gather velocities and update particle state."""
    new_x, new_v, new_C, new_F = jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, None, None, None),
    )(grid_v, state.F, state.x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
