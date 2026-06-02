import jax
import jax.numpy as jnp


def _single_particle_g2p(grid_mv, weight, dweight, dpos, index, F_p, x_p, dt, inv_dx, clip_bound):
    """Compute G2P gather for one particle (one CUDA thread).

    Args:
        grid_mv: (G^3, 3) grid velocities (read-only)
        weight:  (27,)  B-spline weights
        dweight: (27, 3) weight gradients
        dpos:    (27, 3) particle-to-node offsets
        index:   (27,)  flat grid indices
        F_p:     (3, 3) deformation gradient
        x_p:     (3,)   particle position
        dt, inv_dx, clip_bound: scalars

    Returns:
        new_x: (3,)   updated position
        new_v: (3,)   updated velocity
        new_C: (3, 3) updated APIC matrix
        new_F: (3, 3) updated deformation gradient
    """
    gv = grid_mv[index]  # (27, 3) — gather from grid
    new_v = (weight[:, None] * gv).sum(axis=0)  # (3,)
    new_C = 4.0 * inv_dx * inv_dx * (weight[:, None, None] * jnp.einsum('ij,ik->ijk', gv, dpos)).sum(axis=0)  # (3, 3)
    grad_v = jnp.einsum('ij,ik->ijk', gv, dweight).sum(axis=0)  # (3, 3)

    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)  # (3,)
    new_F = jnp.clip(F_p + dt * grad_v @ F_p, -2.0, 2.0)  # (3, 3)

    return new_x, new_v, new_C, new_F


def g2p(grid_mv, weight, dweight, dpos, index, F, x, dt, inv_dx, clip_bound):
    """G2P gather via vmap (embarrassingly parallel over particles)."""
    return jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, None),
    )(grid_mv, weight, dweight, dpos, index, F, x, dt, inv_dx, clip_bound)
