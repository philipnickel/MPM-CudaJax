"""Quadratic B-spline weights and grid indices for 3x3x3 stencil."""
import jax
import jax.numpy as jnp
from mpm_jax.state import OFFSET_27

_OFFSETS_INT = OFFSET_27.astype(int)


def compute_weights_single(x_p, inv_dx, num_grids):
    """Compute weights, grid-space dpos, and flat indices for one particle.

    Returns:
        weight:    (27,)  B-spline weights
        dpos_grid: (27,3) grid-space offsets (unitless)
        index:     (27,)  flat grid indices
    """
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)

    w = jnp.stack([0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2])
    weight = w[_OFFSETS_INT[:,0],0] * w[_OFFSETS_INT[:,1],1] * w[_OFFSETS_INT[:,2],2]

    dpos_grid = OFFSET_27 - fx[None, :]

    idx_3d = base[None, :] + _OFFSETS_INT
    index = idx_3d[:,0]*num_grids*num_grids + idx_3d[:,1]*num_grids + idx_3d[:,2]
    index = jnp.clip(index, 0, num_grids**3 - 1)

    return weight, dpos_grid, index


def compute_weights_batch(x, inv_dx, dx, num_grids):
    """Batched weights. Returns real-space dpos (grid-space * dx).

    Returns:
        weight: (N, 27)  B-spline weights
        dpos:   (N, 27, 3) real-space offsets
        index:  (N, 27)  flat grid indices
    """
    weight, dpos_grid, index = jax.vmap(
        compute_weights_single, in_axes=(0, None, None)
    )(x, inv_dx, num_grids)
    return weight, dpos_grid * dx, index
