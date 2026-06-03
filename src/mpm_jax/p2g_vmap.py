"""Materialized P2G scatter baseline using vmap-friendly array operations.

This is the simple pure-JAX baseline: compute all 27 stencil-node
contributions as ``(N, 27, *)`` arrays, flatten them, and scatter-add into the
grid. It is useful as a reference implementation and for profiling because XLA
has fewer loop-carried scatter bodies than the memory-saving ``jax_v1_5`` scan
path, but its peak HBM footprint is much larger.
"""

import jax.numpy as jnp


def _p2g_vmap(weight, dweight, dpos, index, v, C, stress, dt, vol, p_mass, num_grids):
    """P2G with fully materialized 27-stencil particle contributions."""
    G3 = num_grids ** 3
    grid_mv0 = jnp.zeros((G3, 3), dtype=jnp.float32)
    grid_m0 = jnp.zeros((G3,), dtype=jnp.float32)

    stress_term = jnp.einsum("nij,nkj->nki", stress, dweight)
    affine = v[:, None, :] + jnp.einsum("nij,nkj->nki", C, dpos)
    mv = -dt * vol * stress_term + p_mass * weight[:, :, None] * affine
    mass = p_mass * weight

    flat_index = index.reshape(-1)
    grid_mv = grid_mv0.at[flat_index].add(mv.reshape(-1, 3))
    grid_m = grid_m0.at[flat_index].add(mass.reshape(-1))
    return grid_mv, grid_m
