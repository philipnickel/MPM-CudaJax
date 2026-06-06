"""P2G scatter via lax.scan over the 27 stencil offsets.

Motivation
----------
The default JAX path materialises ``(N, 27, 3)`` momentum and ``(N, 27)`` mass
tensors before the scatter — peak HBM traffic that does not exist in the fused
CUDA kernel where each thread keeps the 27 contributions in registers.

This module attacks that single bottleneck without leaving JAX: rather than
vmap over particles for all 27 stencil nodes at once, we ``lax.scan`` over the
27 offsets. Each scan iteration computes one batched ``(N, 4)`` contribution
tile, ``[mv_x, mv_y, mv_z, mass]``, and scatters it into a packed grid via
``at[].add``. Peak intermediate buffer is ``(N, *)``, never ``(N, 27, *)``.

The CUDA backends take the same HBM-saving idea further by running the
27-stencil scatter inside one register-resident CUDA kernel.
"""

import jax
import jax.numpy as jnp

from mpm_jax.p2g.stencil import OFFSET_27, weight_dweight_dpos_index


def _one_offset_contrib(
    x, v, C, stress, offset_int, dt, vol, p_mass, dx, inv_dx, num_grids
):
    """All particles' contribution to one stencil node."""
    idx, weight, dweight, dpos = weight_dweight_dpos_index(
        x, offset_int, inv_dx, dx, num_grids
    )

    stress_term = jnp.matvec(stress, dweight)
    affine = jnp.matvec(C, dpos)
    mv = -dt * vol * stress_term + p_mass * weight[:, None] * (v + affine)
    mass = weight * p_mass

    return idx, jnp.concatenate((mv, mass[:, None]), axis=1)


def _p2g_scan(x, v, C, stress, dt, vol, p_mass, dx, inv_dx, num_grids):
    """P2G via lax.scan over the 27 stencil offsets.

    Each scan body call computes all particles' contribution for ONE stencil
    offset and scatters the (N, 4) packed momentum/mass tile into the carry
    grid. The (N, 27, *) intermediate never materialises.
    """
    G = num_grids
    grid0 = jnp.zeros((G**3, 4), dtype=jnp.float32)

    def scan_body(grid, offset_int):
        idx, contrib = _one_offset_contrib(
            x,
            v,
            C,
            stress,
            offset_int,
            dt,
            vol,
            p_mass,
            dx,
            inv_dx,
            num_grids,
        )
        return grid.at[idx].add(contrib), None

    grid, _ = jax.lax.scan(scan_body, grid0, OFFSET_27, unroll=True)
    return grid[:, :3], grid[:, 3]
