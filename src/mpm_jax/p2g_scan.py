"""P2G scatter via lax.scan over the 27 stencil offsets.

Motivation
----------
The default JAX path materialises ``(N, 27, 3)`` momentum and ``(N, 27)`` mass
tensors before the scatter — peak HBM traffic that does not exist in the fused
CUDA kernel where each thread keeps the 27 contributions in registers.

This module attacks that single bottleneck without leaving JAX: rather than
vmap over particles for all 27 stencil nodes at once, we ``lax.scan`` over the
27 offsets. Each scan iteration vmaps over the N particles for one offset and
scatters the resulting ``(N, 3)`` / ``(N,)`` contributions into the grid via
``at[].add``. Peak intermediate buffer is ``(N, *)``, never ``(N, 27, *)``.

The CUDA inline backends take the same HBM-saving idea further by running the
27-stencil scatter inside one register-resident CUDA kernel.
"""

import jax.numpy as jnp

import jax

from mpm_jax.blocks.weights import OFFSET_27


# 27 integer stencil offsets in (i, j, k) order.
# Materialise as int32 once at import; the per-stencil B-spline indices and
# index arithmetic are integer ops.
OFFSET_27_INT = OFFSET_27.astype(jnp.int32)  # (27, 3)


def _single_particle_one_stencil(x_p, v_p, C_p, stress_p, offset_int,
                                  dt, vol, p_mass, dx, inv_dx, num_grids):
    """One particle's contribution to ONE stencil node.

    Mirrors ``solver._single_particle_p2g`` mathematically but for a single
    ``(i, j, k)`` offset triplet, so the work fits in registers and avoids
    materialising the 27-axis dimension.

    Parameters
    ----------
    x_p, v_p : (3,) float32
    C_p, stress_p : (3, 3) float32
    offset_int : (3,) int32   — one of the 27 integer offsets
    dt, vol, p_mass, dx, inv_dx : scalars
    num_grids : Python int

    Returns
    -------
    idx : ()    int32   — flat grid index for this particle's stencil node
    mv_s : (3,) float32 — momentum contribution
    m_s  : ()   float32 — mass contribution
    """
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)  # (3,)
    fx = px - base.astype(jnp.float32)            # (3,)

    # 1D quadratic B-spline weights, evaluated only for THIS stencil triple.
    # w_axis[k] for axis a is the per-axis weight when the offset along
    # axis a equals k. We index by the integer offset value (0, 1, or 2).
    w_table = jnp.stack([
        0.5 * (1.5 - fx) ** 2,    # offset == 0
        0.75 - (fx - 1.0) ** 2,   # offset == 1
        0.5 * (fx - 0.5) ** 2,    # offset == 2
    ])  # (3, 3): [offset_value, spatial_axis]
    dw_table = jnp.stack([
        fx - 1.5,
        -2.0 * (fx - 1.0),
        fx - 0.5,
    ])  # (3, 3)

    ix, iy, iz = offset_int[0], offset_int[1], offset_int[2]
    wx, wy, wz = w_table[ix, 0], w_table[iy, 1], w_table[iz, 2]
    dwx, dwy, dwz = dw_table[ix, 0], dw_table[iy, 1], dw_table[iz, 2]

    weight = wx * wy * wz                                    # scalar
    dweight = inv_dx * jnp.stack([                            # (3,)
        dwx * wy * wz,
        wx * dwy * wz,
        wx * wy * dwz,
    ])

    offset_f = offset_int.astype(jnp.float32)                 # (3,)
    dpos = (offset_f - fx) * dx                               # (3,)

    # Affine momentum — single-stencil version of solver._single_particle_p2g.
    # stress @ dweight is (3, 3) @ (3,) -> (3,); C @ dpos is (3, 3) @ (3,) -> (3,).
    mv_s = (
        -dt * vol * (stress_p @ dweight)
        + p_mass * weight * (v_p + C_p @ dpos)
    )
    m_s = weight * p_mass

    # Flat index, identical to solver._single_particle_weights.
    idx_3d = base + offset_int  # (3,)
    idx = idx_3d[0] * num_grids * num_grids + idx_3d[1] * num_grids + idx_3d[2]
    idx = jnp.clip(idx, 0, num_grids ** 3 - 1)

    return idx, mv_s, m_s


def _p2g_scan(x, v, C, stress, dt, vol, p_mass, dx, inv_dx, num_grids):
    """P2G via lax.scan over the 27 stencil offsets.

    Each scan body call vmaps the single-particle, single-stencil function
    over the N particles for ONE stencil offset and scatters the (N, 3) /
    (N,) contributions into the carry grids. The (N, 27, *) intermediate
    never materialises.
    """
    G = num_grids
    grid_mv0 = jnp.zeros((G ** 3, 3), dtype=jnp.float32)
    grid_m0 = jnp.zeros((G ** 3,), dtype=jnp.float32)

    # vmap the single-particle, single-stencil function over the N particles.
    # offset / scalars are broadcast (in_axes=None).
    per_particle_one_stencil = jax.vmap(
        _single_particle_one_stencil,
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None),
    )

    def scan_body(carry, offset_int):
        gm, gmv = carry  # gm: (G^3,), gmv: (G^3, 3)
        idx, mv_s, m_s = per_particle_one_stencil(
            x, v, C, stress, offset_int,
            dt, vol, p_mass, dx, inv_dx, num_grids,
        )
        # Scatter the (N,) / (N, 3) contributions onto the grid. XLA lowers
        # this to atomicAdd on GPU exactly like the baseline scatter.
        gmv = gmv.at[idx].add(mv_s)
        gm = gm.at[idx].add(m_s)
        return (gm, gmv), None

    (grid_m, grid_mv), _ = jax.lax.scan(scan_body, (grid_m0, grid_mv0), OFFSET_27_INT)
    return grid_mv, grid_m
