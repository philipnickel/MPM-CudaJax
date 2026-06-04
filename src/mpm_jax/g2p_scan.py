"""G2P gather via lax.scan over the 27 stencil offsets.

Motivation
----------
The vmap G2P in ``blocks/g2p.py`` gathers all 27 neighbour velocities at once
(``gv = grid_v[index]`` → ``(N, 27, 3)``) and forms the APIC outer products
``(N, 27, 3, 3)`` before reducing over the 27 axis. At 8M particles those
intermediates are ~2.6 GB + ~15 GB of HBM traffic per substep — the G2P analog
of the ``(N, 27, *)`` blow-up that ``p2g_scan`` was written to avoid (its
``loop_multiply_fusion`` was the single most expensive kernel in the JAX
baseline).

This module mirrors ``p2g_scan``: ``lax.scan`` over the 27 offsets, gather one
``(N, 3)`` node per step, and accumulate the velocity / APIC-affine /
velocity-gradient reconstruction into the carry. Peak intermediate is
``(N, 3, 3)``; the ``(N, 27, 3, 3)`` never materialises. It is the JAX analog of
the register-resident ``g2p_fused.cu`` used by the CUDA backends. Weights are
recomputed inline (identical math to ``weights._single_particle_weights``), so
``prepare`` need not build the ``(N, 27, *)`` weight arrays either.
"""

import jax
import jax.numpy as jnp

from mpm_jax.blocks.weights import OFFSET_27

OFFSET_27_INT = OFFSET_27.astype(jnp.int32)  # (27, 3)


def _weights_one_stencil(x_p, offset_int, inv_dx, dx, num_grids):
    """B-spline weight / dweight / dpos / flat-index for one particle, one node.

    Identical math to ``p2g_scan._single_particle_one_stencil`` (weights part)
    and ``weights._single_particle_weights`` (one offset slice), so the gathered
    reconstruction is bit-for-bit equal to the vmap g2p in ``blocks/g2p.py``.
    """
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)  # (3,)
    fx = px - base.astype(jnp.float32)            # (3,)

    w_table = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
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
    dpos = (offset_int.astype(jnp.float32) - fx) * dx        # (3,)

    idx_3d = base + offset_int                               # (3,)
    idx = idx_3d[0] * num_grids * num_grids + idx_3d[1] * num_grids + idx_3d[2]
    idx = jnp.clip(idx, 0, num_grids ** 3 - 1)
    return idx, weight, dweight, dpos


def _g2p_scan(grid_v, x, F, dt, inv_dx, dx, num_grids, clip_bound):
    """G2P via lax.scan over the 27 stencil offsets.

    Each scan body vmaps the single-particle weights over the N particles for
    ONE stencil offset, gathers the corresponding ``(N, 3)`` grid velocities,
    and accumulates the APIC reconstruction into the carry. The ``(N, 27, *)``
    intermediate never materialises.
    """
    N = x.shape[0]
    per_particle = jax.vmap(
        _weights_one_stencil, in_axes=(0, None, None, None, None))

    def scan_body(carry, offset_int):
        v_acc, C_acc, g_acc = carry
        idx, weight, dweight, dpos = per_particle(x, offset_int, inv_dx, dx, num_grids)
        gv = grid_v[idx]                                          # (N, 3) gather one node
        v_acc = v_acc + weight[:, None] * gv                      # (N, 3)
        C_acc = C_acc + weight[:, None, None] * (gv[:, :, None] * dpos[:, None, :])  # (N, 3, 3)
        g_acc = g_acc + gv[:, :, None] * dweight[:, None, :]      # (N, 3, 3)
        return (v_acc, C_acc, g_acc), None

    init = (jnp.zeros((N, 3), jnp.float32),
            jnp.zeros((N, 3, 3), jnp.float32),
            jnp.zeros((N, 3, 3), jnp.float32))
    (new_v, C_acc, grad_v), _ = jax.lax.scan(scan_body, init, OFFSET_27_INT)

    new_C = 4.0 * inv_dx * inv_dx * C_acc
    new_x = jnp.clip(x + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F + dt * jnp.einsum('nij,njk->nik', grad_v, F), -2.0, 2.0)
    return new_x, new_v, new_C, new_F


def _g2p_scan_mls(grid_v, x, F, dt, inv_dx, dx, num_grids, clip_bound):
    """Unified MLS-MPM G2P: reuse the APIC affine matrix C as the velocity
    gradient for the F-update (``F_new = (I + dt*C)*F``), dropping the separate
    B-spline ``grad_v`` (and its outer product + the ``dweight`` it needs). G2P
    therefore builds only ONE (N, 3, 3) accumulator instead of two — the true
    MLS-MPM formulation (Hu et al. 2018).

    NOTE: this is NOT bit-compatible with ``_g2p_scan`` — the APIC C differs
    from the B-spline ``grad_v`` by ~16%, so the deformation-gradient evolution
    (and hence the trajectory) changes. Same physics family, different (more
    standard) velocity-gradient discretisation.
    """
    N = x.shape[0]
    per_particle = jax.vmap(
        _weights_one_stencil, in_axes=(0, None, None, None, None))

    def scan_body(carry, offset_int):
        v_acc, C_acc = carry
        # dweight is unused here (XLA dead-code-eliminates it) -> no grad_v build.
        idx, weight, _dweight, dpos = per_particle(x, offset_int, inv_dx, dx, num_grids)
        gv = grid_v[idx]                                       # (N, 3) gather one node
        v_acc = v_acc + weight[:, None] * gv                   # (N, 3)
        C_acc = C_acc + weight[:, None, None] * (gv[:, :, None] * dpos[:, None, :])  # (N, 3, 3)
        return (v_acc, C_acc), None

    init = (jnp.zeros((N, 3), jnp.float32), jnp.zeros((N, 3, 3), jnp.float32))
    (new_v, C_acc), _ = jax.lax.scan(scan_body, init, OFFSET_27_INT)

    new_C = 4.0 * inv_dx * inv_dx * C_acc                      # APIC affine = MLS velocity gradient
    new_x = jnp.clip(x + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F + dt * jnp.einsum('nij,njk->nik', new_C, F), -2.0, 2.0)
    return new_x, new_v, new_C, new_F
