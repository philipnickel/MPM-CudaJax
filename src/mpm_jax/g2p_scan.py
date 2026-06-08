"""G2P gather via lax.scan over the 27 stencil offsets.

Scanning keeps the APIC reconstruction at ``(N, 3, 3)`` peak instead of
materialising ``(N, 27, *)`` intermediates. This G2P path is shared by every
backend, so only P2G varies.
"""

import jax
import jax.numpy as jnp

from mpm_jax.p2g.stencil import OFFSET_27, weight_dpos_index


def _g2p_scan_mls(grid_v, x, F, dt, inv_dx, dx, num_grids, clip_bound):
    """Unified MLS-MPM G2P using APIC affine ``C`` as the velocity gradient.

    This keeps one ``(N, 3, 3)`` accumulator and follows the standard MLS-MPM
    discretisation rather than a separate B-spline ``grad_v`` estimator.
    """
    N = x.shape[0]

    def scan_body(carry, offset_int):
        v_acc, C_acc = carry
        idx, weight, dpos = weight_dpos_index(x, offset_int, inv_dx, dx, num_grids)
        gv = grid_v[idx]
        v_acc = v_acc + weight[:, None] * gv
        C_acc = C_acc + weight[:, None, None] * (
            gv[:, :, None] * dpos[:, None, :]
        )
        return (v_acc, C_acc), None

    init = (jnp.zeros((N, 3), jnp.float32), jnp.zeros((N, 3, 3), jnp.float32))
    (new_v, C_acc), _ = jax.lax.scan(scan_body, init, OFFSET_27, unroll=True)

    new_C = 4.0 * inv_dx * inv_dx * C_acc  # APIC C is the velocity gradient
    new_x = jnp.clip(x + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F + dt * jnp.einsum("nij,njk->nik", new_C, F), -2.0, 2.0)
    return new_x, new_v, new_C, new_F
