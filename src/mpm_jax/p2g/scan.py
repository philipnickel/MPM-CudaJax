"""P2G scatter via lax.scan over the 27 stencil offsets, avoiding the (N, 27, *) intermediate."""

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
    """P2G via lax.scan over the 27 stencil offsets, scattering one packed tile per offset."""
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
