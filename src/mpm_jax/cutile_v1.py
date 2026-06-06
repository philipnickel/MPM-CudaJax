"""Direct cuTile P2G scatter used as a comparison backend."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.cutile_common import _load_particle_columns, _node_contribution_columns


DIRECT_STENCIL_TILE = 32
DIRECT_PARTICLE_TILE = 16


@ct.kernel
def _p2g_direct_kernel(
    x,
    v,
    C,
    stress,
    grid,
    n_particles: ct.Constant[int],
    G: ct.Constant[int],
    dt: ct.Constant[float],
    vol: ct.Constant[float],
    p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float],
    dx: ct.Constant[float],
    particle_tile: ct.Constant[int],
    stencil_tile: ct.Constant[int],
):
    block = ct.bid(0)
    p_lane = ct.arange(particle_tile, dtype=ct.int32)
    chunk_start = block * particle_tile
    p = chunk_start + p_lane
    in_bounds = p < n_particles
    pcols = _load_particle_columns(
        x, v, C, stress, p, in_bounds, inv_dx, particle_tile
    )

    active_lane = ct.reshape(in_bounds, (particle_tile, 1))
    offset = ct.reshape(ct.arange(stencil_tile, dtype=ct.int32), (1, stencil_tile))
    oi = offset // 9
    oj = (offset // 3) % 3
    ok = offset % 3

    b0, b1, b2 = pcols[0]
    gi = b0 + oi
    gj = b1 + oj
    gk = b2 + ok
    mv0, mv1, mv2, mass, contributes = _node_contribution_columns(
        pcols, (gi, gj, gk), dt, vol, p_mass, inv_dx, dx
    )

    valid_stencil = offset < 27
    mask = contributes & active_lane & valid_stencil
    flat = gi * (G * G) + gj * G + gk
    flat = ct.maximum(0, ct.minimum(flat, G * G * G - 1))

    channel = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 1, 4))
    flat = ct.reshape(flat, (particle_tile, stencil_tile, 1))
    mask = ct.reshape(mask, (particle_tile, stencil_tile, 1))
    contrib = ct.cat(
        (
            ct.cat(
                (
                    ct.reshape(mv0, (particle_tile, stencil_tile, 1)),
                    ct.reshape(mv1, (particle_tile, stencil_tile, 1)),
                ),
                2,
            ),
            ct.cat(
                (
                    ct.reshape(mv2, (particle_tile, stencil_tile, 1)),
                    ct.reshape(mass, (particle_tile, stencil_tile, 1)),
                ),
                2,
            ),
        ),
        2,
    )
    ct.atomic_add(grid, (flat, channel), ct.where(mask, contrib, 0.0))


def cutile_p2g_v1(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Direct cuTile scatter for comparison with the arena backend."""
    n = int(x.shape[0])
    g = int(num_grids)
    g3 = g**3
    blocks = (n + DIRECT_PARTICLE_TILE - 1) // DIRECT_PARTICLE_TILE
    C = C.reshape((n, 9))
    stress = stress.reshape((n, 9))

    grid = jnp.zeros((g3, 4), dtype=jnp.float32)
    grid = cutile_call(
        (blocks,),
        _p2g_direct_kernel,
        (
            x,
            v,
            C,
            stress,
            InputOutput(grid),
            n,
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            DIRECT_PARTICLE_TILE,
            DIRECT_STENCIL_TILE,
        ),
    )
    return grid[:, :3], grid[:, 3]
