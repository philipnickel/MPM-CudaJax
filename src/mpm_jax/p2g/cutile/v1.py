"""Direct cuTile P2G scatter used as a comparison backend."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.p2g.cutile.common import (
    GRID_CHANNELS,
    MAT_USED,
    STENCIL_LANES,
    STENCIL_NODES,
    _atomic_add_grid_channels,
    _channel_tile,
    _load_particle_columns,
    _stencil_contribution_columns,
)


PARTICLES_PER_TILE = 4


@ct.kernel
def cutile_v1_p2g_kernel(
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
    particles_per_tile: ct.Constant[int],
):
    block = ct.bid(0)
    p_lane = ct.arange(particles_per_tile, dtype=ct.int32)
    chunk_start = block * particles_per_tile
    p = chunk_start + p_lane
    in_bounds = p < n_particles
    pcols = _load_particle_columns(x, v, C, stress, p, inv_dx, particles_per_tile)

    active_lane = ct.reshape(in_bounds, (particles_per_tile, 1))
    offset = ct.reshape(
        ct.arange(STENCIL_LANES, dtype=ct.int32), (1, STENCIL_LANES)
    )
    oi = offset // 9
    oj = (offset // 3) % 3
    ok = offset % 3

    b0, b1, b2 = pcols[0]
    gi = b0 + oi
    gj = b1 + oj
    gk = b2 + ok
    mv0, mv1, mv2, mass = _stencil_contribution_columns(
        pcols, (oi, oj, ok), dt, vol, p_mass, inv_dx, dx
    )

    valid_stencil = offset < STENCIL_NODES
    mask = active_lane & valid_stencil
    contrib = _channel_tile(
        mv0, mv1, mv2, mass, (particles_per_tile, STENCIL_LANES, 1), 2
    )
    _atomic_add_grid_channels(
        grid,
        (gi, gj, gk),
        contrib,
        G,
        (particles_per_tile, STENCIL_LANES, 1),
        (1, 1, GRID_CHANNELS),
        mask,
    )


def cutile_p2g_v1(
    x,
    v,
    C,
    stress,
    num_grids,
    dt,
    vol,
    p_mass,
    inv_dx,
    dx,
    *,
    particles_per_tile=PARTICLES_PER_TILE,
    kernel_hints=None,
):
    """Direct cuTile scatter for comparison with the arena backend."""
    n = int(x.shape[0])
    g = int(num_grids)
    g3 = g**3
    particles_per_tile = int(particles_per_tile)
    blocks = (n + particles_per_tile - 1) // particles_per_tile
    C = C.reshape((n, MAT_USED))
    stress = stress.reshape((n, MAT_USED))

    grid = jnp.zeros((g, g, g, GRID_CHANNELS), dtype=jnp.float32)
    kernel = cutile_v1_p2g_kernel
    if kernel_hints:
        kernel = kernel.replace_hints(**kernel_hints)
    grid = cutile_call(
        (blocks,),
        kernel,
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
            particles_per_tile,
        ),
    )
    return grid[..., :3].reshape((g3, 3)), grid[..., 3].reshape((g3,))
