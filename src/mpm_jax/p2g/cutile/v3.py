"""One-cell cuTile P2G scatter with a shared 27-node stencil."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.p2g.cutile.common import (
    GRID_CHANNELS,
    MAT_USED,
    STENCIL_LANES,
    STENCIL_NODES,
    _atomic_add_grid_channels,
    _load_particle_columns,
    _masked_channel_sum,
    _stencil_contribution_columns,
)


PARTICLES_PER_TILE = 8


def _home_stencil(home_i, home_j, home_k):
    node = ct.reshape(ct.arange(STENCIL_LANES, dtype=ct.int32), (1, STENCIL_LANES))
    oi = node // 9
    oj = (node // 3) % 3
    ok = node % 3
    offsets = (oi, oj, ok)
    nodes = (home_i - 1 + oi, home_j - 1 + oj, home_k - 1 + ok)
    return nodes, offsets, node < STENCIL_NODES


def _scatter_stencil(grid, nodes, values, G):
    _atomic_add_grid_channels(
        grid,
        nodes,
        values,
        G,
        (STENCIL_LANES, 1),
        (1, GRID_CHANNELS),
    )


@ct.kernel
def _p2g_home_cell_kernel(
    x,
    v,
    C,
    stress,
    cell_bounds,
    grid,
    G: ct.Constant[int],
    dt: ct.Constant[float],
    vol: ct.Constant[float],
    p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float],
    dx: ct.Constant[float],
    particles_per_tile: ct.Constant[int],
):
    hi = ct.bid(0)
    hj = ct.bid(1)
    hk = ct.bid(2)

    p_start = ct.gather(cell_bounds, (hi, hj, hk, 0), check_bounds=False)
    p_end = ct.gather(cell_bounds, (hi, hj, hk, 1), check_bounds=False)
    if p_start < p_end:
        p_lane = ct.arange(particles_per_tile, dtype=ct.int32)
        nodes, offsets, valid_node = _home_stencil(hi, hj, hk)

        acc = ct.zeros((STENCIL_LANES, GRID_CHANNELS), ct.float32)
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end
            pcols = _load_particle_columns(
                x, v, C, stress, p, inv_dx, particles_per_tile
            )
            active_col = ct.reshape(active, (particles_per_tile, 1))
            mv0, mv1, mv2, mass = _stencil_contribution_columns(
                pcols, offsets, dt, vol, p_mass, inv_dx, dx
            )
            mask = active_col & valid_node
            acc = acc + _masked_channel_sum(
                mask,
                mv0,
                mv1,
                mv2,
                mass,
                (particles_per_tile, STENCIL_LANES, 1),
                2,
            )
            chunk_start += particles_per_tile

        _scatter_stencil(grid, nodes, acc, G)


def cutile_p2g_v3(
    x,
    v,
    C,
    stress,
    cell_bounds,
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
    """Cell-owned cuTile scatter: reduce one home cell's 27-node stencil locally."""
    g = int(num_grids)
    g3 = g**3
    n = int(v.shape[0])
    particles_per_tile = int(particles_per_tile)
    C = C.reshape((n, MAT_USED))
    stress = stress.reshape((n, MAT_USED))

    grid = jnp.zeros((g, g, g, GRID_CHANNELS), dtype=jnp.float32)
    cells_per_axis = g + 1
    kernel = _p2g_home_cell_kernel
    if kernel_hints:
        kernel = kernel.replace_hints(**kernel_hints)
    grid = cutile_call(
        (cells_per_axis, cells_per_axis, cells_per_axis),
        kernel,
        (
            x,
            v,
            C,
            stress,
            cell_bounds,
            InputOutput(grid),
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
