"""Fast cuTile arena P2G scatter."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.cutile_common import (
    _load_particle_columns,
    _node_contribution_columns,
    _pack4,
)


ARENA_SC = 2
ARENA_DIM = ARENA_SC + 2
ARENA_NODES = ARENA_DIM**3
ARENA_PARTICLE_TILE = 16


def _add4(a0, a1, a2, a3, b0, b1, b2, b3):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3


def _masked_sum4(mask, mv0, mv1, mv2, mass):
    return ct.reduce(
        (
            ct.where(mask, mv0, 0.0),
            ct.where(mask, mv1, 0.0),
            ct.where(mask, mv2, 0.0),
            ct.where(mask, mass, 0.0),
        ),
        axis=0,
        func=_add4,
        identity=(0.0, 0.0, 0.0, 0.0),
    )


@ct.kernel
def _p2g_atomic_tile_kernel(
    x,
    v,
    C,
    stress,
    cell_bounds,
    grid,
    dt: ct.Constant[float],
    vol: ct.Constant[float],
    p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float],
    dx: ct.Constant[float],
    particle_tile: ct.Constant[int],
    node_tile: ct.Constant[int],
):
    SC = ARENA_SC
    DIM = ARENA_DIM
    si = ct.bid(0)
    sj = ct.bid(1)
    sk = ct.bid(2)
    tile_i = si * SC - 1
    tile_j = sj * SC - 1
    tile_k = sk * SC - 1

    p_start = ct.gather(cell_bounds, (si, sj, sk, 0))
    p_end = ct.gather(cell_bounds, (si, sj, sk, 1))
    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    node_lane = ct.arange(node_tile, dtype=ct.int32)
    ti = node_lane // (DIM * DIM)
    tj = (node_lane // DIM) % DIM
    tk = node_lane % DIM
    gi_row = ct.reshape(tile_i + ti, (1, node_tile))
    gj_row = ct.reshape(tile_j + tj, (1, node_tile))
    gk_row = ct.reshape(tile_k + tk, (1, node_tile))

    acc0 = ct.zeros((node_tile,), ct.float32)
    acc1 = ct.zeros((node_tile,), ct.float32)
    acc2 = ct.zeros((node_tile,), ct.float32)
    accm = ct.zeros((node_tile,), ct.float32)
    chunk_start = p_start
    while chunk_start < p_end:
        p = chunk_start + p_lane
        active = p < p_end
        pcols = _load_particle_columns(
            x, v, C, stress, p, active, inv_dx, particle_tile
        )
        active_col = ct.reshape(active, (particle_tile, 1))
        mv0, mv1, mv2, mass, contributes = _node_contribution_columns(
            pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx
        )
        m = contributes & active_col
        sum0, sum1, sum2, summ = _masked_sum4(m, mv0, mv1, mv2, mass)
        acc0 = acc0 + sum0
        acc1 = acc1 + sum1
        acc2 = acc2 + sum2
        accm = accm + summ
        chunk_start += particle_tile

    arena = _pack4(acc0, acc1, acc2, accm, (DIM, DIM, DIM, 1), 3)
    view = grid.tiled_view((DIM, DIM, DIM, 4), traversal_steps=(SC, SC, SC, 4))
    view.atomic_store_add((si, sj, sk, 0), arena)


def cutile_p2g_v2(
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
):
    """One-launch arena P2G using one tile-coalesced atomic write per super-cell."""
    g = int(num_grids)
    g3 = g**3
    gp = g + 2
    gs = g // ARENA_SC
    n = int(v.shape[0])
    C = C.reshape((n, 9))
    stress = stress.reshape((n, 9))

    grid = jnp.zeros((gp, gp, gp, 4), dtype=jnp.float32)

    grid = cutile_call(
        (gs, gs, gs),
        _p2g_atomic_tile_kernel,
        (
            x,
            v,
            C,
            stress,
            cell_bounds,
            InputOutput(grid),
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            ARENA_PARTICLE_TILE,
            ARENA_NODES,
        ),
    )
    grid = grid[1 : g + 1, 1 : g + 1, 1 : g + 1, :]
    grid_mv = grid[..., :3].reshape((g3, 3))
    grid_m = grid[..., 3].reshape((g3,))
    return grid_mv, grid_m
