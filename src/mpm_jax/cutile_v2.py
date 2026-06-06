"""Fast cuTile arena P2G scatter."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.cutile_common import _quad_dw, _quad_w


ARENA_SC = 2
ARENA_DIM = ARENA_SC + 2
ARENA_NODES = ARENA_DIM**3
ARENA_PARTICLE_TILE = 16


def _vector_columns(tile, particle_tile):
    columns = ()
    for axis in ct.static_iter(range(3)):
        column = ct.extract(tile, (0, axis), shape=(particle_tile, 1))
        columns += (column,)
    return columns


def _matrix_columns(tile, particle_tile):
    columns = ()
    for row in ct.static_iter(range(3)):
        for col in ct.static_iter(range(3)):
            column = ct.extract(tile, (0, row, col), shape=(particle_tile, 1, 1))
            columns += (ct.reshape(column, (particle_tile, 1)),)
    return columns


def _load_particle_columns(x, v, C, stress, p, active, inv_dx, particle_tile):
    active_v = ct.reshape(active, (particle_tile, 1))
    active_m = ct.reshape(active, (particle_tile, 1, 1))

    x_p = ct.load_advanced_indexing(
        x,
        (p, ct.Slice(0, 4)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    v_p = ct.load_advanced_indexing(
        v,
        (p, ct.Slice(0, 4)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    C_p = ct.load_advanced_indexing(
        C,
        (p, ct.Slice(0, 4), ct.Slice(0, 4)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    stress_p = ct.load_advanced_indexing(
        stress,
        (p, ct.Slice(0, 4), ct.Slice(0, 4)),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x_p = ct.where(active_v, x_p, 0.0) * inv_dx
    v_p = ct.where(active_v, v_p, 0.0)
    C_p = ct.where(active_m, C_p, 0.0)
    stress_p = ct.where(active_m, stress_p, 0.0)

    b = ct.astype(ct.floor(x_p - 0.5), ct.int32)
    fx = x_p - ct.astype(b, ct.float32)
    return (
        _vector_columns(b, particle_tile),
        _vector_columns(fx, particle_tile),
        _vector_columns(v_p, particle_tile),
        _matrix_columns(C_p, particle_tile),
        _matrix_columns(stress_p, particle_tile),
    )


def _node_contribution_columns(pcols, node_rows, dt, vol, p_mass, inv_dx, dx):
    b, fx, velocity, C, stress = pcols
    b0, b1, b2 = b
    fx0, fx1, fx2 = fx
    vp0, vp1, vp2 = velocity
    C00, C01, C02, C10, C11, C12, C20, C21, C22 = C
    s00, s01, s02, s10, s11, s12, s20, s21, s22 = stress
    gi_row, gj_row, gk_row = node_rows

    ox = gi_row - b0
    oy = gj_row - b1
    oz = gk_row - b2
    contributes = (ox >= 0) & (ox < 3) & (oy >= 0) & (oy < 3) & (oz >= 0) & (oz < 3)

    wx, wy, wz = _quad_w(ox, fx0), _quad_w(oy, fx1), _quad_w(oz, fx2)
    dwx, dwy, dwz = _quad_dw(ox, fx0), _quad_dw(oy, fx1), _quad_dw(oz, fx2)

    weight = wx * wy * wz
    dw0 = inv_dx * dwx * wy * wz
    dw1 = inv_dx * wx * dwy * wz
    dw2 = inv_dx * wx * wy * dwz

    dpos0 = (ct.astype(ox, ct.float32) - fx0) * dx
    dpos1 = (ct.astype(oy, ct.float32) - fx1) * dx
    dpos2 = (ct.astype(oz, ct.float32) - fx2) * dx

    affine0 = vp0 + C00 * dpos0 + C01 * dpos1 + C02 * dpos2
    affine1 = vp1 + C10 * dpos0 + C11 * dpos1 + C12 * dpos2
    affine2 = vp2 + C20 * dpos0 + C21 * dpos1 + C22 * dpos2

    stress_dw0 = s00 * dw0 + s01 * dw1 + s02 * dw2
    stress_dw1 = s10 * dw0 + s11 * dw1 + s12 * dw2
    stress_dw2 = s20 * dw0 + s21 * dw1 + s22 * dw2

    return (
        -dt * vol * stress_dw0 + p_mass * weight * affine0,
        -dt * vol * stress_dw1 + p_mass * weight * affine1,
        -dt * vol * stress_dw2 + p_mass * weight * affine2,
        p_mass * weight,
        contributes,
    )


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
    cell_start,
    grid,
    G: ct.Constant[int],
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
    Gs = G // SC
    si = ct.bid(0)
    sj = ct.bid(1)
    sk = ct.bid(2)
    super_id = si * (Gs * Gs) + sj * Gs + sk
    tile_i = si * SC - 1
    tile_j = sj * SC - 1
    tile_k = sk * SC - 1

    p_start = ct.gather(cell_start, super_id)
    p_end = ct.gather(cell_start, super_id + 1)
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

    arena0 = ct.reshape(acc0, (DIM, DIM, DIM, 1))
    arena1 = ct.reshape(acc1, (DIM, DIM, DIM, 1))
    arena2 = ct.reshape(acc2, (DIM, DIM, DIM, 1))
    arenam = ct.reshape(accm, (DIM, DIM, DIM, 1))
    arena = ct.cat((ct.cat((arena0, arena1), 3), ct.cat((arena2, arenam), 3)), 3)
    view = grid.tiled_view((DIM, DIM, DIM, 4), traversal_steps=(SC, SC, SC, 4))
    view.atomic_store_add((si, sj, sk, 0), arena)


def cutile_p2g_v2(
    x,
    v,
    C,
    stress,
    cell_start,
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

    grid = jnp.zeros((gp, gp, gp, 4), dtype=jnp.float32)

    grid = cutile_call(
        (gs, gs, gs),
        _p2g_atomic_tile_kernel,
        (
            x,
            v,
            C,
            stress,
            cell_start,
            InputOutput(grid),
            g,
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
