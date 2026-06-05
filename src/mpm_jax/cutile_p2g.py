"""cuTile P2G kernel called from the shared JAX-owned frame loop.

A single variant (``cutile_v6_atomic_tile``), sharing the JAX baseline G2P
(only P2G varies). It uses genuine tile primitives (``gather`` loads,
broadcast + ``where`` + ``sum`` reductions, a tile-coalesced ``atomic_store_add``
write-back) — no raw ``get_raw_memory``.

``cutile_v6_atomic_tile`` — SPGrid-style arena scatter. Each block sorts by home
super-cell (SC=2), reduces its OWN super-cell's particles (read once, no gather
redundancy) into a small L1-resident arena (4**3 = 64 nodes), then writes the
arena back to the global grid with a SINGLE tile-coalesced ``atomic_store_add``
per block. A 1-node halo pad aligns the arena's apron to an overlapping tiled
view, so overlapping apron nodes are reconciled by the per-element atomics — one
launch, no coloring, no parity.
"""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


# --- arena scatter (atomic-free reduction, single tile-coalesced atomic write) ---
# SPGrid-style: each block reduces its OWN super-cell's particles into a padded
# local arena (L1-resident), then writes the arena back to the global grid with
# one tile-coalesced atomic_store_add (no coloring, no read redundancy).
#
# Uses a SMALL super-cell (SC=2) so the arena is exactly 4**3 = 64 nodes -- a
# clean power-of-two tile with great occupancy, where each particle is read only
# ONCE (no gather redundancy) and evaluated against 64 nodes.
ARENA_SC = 2  # arena super-cell width
ARENA_DIM = ARENA_SC + 2  # 4 nodes per axis (SC + 1 apron each side)
ARENA_NODES = ARENA_DIM**3  # 64 arena nodes (power of two)
ARENA_PARTICLE_TILE = 16  # particles per chunk (occupancy sweet spot)


# ============================================================================
# Shared tile-math helpers (metaprogramming: emit cuTile ops inline)
# ============================================================================
def _gather_particle(x, v, C, stress, p, active, inv_dx):
    """Gather one chunk of P particles' state as (P,) tiles via real tile ops.

    ``x``/``v`` are flat (3N,), ``C``/``stress`` flat (9N,). OOB/inactive lanes
    return 0 (bounds-checked + ``mask``), matching a masked load. Returns 27 (P,)
    tiles; the kernel reshapes them to (P,1) columns via ``_cols``.
    """
    px0 = ct.gather(x, p * 3 + 0, mask=active, padding_value=0.0) * inv_dx
    px1 = ct.gather(x, p * 3 + 1, mask=active, padding_value=0.0) * inv_dx
    px2 = ct.gather(x, p * 3 + 2, mask=active, padding_value=0.0) * inv_dx
    b0 = ct.astype(ct.floor(px0 - 0.5), ct.int32)
    b1 = ct.astype(ct.floor(px1 - 0.5), ct.int32)
    b2 = ct.astype(ct.floor(px2 - 0.5), ct.int32)
    fx0 = px0 - ct.astype(b0, ct.float32)
    fx1 = px1 - ct.astype(b1, ct.float32)
    fx2 = px2 - ct.astype(b2, ct.float32)
    vp0 = ct.gather(v, p * 3 + 0, mask=active, padding_value=0.0)
    vp1 = ct.gather(v, p * 3 + 1, mask=active, padding_value=0.0)
    vp2 = ct.gather(v, p * 3 + 2, mask=active, padding_value=0.0)
    C00 = ct.gather(C, p * 9 + 0, mask=active, padding_value=0.0)
    C01 = ct.gather(C, p * 9 + 1, mask=active, padding_value=0.0)
    C02 = ct.gather(C, p * 9 + 2, mask=active, padding_value=0.0)
    C10 = ct.gather(C, p * 9 + 3, mask=active, padding_value=0.0)
    C11 = ct.gather(C, p * 9 + 4, mask=active, padding_value=0.0)
    C12 = ct.gather(C, p * 9 + 5, mask=active, padding_value=0.0)
    C20 = ct.gather(C, p * 9 + 6, mask=active, padding_value=0.0)
    C21 = ct.gather(C, p * 9 + 7, mask=active, padding_value=0.0)
    C22 = ct.gather(C, p * 9 + 8, mask=active, padding_value=0.0)
    s00 = ct.gather(stress, p * 9 + 0, mask=active, padding_value=0.0)
    s01 = ct.gather(stress, p * 9 + 1, mask=active, padding_value=0.0)
    s02 = ct.gather(stress, p * 9 + 2, mask=active, padding_value=0.0)
    s10 = ct.gather(stress, p * 9 + 3, mask=active, padding_value=0.0)
    s11 = ct.gather(stress, p * 9 + 4, mask=active, padding_value=0.0)
    s12 = ct.gather(stress, p * 9 + 5, mask=active, padding_value=0.0)
    s20 = ct.gather(stress, p * 9 + 6, mask=active, padding_value=0.0)
    s21 = ct.gather(stress, p * 9 + 7, mask=active, padding_value=0.0)
    s22 = ct.gather(stress, p * 9 + 8, mask=active, padding_value=0.0)
    return (
        b0,
        b1,
        b2,
        fx0,
        fx1,
        fx2,
        vp0,
        vp1,
        vp2,
        C00,
        C01,
        C02,
        C10,
        C11,
        C12,
        C20,
        C21,
        C22,
        s00,
        s01,
        s02,
        s10,
        s11,
        s12,
        s20,
        s21,
        s22,
    )


def _quad_w(o, fx):
    """Quadratic B-spline weight for offset tile ``o`` and fractional ``fx``."""
    return ct.where(
        o == 0,
        0.5 * (1.5 - fx) * (1.5 - fx),
        ct.where(o == 1, 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5)),
    )


def _quad_dw(o, fx):
    return ct.where(o == 0, fx - 1.5, ct.where(o == 1, -2.0 * (fx - 1.0), fx - 0.5))


def _node_contribution(pcols, node_rows, dt, vol, p_mass, inv_dx, dx):
    """Per (particle, node) MLS-MPM contribution as (P, Nn) tiles.

    ``pcols`` are the 27 per-particle (P,1) columns from ``_gather_particle``
    (reshaped); ``node_rows`` are the (1,Nn) grid-node coords. ``offset = node -
    base``; ``contributes`` masks the 3**3 stencil. Math is identical to the JAX
    baseline so the P2G matches it to fp32 round-off.
    """
    (
        b0,
        b1,
        b2,
        fx0,
        fx1,
        fx2,
        vp0,
        vp1,
        vp2,
        C00,
        C01,
        C02,
        C10,
        C11,
        C12,
        C20,
        C21,
        C22,
        s00,
        s01,
        s02,
        s10,
        s11,
        s12,
        s20,
        s21,
        s22,
    ) = pcols
    gi_row, gj_row, gk_row = node_rows

    ox = gi_row - b0
    oy = gj_row - b1
    oz = gk_row - b2
    contributes = (ox >= 0) & (ox < 3) & (oy >= 0) & (oy < 3) & (oz >= 0) & (oz < 3)
    ox_f = ct.astype(ox, ct.float32)
    oy_f = ct.astype(oy, ct.float32)
    oz_f = ct.astype(oz, ct.float32)

    wx, wy, wz = _quad_w(ox, fx0), _quad_w(oy, fx1), _quad_w(oz, fx2)
    dwx, dwy, dwz = _quad_dw(ox, fx0), _quad_dw(oy, fx1), _quad_dw(oz, fx2)

    weight = wx * wy * wz
    dw0 = inv_dx * dwx * wy * wz
    dw1 = inv_dx * wx * dwy * wz
    dw2 = inv_dx * wx * wy * dwz

    dpos0 = (ox_f - fx0) * dx
    dpos1 = (oy_f - fx1) * dx
    dpos2 = (oz_f - fx2) * dx

    affine0 = vp0 + C00 * dpos0 + C01 * dpos1 + C02 * dpos2
    affine1 = vp1 + C10 * dpos0 + C11 * dpos1 + C12 * dpos2
    affine2 = vp2 + C20 * dpos0 + C21 * dpos1 + C22 * dpos2

    stress_dw0 = s00 * dw0 + s01 * dw1 + s02 * dw2
    stress_dw1 = s10 * dw0 + s11 * dw1 + s12 * dw2
    stress_dw2 = s20 * dw0 + s21 * dw1 + s22 * dw2

    mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
    mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
    mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
    mass = p_mass * weight
    return mv0, mv1, mv2, mass, contributes


def _cols(particle, tile):
    """Reshape the 27 gathered (P,) tiles to (P,1) broadcast columns.

    Explicit (no comprehension/generator — cuTile compiles the kernel AST and
    rejects those).
    """
    (
        b0,
        b1,
        b2,
        fx0,
        fx1,
        fx2,
        vp0,
        vp1,
        vp2,
        C00,
        C01,
        C02,
        C10,
        C11,
        C12,
        C20,
        C21,
        C22,
        s00,
        s01,
        s02,
        s10,
        s11,
        s12,
        s20,
        s21,
        s22,
    ) = particle
    return (
        ct.reshape(b0, (tile, 1)),
        ct.reshape(b1, (tile, 1)),
        ct.reshape(b2, (tile, 1)),
        ct.reshape(fx0, (tile, 1)),
        ct.reshape(fx1, (tile, 1)),
        ct.reshape(fx2, (tile, 1)),
        ct.reshape(vp0, (tile, 1)),
        ct.reshape(vp1, (tile, 1)),
        ct.reshape(vp2, (tile, 1)),
        ct.reshape(C00, (tile, 1)),
        ct.reshape(C01, (tile, 1)),
        ct.reshape(C02, (tile, 1)),
        ct.reshape(C10, (tile, 1)),
        ct.reshape(C11, (tile, 1)),
        ct.reshape(C12, (tile, 1)),
        ct.reshape(C20, (tile, 1)),
        ct.reshape(C21, (tile, 1)),
        ct.reshape(C22, (tile, 1)),
        ct.reshape(s00, (tile, 1)),
        ct.reshape(s01, (tile, 1)),
        ct.reshape(s02, (tile, 1)),
        ct.reshape(s10, (tile, 1)),
        ct.reshape(s11, (tile, 1)),
        ct.reshape(s12, (tile, 1)),
        ct.reshape(s20, (tile, 1)),
        ct.reshape(s21, (tile, 1)),
        ct.reshape(s22, (tile, 1)),
    )


# ============================================================================
# v6 — arena scatter via a single tile-coalesced atomic_store_add (no coloring)
# ============================================================================
# Each block reduces its SC=2 super-cell's particles into a 4**3-node L1-resident
# arena, then writes the arena back with ONE tile-coalesced atomic per arena.
# A 1-node halo pad makes the arena's -1 apron origin align to an overlapping
# TiledView (tile=(SC+2)**3, traversal_steps=SC), so each block does
# view.atomic_store_add(super_cell_index, arena) and overlapping apron nodes are
# reconciled by the per-element atomics. No coloring, no parity, one launch.
# Occupancy is left to the cuTile compiler default (ncu flags this kernel
# register-limited, so the best value is GPU-specific; no hint is forced here).
@ct.kernel
def _p2g_atomic_tile_kernel(
    x,
    v,
    C,
    stress,
    cell_start,
    grid_mv,
    grid_m,
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
    tile_i = si * SC - 1  # real-grid arena origin (apron at -1)
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
        pcols = _cols(
            _gather_particle(x, v, C, stress, p, active, inv_dx), particle_tile
        )
        active_col = ct.reshape(active, (particle_tile, 1))
        mv0, mv1, mv2, mass, contributes = _node_contribution(
            pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx
        )
        m = contributes & active_col
        acc0 = acc0 + ct.sum(ct.where(m, mv0, 0.0), axis=0)
        acc1 = acc1 + ct.sum(ct.where(m, mv1, 0.0), axis=0)
        acc2 = acc2 + ct.sum(ct.where(m, mv2, 0.0), axis=0)
        accm = accm + ct.sum(ct.where(m, mass, 0.0), axis=0)
        chunk_start += particle_tile

    # One coalesced atomic-add per arena into the padded grid. The overlapping
    # tiled view places this super-cell's (SC+2)**3 arena at tile index (si,sj,sk);
    # apron nodes shared with neighbours are reconciled by the per-element atomics.
    view_mv = grid_mv.tiled_view((DIM, DIM, DIM, 1), traversal_steps=(SC, SC, SC, 1))
    view_m = grid_m.tiled_view((DIM, DIM, DIM), traversal_steps=(SC, SC, SC))
    view_mv.atomic_store_add((si, sj, sk, 0), ct.reshape(acc0, (DIM, DIM, DIM, 1)))
    view_mv.atomic_store_add((si, sj, sk, 1), ct.reshape(acc1, (DIM, DIM, DIM, 1)))
    view_mv.atomic_store_add((si, sj, sk, 2), ct.reshape(acc2, (DIM, DIM, DIM, 1)))
    view_m.atomic_store_add((si, sj, sk), ct.reshape(accm, (DIM, DIM, DIM)))


def cutile_p2g_atomic_tile(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """One-launch arena P2G: a single tile-coalesced atomic_store_add per block,
    no coloring. A 1-node halo pad aligns the apron to an overlapping tiled view.
    """
    g = int(num_grids)
    g3 = g**3
    gp = g + 2  # 1-node halo on each side
    gs = g // ARENA_SC
    x_flat = x.reshape(-1)
    v_flat = v.reshape(-1)
    C_flat = C.reshape(-1)
    stress_flat = stress.reshape(-1)

    grid_mv = jnp.zeros((gp, gp, gp, 3), dtype=jnp.float32)
    grid_m = jnp.zeros((gp, gp, gp), dtype=jnp.float32)

    grid_mv, grid_m = cutile_call(
        (gs, gs, gs),
        _p2g_atomic_tile_kernel,
        (
            x_flat,
            v_flat,
            C_flat,
            stress_flat,
            cell_start,
            InputOutput(grid_mv),
            InputOutput(grid_m),
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
    # Strip the halo -> real (G**3, 3) / (G**3,) grids.
    grid_mv = grid_mv[1 : g + 1, 1 : g + 1, 1 : g + 1, :].reshape((g3, 3))
    grid_m = grid_m[1 : g + 1, 1 : g + 1, 1 : g + 1].reshape((g3,))
    return grid_mv, grid_m
