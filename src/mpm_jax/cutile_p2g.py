"""cuTile P2G kernel called from the shared JAX-owned frame loop.

The backend shares the JAX baseline G2P; only P2G varies. It uses genuine tile
primitives (``load``/``gather`` reads, broadcast + ``where`` + ``sum``
reductions, and a tile-coalesced ``atomic_store_add`` write-back) — no raw
``get_raw_memory``.

``cutile_v1`` — simple comparison path. Each block loads a particle tile,
broadcasts it against the 27 quadratic B-spline offsets, and atomically adds
directly to the flat grid. This keeps a cuTile-native baseline around without
the arena reduction.

``cutile_v2`` — SPGrid-style arena scatter. Each block owns one super-cell,
reduces all of that super-cell's particle chunks into a small L1-resident arena
(4**3 = 64 nodes), then writes the arena back once with a tile-coalesced
``atomic_store_add``. A 1-node halo pad aligns the arena's apron to an
overlapping tiled view, so overlapping apron nodes are reconciled by per-element
atomics — one launch, no coloring, no parity.
"""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


# --- arena scatter (local chunk reductions, one tile-coalesced atomic write) ---
# SPGrid-style: each block reduces its super-cell's particle chunks into a
# padded local arena (L1-resident), then writes the arena back to the global
# grid with one tile-coalesced atomic_store_add (no coloring).
#
# Uses a SMALL super-cell (SC=2) so the arena is exactly 4**3 = 64 nodes -- a
# clean power-of-two tile with great occupancy, where each particle is read only
# once and evaluated against 64 nodes.
ARENA_SC = 2  # arena super-cell width
ARENA_DIM = ARENA_SC + 2  # 4 nodes per axis (SC + 1 apron each side)
ARENA_NODES = ARENA_DIM**3  # 64 arena nodes (power of two)
ARENA_PARTICLE_TILE = 16  # particles per chunk (occupancy sweet spot)
DIRECT_STENCIL_TILE = 32
DIRECT_PARTICLE_TILE = 16


# ============================================================================
# Shared tile-math helpers (metaprogramming: emit cuTile ops inline)
# ============================================================================
def _quad_w(o, fx):
    """Quadratic B-spline weight for offset tile ``o`` and fractional ``fx``."""
    return ct.where(
        o == 0,
        0.5 * (1.5 - fx) * (1.5 - fx),
        ct.where(o == 1, 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5)),
    )


def _quad_dw(o, fx):
    return ct.where(o == 0, fx - 1.5, ct.where(o == 1, -2.0 * (fx - 1.0), fx - 0.5))


def _load_particle_chunk(x, v, C, stress, start, stop, inv_dx, particle_tile):
    x_p = (
        ct.load(
            x.slice(0, start, stop),
            (0, 0),
            (particle_tile, 4),
            padding_mode=ct.PaddingMode.ZERO,
        )
        * inv_dx
    )
    v_p = ct.load(
        v.slice(0, start, stop),
        (0, 0),
        (particle_tile, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    C_p = ct.load(
        C.slice(0, start, stop),
        (0, 0, 0),
        (particle_tile, 4, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    stress_p = ct.load(
        stress.slice(0, start, stop),
        (0, 0, 0),
        (particle_tile, 4, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b = ct.astype(ct.floor(x_p - 0.5), ct.int32)
    return b, x_p - ct.astype(b, ct.float32), v_p, C_p, stress_p


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


def _component_axis(shape):
    return ct.broadcast_to(ct.reshape(ct.arange(4, dtype=ct.int32), (1, 1, 4)), shape)


def _node_xyz(gi, gj, gk, shape):
    axis = _component_axis(shape)
    return ct.where(
        axis == 0,
        ct.broadcast_to(gi, shape),
        ct.where(axis == 1, ct.broadcast_to(gj, shape), ct.broadcast_to(gk, shape)),
    )


def _node_contribution(
    particles, node, dt, vol, p_mass, inv_dx, dx, particle_tile, node_tile
):
    """Per (particle, node, channel) MLS-MPM contribution tile.

    Channels 0-2 are momentum and channel 3 is mass. Inputs keep their natural
    cuTile shapes: particles are vectors/matrices, nodes are grid XYZ vectors.
    """
    b, fx, velocity, C, stress = particles
    channel = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 1, 4))
    component = channel < 3

    base = ct.reshape(b, (particle_tile, 1, 4))
    fx = ct.reshape(fx, (particle_tile, 1, 4))
    offset = node - base

    in_stencil = (offset >= 0) & (offset < 3)
    valid_component = ct.where(component, ct.astype(in_stencil, ct.int32), 1)
    contributes = ct.prod(valid_component, axis=2) != 0

    w = ct.where(component, _quad_w(offset, fx), 1.0)
    dw = ct.where(component, _quad_dw(offset, fx), 0.0)
    weight = ct.prod(w, axis=2)
    dweight = inv_dx * dw * ct.reshape(weight, (particle_tile, node_tile, 1)) / w
    dpos = ct.where(component, (ct.astype(offset, ct.float32) - fx) * dx, 0.0)

    velocity = ct.reshape(velocity, (particle_tile, 1, 4))
    C = ct.reshape(C, (particle_tile, 1, 4, 4))
    stress = ct.reshape(stress, (particle_tile, 1, 4, 4))
    affine = velocity + ct.sum(
        C * ct.reshape(dpos, (particle_tile, node_tile, 1, 4)),
        axis=3,
    )
    stress_dw = ct.sum(
        stress * ct.reshape(dweight, (particle_tile, node_tile, 1, 4)),
        axis=3,
    )

    mv = (
        -dt * vol * stress_dw
        + p_mass * ct.reshape(weight, (particle_tile, node_tile, 1)) * affine
    )
    mass = p_mass * weight
    return (
        ct.where(channel == 3, ct.reshape(mass, (particle_tile, node_tile, 1)), mv),
        contributes,
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


# ============================================================================
# arena scatter via one tile-coalesced atomic_store_add per super-cell
# ============================================================================
# Each block reduces one SC=2 super-cell into a 4**3-node L1-resident arena,
# then writes the arena back with one tile-coalesced atomic.
# A 1-node halo pad makes the arena's -1 apron origin align to an overlapping
# TiledView (tile=(SC+2)**3, traversal_steps=SC); overlapping apron nodes are
# reconciled by per-element atomics.
# Occupancy is left to the cuTile compiler default (ncu flags this kernel
# register-limited, so the best value is GPU-specific; no hint is forced here).
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
        pcols = _load_particle_columns(
            x, v, C, stress, p, active, inv_dx, particle_tile
        )
        active_col = ct.reshape(active, (particle_tile, 1))
        mv0, mv1, mv2, mass, contributes = _node_contribution_columns(
            pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx
        )
        m = contributes & active_col
        acc0 = acc0 + ct.sum(ct.where(m, mv0, 0.0), axis=0)
        acc1 = acc1 + ct.sum(ct.where(m, mv1, 0.0), axis=0)
        acc2 = acc2 + ct.sum(ct.where(m, mv2, 0.0), axis=0)
        accm = accm + ct.sum(ct.where(m, mass, 0.0), axis=0)
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
    gp = g + 2  # 1-node halo on each side
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
    # Strip the halo -> real (G**3, 3) / (G**3,) grids.
    grid = grid[1 : g + 1, 1 : g + 1, 1 : g + 1, :]
    grid_mv = grid[..., :3].reshape((g3, 3))
    grid_m = grid[..., 3].reshape((g3,))
    return grid_mv, grid_m


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
    particles = _load_particle_chunk(
        x,
        v,
        C,
        stress,
        chunk_start,
        n_particles,
        inv_dx,
        particle_tile,
    )

    active_lane = ct.reshape(in_bounds, (particle_tile, 1))
    offset = ct.reshape(ct.arange(stencil_tile, dtype=ct.int32), (1, stencil_tile))
    oi = offset // 9
    oj = (offset // 3) % 3
    ok = offset % 3
    oi_vec = ct.reshape(oi, (1, stencil_tile, 1))
    oj_vec = ct.reshape(oj, (1, stencil_tile, 1))
    ok_vec = ct.reshape(ok, (1, stencil_tile, 1))

    b = particles[0]
    base = ct.reshape(b, (particle_tile, 1, 4))
    node = _node_xyz(
        ct.extract(base, (0, 0, 0), (particle_tile, 1, 1)) + oi_vec,
        ct.extract(base, (0, 0, 1), (particle_tile, 1, 1)) + oj_vec,
        ct.extract(base, (0, 0, 2), (particle_tile, 1, 1)) + ok_vec,
        (particle_tile, stencil_tile, 4),
    )
    contrib, contributes = _node_contribution(
        particles,
        node,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
        particle_tile,
        stencil_tile,
    )

    valid_stencil = offset < 27
    mask = contributes & active_lane & valid_stencil
    flat = (
        ct.extract(node, (0, 0, 0), (particle_tile, stencil_tile, 1))
        * (G * G)
        + ct.extract(node, (0, 0, 1), (particle_tile, stencil_tile, 1)) * G
        + ct.extract(node, (0, 0, 2), (particle_tile, stencil_tile, 1))
    )
    flat = ct.reshape(flat, (particle_tile, stencil_tile))
    flat = ct.maximum(0, ct.minimum(flat, G * G * G - 1))

    channel = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 1, 4))
    flat = ct.reshape(flat, (particle_tile, stencil_tile, 1))
    mask = ct.reshape(mask, (particle_tile, stencil_tile, 1))
    ct.atomic_add(grid, (flat, channel), ct.where(mask, contrib, 0.0))


def cutile_p2g_v1(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Direct cuTile scatter for comparison with the arena backend."""
    n = int(x.shape[0])
    g = int(num_grids)
    g3 = g**3
    blocks = (n + DIRECT_PARTICLE_TILE - 1) // DIRECT_PARTICLE_TILE

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
