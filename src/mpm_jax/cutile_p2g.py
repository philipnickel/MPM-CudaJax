"""cuTile P2G kernels called from the shared JAX-owned frame loop.

Four variants, all sharing the JAX baseline G2P (only P2G varies). All but v1 use
genuine tile primitives (``gather`` loads, broadcast + ``where`` + ``sum``
reductions, ``scatter``/``atomic_add`` writes) — no raw ``get_raw_memory``.

- ``cutile_v1_atomic`` — deliberately naive: one lane per particle, 27 global
  atomics each. Kept as the "SIMT-in-cuTile" reference (uses the raw
  ``get_raw_memory`` escape hatch on purpose).
- ``cutile_v2_supercell_reduce`` — particles sorted by home super-cell; each
  block reduces its particles into the 6**3 apron-node tile with genuine tile
  ops and finishes with one ``atomic_add`` per node (apron nodes are shared).
- ``cutile_v3_gather`` — atomic-free. Each block *owns* one super-cell's 4**3
  interior nodes, *gathers* every contributing particle from the 6**3 covering
  home-cells (per-cell CSR), reduces in-tile, and writes each owned node exactly
  once with a plain ``scatter``. Inverts the scatter into a gather (no two blocks
  write the same node) at the cost of ~3.4x redundant particle reads.
- ``cutile_v4_arena`` — SPGrid-style colored arena scatter; the fastest variant
  (beats hand-written CUDA v4). Each block reduces its OWN super-cell's particles
  (read once, no redundancy) into a small L1-resident arena, then writes the arena
  back with a NON-atomic read-modify-write. 8 colored launches (parity of
  super-cell coords) make the write-back race-free without atomics. A small SC=2
  super-cell keeps the arena at 4**3 = 64 nodes (great occupancy, few node-evals).
"""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


P2G_TILE_SIZE = 256

# --- v2 (sorted-scatter + tile reduce) --------------------------------------
SUPER_CELL_WIDTH = 4
SUPER_TILE_DIM = SUPER_CELL_WIDTH + 2           # 6**3 apron-node tile
SUPER_TILE_NODES = SUPER_TILE_DIM ** 3          # 216
SUPERCELL_NODE_TILE = 64                        # node chunk (power of two)
SUPERCELL_NODE_CHUNKS = (SUPER_TILE_NODES + SUPERCELL_NODE_TILE - 1) // SUPERCELL_NODE_TILE
SUPERCELL_PARTICLE_TILE = 8                     # particles per chunk (occupancy sweet spot)

# --- v3 (atomic-free gather) ------------------------------------------------
V3_SUPER_CELL_WIDTH = 4
V3_OWNED_NODES = V3_SUPER_CELL_WIDTH ** 3       # 64 interior nodes owned per block
V3_COVER_DIM = V3_SUPER_CELL_WIDTH + 2          # 6 covering home-cells per axis
V3_COVER_COLUMNS = V3_COVER_DIM * V3_COVER_DIM  # 36 (ci,cj) columns; k merged
V3_PARTICLE_TILE = 16                          # particles per chunk (occupancy sweet spot)

# --- v4 (colored arena scatter, atomic-free, no read redundancy) ------------
# SPGrid-style: each block reduces its OWN super-cell's particles into a padded
# local arena (L1-resident), then writes the arena back to the global grid with a
# plain (non-atomic) read-modify-write. Race-freedom comes from 8-colour block
# grouping (parity of super-cell coords): the 8 super-cells that share any apron
# node have 8 distinct parities, so within one colour launch no two blocks ever
# touch the same node.
#
# Uses a SMALL super-cell (SC=2) so the arena is exactly 4**3 = 64 nodes -- a
# clean power-of-two tile with the same occupancy as v3, but each particle is read
# only ONCE (no 3.4x gather redundancy) and evaluated against 64 (not 216) nodes.
V4_ARENA_SC = 2                                 # arena super-cell width
V4_ARENA_DIM = V4_ARENA_SC + 2                  # 4 nodes per axis (SC + 1 apron each side)
V4_ARENA_NODES = V4_ARENA_DIM ** 3              # 64 arena nodes (power of two)
V4_ARENA_PARTICLE_TILE = 16                     # particles per chunk (occupancy sweet spot)


def is_available():
    """Return True when cuTile's Python package can be imported."""
    return True


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
        b0, b1, b2, fx0, fx1, fx2, vp0, vp1, vp2,
        C00, C01, C02, C10, C11, C12, C20, C21, C22,
        s00, s01, s02, s10, s11, s12, s20, s21, s22,
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
    baseline so v2/v3 match it to fp32 round-off.
    """
    (b0, b1, b2, fx0, fx1, fx2, vp0, vp1, vp2,
     C00, C01, C02, C10, C11, C12, C20, C21, C22,
     s00, s01, s02, s10, s11, s12, s20, s21, s22) = pcols
    gi_row, gj_row, gk_row = node_rows

    ox = gi_row - b0
    oy = gj_row - b1
    oz = gk_row - b2
    contributes = (
        (ox >= 0) & (ox < 3) & (oy >= 0) & (oy < 3) & (oz >= 0) & (oz < 3)
    )
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
    (b0, b1, b2, fx0, fx1, fx2, vp0, vp1, vp2,
     C00, C01, C02, C10, C11, C12, C20, C21, C22,
     s00, s01, s02, s10, s11, s12, s20, s21, s22) = particle
    return (
        ct.reshape(b0, (tile, 1)), ct.reshape(b1, (tile, 1)), ct.reshape(b2, (tile, 1)),
        ct.reshape(fx0, (tile, 1)), ct.reshape(fx1, (tile, 1)), ct.reshape(fx2, (tile, 1)),
        ct.reshape(vp0, (tile, 1)), ct.reshape(vp1, (tile, 1)), ct.reshape(vp2, (tile, 1)),
        ct.reshape(C00, (tile, 1)), ct.reshape(C01, (tile, 1)), ct.reshape(C02, (tile, 1)),
        ct.reshape(C10, (tile, 1)), ct.reshape(C11, (tile, 1)), ct.reshape(C12, (tile, 1)),
        ct.reshape(C20, (tile, 1)), ct.reshape(C21, (tile, 1)), ct.reshape(C22, (tile, 1)),
        ct.reshape(s00, (tile, 1)), ct.reshape(s01, (tile, 1)), ct.reshape(s02, (tile, 1)),
        ct.reshape(s10, (tile, 1)), ct.reshape(s11, (tile, 1)), ct.reshape(s12, (tile, 1)),
        ct.reshape(s20, (tile, 1)), ct.reshape(s21, (tile, 1)), ct.reshape(s22, (tile, 1)),
    )


# ============================================================================
# v1 — naive particle-owned global atomics (raw escape hatch, kept as baseline)
# ============================================================================
@ct.kernel
def _p2g_atomic_kernel(
    x, v, C, stress, grid_mv, grid_m,
    N: ct.Constant[int], G: ct.Constant[int],
    dt: ct.Constant[float], vol: ct.Constant[float], p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float], dx: ct.Constant[float], tile_size: ct.Constant[int],
):
    bid = ct.bid(0)
    lane = ct.arange(tile_size, dtype=ct.int32)
    p = bid * tile_size + lane
    active = p < N

    x_mem = x.get_raw_memory()
    v_mem = v.get_raw_memory()
    C_mem = C.get_raw_memory()
    stress_mem = stress.get_raw_memory()
    grid_mv_mem = grid_mv.get_raw_memory()
    grid_m_mem = grid_m.get_raw_memory()

    px0 = x_mem.load_offset(p * 3 + 0, mask=active) * inv_dx
    px1 = x_mem.load_offset(p * 3 + 1, mask=active) * inv_dx
    px2 = x_mem.load_offset(p * 3 + 2, mask=active) * inv_dx

    b0 = ct.astype(ct.floor(px0 - 0.5), ct.int32)
    b1 = ct.astype(ct.floor(px1 - 0.5), ct.int32)
    b2 = ct.astype(ct.floor(px2 - 0.5), ct.int32)

    fx0 = px0 - ct.astype(b0, ct.float32)
    fx1 = px1 - ct.astype(b1, ct.float32)
    fx2 = px2 - ct.astype(b2, ct.float32)

    vp0 = v_mem.load_offset(p * 3 + 0, mask=active)
    vp1 = v_mem.load_offset(p * 3 + 1, mask=active)
    vp2 = v_mem.load_offset(p * 3 + 2, mask=active)

    C00 = C_mem.load_offset(p * 9 + 0, mask=active)
    C01 = C_mem.load_offset(p * 9 + 1, mask=active)
    C02 = C_mem.load_offset(p * 9 + 2, mask=active)
    C10 = C_mem.load_offset(p * 9 + 3, mask=active)
    C11 = C_mem.load_offset(p * 9 + 4, mask=active)
    C12 = C_mem.load_offset(p * 9 + 5, mask=active)
    C20 = C_mem.load_offset(p * 9 + 6, mask=active)
    C21 = C_mem.load_offset(p * 9 + 7, mask=active)
    C22 = C_mem.load_offset(p * 9 + 8, mask=active)

    s00 = stress_mem.load_offset(p * 9 + 0, mask=active)
    s01 = stress_mem.load_offset(p * 9 + 1, mask=active)
    s02 = stress_mem.load_offset(p * 9 + 2, mask=active)
    s10 = stress_mem.load_offset(p * 9 + 3, mask=active)
    s11 = stress_mem.load_offset(p * 9 + 4, mask=active)
    s12 = stress_mem.load_offset(p * 9 + 5, mask=active)
    s20 = stress_mem.load_offset(p * 9 + 6, mask=active)
    s21 = stress_mem.load_offset(p * 9 + 7, mask=active)
    s22 = stress_mem.load_offset(p * 9 + 8, mask=active)

    for ox in ct.static_iter(range(3)):
        if ox == 0:
            wx = 0.5 * (1.5 - fx0) * (1.5 - fx0)
            dwx = fx0 - 1.5
        elif ox == 1:
            wx = 0.75 - (fx0 - 1.0) * (fx0 - 1.0)
            dwx = -2.0 * (fx0 - 1.0)
        else:
            wx = 0.5 * (fx0 - 0.5) * (fx0 - 0.5)
            dwx = fx0 - 0.5

        for oy in ct.static_iter(range(3)):
            if oy == 0:
                wy = 0.5 * (1.5 - fx1) * (1.5 - fx1)
                dwy = fx1 - 1.5
            elif oy == 1:
                wy = 0.75 - (fx1 - 1.0) * (fx1 - 1.0)
                dwy = -2.0 * (fx1 - 1.0)
            else:
                wy = 0.5 * (fx1 - 0.5) * (fx1 - 0.5)
                dwy = fx1 - 0.5

            for oz in ct.static_iter(range(3)):
                if oz == 0:
                    wz = 0.5 * (1.5 - fx2) * (1.5 - fx2)
                    dwz = fx2 - 1.5
                elif oz == 1:
                    wz = 0.75 - (fx2 - 1.0) * (fx2 - 1.0)
                    dwz = -2.0 * (fx2 - 1.0)
                else:
                    wz = 0.5 * (fx2 - 0.5) * (fx2 - 0.5)
                    dwz = fx2 - 0.5

                weight = wx * wy * wz
                dw0 = inv_dx * dwx * wy * wz
                dw1 = inv_dx * wx * dwy * wz
                dw2 = inv_dx * wx * wy * dwz

                dpos0 = (float(ox) - fx0) * dx
                dpos1 = (float(oy) - fx1) * dx
                dpos2 = (float(oz) - fx2) * dx

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

                gi = ct.minimum(ct.maximum(b0 + ox, 0), G - 1)
                gj = ct.minimum(ct.maximum(b1 + oy, 0), G - 1)
                gk = ct.minimum(ct.maximum(b2 + oz, 0), G - 1)
                grid_idx = gi * G * G + gj * G + gk

                grid_mv_mem.atomic_add_offset(grid_idx * 3 + 0, mv0, mask=active)
                grid_mv_mem.atomic_add_offset(grid_idx * 3 + 1, mv1, mask=active)
                grid_mv_mem.atomic_add_offset(grid_idx * 3 + 2, mv2, mask=active)
                grid_m_mem.atomic_add_offset(grid_idx, mass, mask=active)


def cutile_p2g_atomic(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Particle-owned atomic P2G implemented as cuTile JAX FFI calls."""
    n = x.shape[0]
    g = int(num_grids)
    g3 = g ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = cutile_call(
        ((n + P2G_TILE_SIZE - 1) // P2G_TILE_SIZE,),
        _p2g_atomic_kernel,
        (
            x, v, C_flat, stress_flat,
            InputOutput(grid_mv_flat), InputOutput(grid_m),
            int(n), g, float(dt), float(vol), float(p_mass),
            float(inv_dx), float(dx), P2G_TILE_SIZE,
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


# ============================================================================
# v2 — sorted super-cell, genuine tile reduction, one atomic_add per node
# ============================================================================
@ct.kernel
def _p2g_supercell_reduce_kernel(
    x, v, C, stress, cell_start, grid_mv, grid_m,
    G: ct.Constant[int],
    dt: ct.Constant[float], vol: ct.Constant[float], p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float], dx: ct.Constant[float],
    particle_tile: ct.Constant[int], node_tile: ct.Constant[int],
):
    super_id = ct.bid(0)
    Gs = G // SUPER_CELL_WIDTH
    si = super_id // (Gs * Gs)
    sj = (super_id // Gs) % Gs
    sk = super_id % Gs
    tile_i = si * SUPER_CELL_WIDTH - 1
    tile_j = sj * SUPER_CELL_WIDTH - 1
    tile_k = sk * SUPER_CELL_WIDTH - 1

    p_start = ct.gather(cell_start, super_id)
    p_end = ct.gather(cell_start, super_id + 1)
    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    # Node-chunk outer: own the 6**3 apron in chunks of `node_tile`, reduce this
    # super-cell's particles into each chunk, then one atomic_add per node (apron
    # nodes are shared with neighbouring super-cells, so the write must be atomic).
    for nc in ct.static_iter(range(SUPERCELL_NODE_CHUNKS)):
        node_lane = ct.arange(node_tile, dtype=ct.int32)
        local_node = nc * node_tile + node_lane
        node_active = local_node < SUPER_TILE_NODES
        ti = local_node // (SUPER_TILE_DIM * SUPER_TILE_DIM)
        tj = (local_node // SUPER_TILE_DIM) % SUPER_TILE_DIM
        tk = local_node % SUPER_TILE_DIM
        gi_row = ct.reshape(tile_i + ti, (1, node_tile))
        gj_row = ct.reshape(tile_j + tj, (1, node_tile))
        gk_row = ct.reshape(tile_k + tk, (1, node_tile))
        node_mask = ct.reshape(node_active, (1, node_tile))

        acc0 = ct.zeros((node_tile,), ct.float32)
        acc1 = ct.zeros((node_tile,), ct.float32)
        acc2 = ct.zeros((node_tile,), ct.float32)
        accm = ct.zeros((node_tile,), ct.float32)

        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end
            pcols = _cols(
                _gather_particle(x, v, C, stress, p, active, inv_dx), particle_tile)
            active_col = ct.reshape(active, (particle_tile, 1))
            mv0, mv1, mv2, mass, contributes = _node_contribution(
                pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx)
            m = contributes & active_col & node_mask
            acc0 = acc0 + ct.sum(ct.where(m, mv0, 0.0), axis=0)
            acc1 = acc1 + ct.sum(ct.where(m, mv1, 0.0), axis=0)
            acc2 = acc2 + ct.sum(ct.where(m, mv2, 0.0), axis=0)
            accm = accm + ct.sum(ct.where(m, mass, 0.0), axis=0)
            chunk_start += particle_tile

        gi = ct.minimum(ct.maximum(tile_i + ti, 0), G - 1)
        gj = ct.minimum(ct.maximum(tile_j + tj, 0), G - 1)
        gk = ct.minimum(ct.maximum(tile_k + tk, 0), G - 1)
        in_grid = (
            node_active
            & (tile_i + ti >= 0) & (tile_i + ti < G)
            & (tile_j + tj >= 0) & (tile_j + tj < G)
            & (tile_k + tk >= 0) & (tile_k + tk < G)
        )
        grid_idx = gi * G * G + gj * G + gk
        safe = ct.where(in_grid, grid_idx, 0)
        z0 = ct.where(in_grid, acc0, 0.0)
        z1 = ct.where(in_grid, acc1, 0.0)
        z2 = ct.where(in_grid, acc2, 0.0)
        zm = ct.where(in_grid, accm, 0.0)
        ct.atomic_add(grid_mv, (safe * 3 + 0,), z0, memory_scope=ct.MemoryScope.DEVICE)
        ct.atomic_add(grid_mv, (safe * 3 + 1,), z1, memory_scope=ct.MemoryScope.DEVICE)
        ct.atomic_add(grid_mv, (safe * 3 + 2,), z2, memory_scope=ct.MemoryScope.DEVICE)
        ct.atomic_add(grid_m, (safe,), zm, memory_scope=ct.MemoryScope.DEVICE)


def cutile_p2g_supercell_reduce(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Super-cell-local cuTile reduction (real tile ops) before atomic scatter."""
    g = int(num_grids)
    g3 = g ** 3
    x_flat = x.reshape(-1)
    v_flat = v.reshape(-1)
    C_flat = C.reshape(-1)
    stress_flat = stress.reshape(-1)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = cutile_call(
        ((g // SUPER_CELL_WIDTH) ** 3,),
        _p2g_supercell_reduce_kernel,
        (
            x_flat, v_flat, C_flat, stress_flat, cell_start,
            InputOutput(grid_mv_flat), InputOutput(grid_m),
            g, float(dt), float(vol), float(p_mass), float(inv_dx), float(dx),
            SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE,
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


# ============================================================================
# v3 — atomic-free grid-tile gather (the headline)
# ============================================================================
@ct.kernel
def _p2g_gather_kernel(
    x, v, C, stress, cell_start, grid_mv, grid_m,
    G: ct.Constant[int],
    dt: ct.Constant[float], vol: ct.Constant[float], p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float], dx: ct.Constant[float],
    particle_tile: ct.Constant[int],
):
    super_id = ct.bid(0)
    SC = V3_SUPER_CELL_WIDTH
    Gs = G // SC
    si = super_id // (Gs * Gs)
    sj = (super_id // Gs) % Gs
    sk = super_id % Gs

    # Owned interior nodes: the SC**3 block of grid nodes [SC*si .. SC*si+SC-1]^3.
    # These tile the whole grid disjointly, so each is written by exactly one block.
    node_lane = ct.arange(V3_OWNED_NODES, dtype=ct.int32)
    na = node_lane // (SC * SC)
    nb = (node_lane // SC) % SC
    nc = node_lane % SC
    gi = SC * si + na
    gj = SC * sj + nb
    gk = SC * sk + nc
    gi_row = ct.reshape(gi, (1, V3_OWNED_NODES))
    gj_row = ct.reshape(gj, (1, V3_OWNED_NODES))
    gk_row = ct.reshape(gk, (1, V3_OWNED_NODES))

    acc0 = ct.zeros((V3_OWNED_NODES,), ct.float32)
    acc1 = ct.zeros((V3_OWNED_NODES,), ct.float32)
    acc2 = ct.zeros((V3_OWNED_NODES,), ct.float32)
    accm = ct.zeros((V3_OWNED_NODES,), ct.float32)

    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    # Gather contributing particles from the 6**3 covering home-cells
    # home in [SC*si-1 .. SC*si+SC]^3. The 6 home-cells along k are contiguous in
    # row-major flat id, so each (ci,cj) column is ONE particle range
    # [cell_start[home0], cell_start[home0 + 6]) -- 36 ranges of ~48 particles
    # instead of 216 tiny per-cell ranges (far better lane packing).
    #
    # The counter is a runtime tile scalar (``super_id * 0`` == 0) so the body is
    # a runtime loop, NOT a static unroll. No validity branch is needed: an OOB
    # column makes the gather return an empty [a, a) range, and any cell reached by
    # integer wrap-around (or a clamped-home edge cell) is >= G-2 cells from the
    # owned nodes, so its particles fail the ``contributes`` stencil mask.
    col = super_id * 0
    while col < V3_COVER_COLUMNS:
        ci = col // V3_COVER_DIM
        cj = col % V3_COVER_DIM
        hi = SC * si - 1 + ci
        hj = SC * sj - 1 + cj
        hk0 = SC * sk - 1
        home0 = hi * G * G + hj * G + hk0
        p_start = ct.gather(cell_start, home0)
        p_end = ct.gather(cell_start, home0 + V3_COVER_DIM)
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end
            pcols = _cols(
                _gather_particle(x, v, C, stress, p, active, inv_dx), particle_tile)
            active_col = ct.reshape(active, (particle_tile, 1))
            mv0, mv1, mv2, mass, contributes = _node_contribution(
                pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx)
            m = contributes & active_col
            acc0 = acc0 + ct.sum(ct.where(m, mv0, 0.0), axis=0)
            acc1 = acc1 + ct.sum(ct.where(m, mv1, 0.0), axis=0)
            acc2 = acc2 + ct.sum(ct.where(m, mv2, 0.0), axis=0)
            accm = accm + ct.sum(ct.where(m, mass, 0.0), axis=0)
            chunk_start += particle_tile
        col += 1

    # Each owned node written exactly once -> plain scatter, no atomics.
    grid_idx = gi * G * G + gj * G + gk
    ct.scatter(grid_mv, (grid_idx * 3 + 0,), acc0)
    ct.scatter(grid_mv, (grid_idx * 3 + 1,), acc1)
    ct.scatter(grid_mv, (grid_idx * 3 + 2,), acc2)
    ct.scatter(grid_m, (grid_idx,), accm)


def cutile_p2g_gather(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Atomic-free gather P2G: each block owns a super-cell's interior nodes."""
    g = int(num_grids)
    g3 = g ** 3
    x_flat = x.reshape(-1)
    v_flat = v.reshape(-1)
    C_flat = C.reshape(-1)
    stress_flat = stress.reshape(-1)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = cutile_call(
        ((g // V3_SUPER_CELL_WIDTH) ** 3,),
        _p2g_gather_kernel,
        (
            x_flat, v_flat, C_flat, stress_flat, cell_start,
            InputOutput(grid_mv_flat), InputOutput(grid_m),
            g, float(dt), float(vol), float(p_mass), float(inv_dx), float(dx),
            V3_PARTICLE_TILE,
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


# ============================================================================
# v4 — SPGrid-style colored arena scatter (atomic-free, single-pass particles)
# ============================================================================
@ct.kernel
def _p2g_arena_kernel(
    x, v, C, stress, cell_start, grid_mv, grid_m,
    G: ct.Constant[int],
    dt: ct.Constant[float], vol: ct.Constant[float], p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float], dx: ct.Constant[float],
    ci, cj, ck,                       # runtime colour offsets (0/1) per axis
    particle_tile: ct.Constant[int], node_tile: ct.Constant[int],
):
    # 3D launch over one colour's super-cells: super-cell = 2*block + colour.
    SC = V4_ARENA_SC
    DIM = V4_ARENA_DIM
    Gs = G // SC
    si = 2 * ct.bid(0) + ci
    sj = 2 * ct.bid(1) + cj
    sk = 2 * ct.bid(2) + ck
    super_id = si * (Gs * Gs) + sj * Gs + sk
    tile_i = si * SC - 1
    tile_j = sj * SC - 1
    tile_k = sk * SC - 1

    p_start = ct.gather(cell_start, super_id)
    p_end = ct.gather(cell_start, super_id + 1)
    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    # DIM**3 arena node tile (= 64 for SC=2; a clean power of two, no padding).
    node_lane = ct.arange(node_tile, dtype=ct.int32)
    ti = node_lane // (DIM * DIM)
    tj = (node_lane // DIM) % DIM
    tk = node_lane % DIM
    gi = tile_i + ti
    gj = tile_j + tj
    gk = tile_k + tk
    gi_row = ct.reshape(gi, (1, node_tile))
    gj_row = ct.reshape(gj, (1, node_tile))
    gk_row = ct.reshape(gk, (1, node_tile))

    acc0 = ct.zeros((node_tile,), ct.float32)
    acc1 = ct.zeros((node_tile,), ct.float32)
    acc2 = ct.zeros((node_tile,), ct.float32)
    accm = ct.zeros((node_tile,), ct.float32)

    # Reduce this super-cell's OWN particles (one contiguous run) into the arena.
    chunk_start = p_start
    while chunk_start < p_end:
        p = chunk_start + p_lane
        active = p < p_end
        pcols = _cols(
            _gather_particle(x, v, C, stress, p, active, inv_dx), particle_tile)
        active_col = ct.reshape(active, (particle_tile, 1))
        mv0, mv1, mv2, mass, contributes = _node_contribution(
            pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx)
        m = contributes & active_col
        acc0 = acc0 + ct.sum(ct.where(m, mv0, 0.0), axis=0)
        acc1 = acc1 + ct.sum(ct.where(m, mv1, 0.0), axis=0)
        acc2 = acc2 + ct.sum(ct.where(m, mv2, 0.0), axis=0)
        accm = accm + ct.sum(ct.where(m, mass, 0.0), axis=0)
        chunk_start += particle_tile

    # Write the arena back with a NON-atomic read-modify-write. Colouring makes
    # this race-free: within this launch no other block touches these nodes, and
    # the 8 sequential colour launches accumulate shared apron nodes correctly.
    in_grid = (
        (gi >= 0) & (gi < G) & (gj >= 0) & (gj < G) & (gk >= 0) & (gk < G)
    )
    grid_idx = gi * G * G + gj * G + gk
    safe = ct.where(in_grid, grid_idx, 0)
    cur0 = ct.gather(grid_mv, safe * 3 + 0, mask=in_grid, padding_value=0.0)
    cur1 = ct.gather(grid_mv, safe * 3 + 1, mask=in_grid, padding_value=0.0)
    cur2 = ct.gather(grid_mv, safe * 3 + 2, mask=in_grid, padding_value=0.0)
    curm = ct.gather(grid_m, safe, mask=in_grid, padding_value=0.0)
    ct.scatter(grid_mv, (safe * 3 + 0,), cur0 + acc0, mask=in_grid)
    ct.scatter(grid_mv, (safe * 3 + 1,), cur1 + acc1, mask=in_grid)
    ct.scatter(grid_mv, (safe * 3 + 2,), cur2 + acc2, mask=in_grid)
    ct.scatter(grid_m, (safe,), curm + accm, mask=in_grid)


def cutile_p2g_arena(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Colored arena-scatter P2G: 8 race-free atomic-free launches, no redundancy."""
    g = int(num_grids)
    g3 = g ** 3
    x_flat = x.reshape(-1)
    v_flat = v.reshape(-1)
    C_flat = C.reshape(-1)
    stress_flat = stress.reshape(-1)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)
    gs = g // V4_ARENA_SC

    for ci in range(2):
        for cj in range(2):
            for ck in range(2):
                nb_i = (gs - ci + 1) // 2
                nb_j = (gs - cj + 1) // 2
                nb_k = (gs - ck + 1) // 2
                grid_mv_flat, grid_m = cutile_call(
                    (nb_i, nb_j, nb_k),
                    _p2g_arena_kernel,
                    (
                        x_flat, v_flat, C_flat, stress_flat, cell_start,
                        InputOutput(grid_mv_flat), InputOutput(grid_m),
                        g, float(dt), float(vol), float(p_mass),
                        float(inv_dx), float(dx),
                        ci, cj, ck,
                        V4_ARENA_PARTICLE_TILE, V4_ARENA_NODES,
                    ),
                )
    return grid_mv_flat.reshape((g3, 3)), grid_m


# ============================================================================
# v6 — arena scatter via a single tile-coalesced atomic_store_add (no coloring)
# ============================================================================
# Same per-block arena reduction as v4, but the write-back is ONE launch with one
# tile-coalesced atomic per arena instead of 8 colored non-atomic RMW launches.
# A 1-node halo pad makes the arena's -1 apron origin align to an overlapping
# TiledView (tile=(SC+2)**3, traversal_steps=SC), so each block does
# view.atomic_store_add(super_cell_index, arena) and overlapping apron nodes are
# reconciled by the per-element atomics. No coloring, no parity, no 8 launches.
# Occupancy is left to the compiler here; the best value is GPU-specific (ncu flags
# this kernel register-limited), so it is chosen per-machine by cutile_autotune
# (kernel.replace_hints(occupancy=...)) rather than hardcoded for one architecture.
@ct.kernel
def _p2g_atomic_tile_kernel(
    x, v, C, stress, cell_start, grid_mv, grid_m,
    G: ct.Constant[int],
    dt: ct.Constant[float], vol: ct.Constant[float], p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float], dx: ct.Constant[float],
    particle_tile: ct.Constant[int], node_tile: ct.Constant[int],
):
    SC = V4_ARENA_SC
    DIM = V4_ARENA_DIM
    Gs = G // SC
    si = ct.bid(0)
    sj = ct.bid(1)
    sk = ct.bid(2)
    super_id = si * (Gs * Gs) + sj * Gs + sk
    tile_i = si * SC - 1            # real-grid arena origin (apron at -1)
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
            _gather_particle(x, v, C, stress, p, active, inv_dx), particle_tile)
        active_col = ct.reshape(active, (particle_tile, 1))
        mv0, mv1, mv2, mass, contributes = _node_contribution(
            pcols, (gi_row, gj_row, gk_row), dt, vol, p_mass, inv_dx, dx)
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
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx, kernel=None
):
    """One-launch arena P2G: a single tile-coalesced atomic_store_add per block,
    no coloring. A 1-node halo pad aligns the apron to an overlapping tiled view.

    ``kernel`` lets the autotuner pass an occupancy-tuned variant of
    ``_p2g_atomic_tile_kernel``; defaults to the module kernel.
    """
    g = int(num_grids)
    g3 = g ** 3
    gp = g + 2                       # 1-node halo on each side
    gs = g // V4_ARENA_SC
    x_flat = x.reshape(-1)
    v_flat = v.reshape(-1)
    C_flat = C.reshape(-1)
    stress_flat = stress.reshape(-1)

    grid_mv = jnp.zeros((gp, gp, gp, 3), dtype=jnp.float32)
    grid_m = jnp.zeros((gp, gp, gp), dtype=jnp.float32)

    grid_mv, grid_m = cutile_call(
        (gs, gs, gs),
        kernel if kernel is not None else _p2g_atomic_tile_kernel,
        (
            x_flat, v_flat, C_flat, stress_flat, cell_start,
            InputOutput(grid_mv), InputOutput(grid_m),
            g, float(dt), float(vol), float(p_mass), float(inv_dx), float(dx),
            V4_ARENA_PARTICLE_TILE, V4_ARENA_NODES,
        ),
    )
    # Strip the halo -> real (G**3, 3) / (G**3,) grids.
    grid_mv = grid_mv[1:g + 1, 1:g + 1, 1:g + 1, :].reshape((g3, 3))
    grid_m = grid_m[1:g + 1, 1:g + 1, 1:g + 1].reshape((g3,))
    return grid_mv, grid_m
