"""cuTile P2G kernels called from the shared JAX-owned frame loop."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


P2G_TILE_SIZE = 256
SUPER_CELL_WIDTH = 4
SUPER_TILE_PHYSICAL_DIM = SUPER_CELL_WIDTH + 2
SUPER_TILE_DIM = SUPER_TILE_PHYSICAL_DIM
SUPER_TILE_NODES = SUPER_TILE_DIM * SUPER_TILE_DIM * SUPER_TILE_DIM
SC4_TILED_VIEW_TILE_DIM = 8
SC4_TILED_VIEW_TILE_NODES = (
    SC4_TILED_VIEW_TILE_DIM * SC4_TILED_VIEW_TILE_DIM * SC4_TILED_VIEW_TILE_DIM
)
SC4_COLOR_COUNT = 8
GRID_CHANNELS = 4
SUPERCELL_PARTICLE_TILE = 4
SUPERCELL_NODE_TILE = 256
PARTICLE_VECTOR_TILE = 4
PARTICLE_MATRIX_TILE = 16
NATIVE_SUPER_CELL_WIDTH = 2
NATIVE_NODE_TILE_DIM = NATIVE_SUPER_CELL_WIDTH + 2
NATIVE_PARTICLE_TILE = 4


@ct.kernel
def _p2g_atomic_kernel(
    x,
    v,
    C,
    stress,
    grid_mv,
    grid_m,
    N: ct.Constant[int],
    G: ct.Constant[int],
    dt: ct.Constant[float],
    vol: ct.Constant[float],
    p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float],
    dx: ct.Constant[float],
    tile_size: ct.Constant[int],
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


def is_available():
    """Return True when cuTile's Python package can be imported."""
    return True


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
            x,
            v,
            C_flat,
            stress_flat,
            InputOutput(grid_mv_flat),
            InputOutput(grid_m),
            int(n),
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            P2G_TILE_SIZE,
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


@ct.kernel(occupancy=2, num_worker_warps=4)
def _p2g_supercell_reduce_kernel(
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
    tile_loads: ct.Constant[bool],
):
    super_id = ct.bid(0)
    x_mem = x.get_raw_memory()
    v_mem = v.get_raw_memory()
    C_mem = C.get_raw_memory()
    stress_mem = stress.get_raw_memory()
    cell_start_mem = cell_start.get_raw_memory()
    grid_mv_mem = grid_mv.get_raw_memory()
    grid_m_mem = grid_m.get_raw_memory()

    Gs = G // SUPER_CELL_WIDTH
    si = super_id // (Gs * Gs)
    sj = (super_id // Gs) % Gs
    sk = super_id % Gs

    tile_i = si * SUPER_CELL_WIDTH - 1
    tile_j = sj * SUPER_CELL_WIDTH - 1
    tile_k = sk * SUPER_CELL_WIDTH - 1

    p_start = cell_start_mem.load_offset(super_id)
    p_end = cell_start_mem.load_offset(super_id + 1)
    if p_start == p_end:
        return
    interior_tile = (
        tile_i >= 0
        and tile_j >= 0
        and tile_k >= 0
        and tile_i + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_j + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_k + SUPER_TILE_PHYSICAL_DIM <= G
    )

    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    if interior_tile:
        n_node_chunks = (SUPER_TILE_NODES + SUPERCELL_NODE_TILE - 1) // SUPERCELL_NODE_TILE
        for node_chunk in ct.static_iter(range(n_node_chunks)):
            node_lane = ct.arange(node_tile, dtype=ct.int32)
            local_node = node_chunk * node_tile + node_lane
            node_active = local_node < SUPER_TILE_NODES

            ti = local_node // (SUPER_TILE_DIM * SUPER_TILE_DIM)
            tj = (local_node // SUPER_TILE_DIM) % SUPER_TILE_DIM
            tk = local_node % SUPER_TILE_DIM
            gi = tile_i + ti
            gj = tile_j + tj
            gk = tile_k + tk
            in_physical_tile = (
                node_active
                & (ti < SUPER_TILE_PHYSICAL_DIM)
                & (tj < SUPER_TILE_PHYSICAL_DIM)
                & (tk < SUPER_TILE_PHYSICAL_DIM)
            )

            gi_row = ct.reshape(gi, (1, SUPERCELL_NODE_TILE))
            gj_row = ct.reshape(gj, (1, SUPERCELL_NODE_TILE))
            gk_row = ct.reshape(gk, (1, SUPERCELL_NODE_TILE))
            node_mask = ct.reshape(in_physical_tile, (1, SUPERCELL_NODE_TILE))

            acc = ct.zeros((SUPERCELL_NODE_TILE, 4), ct.float32)

            chunk_start = p_start
            while chunk_start < p_end:
                p = chunk_start + p_lane
                active = p < p_end

                if tile_loads:
                    load_p = ct.minimum(p, p_end - 1)
                    x_tile = ct.load_advanced_indexing(
                        x,
                        (load_p, ct.Slice(0, PARTICLE_VECTOR_TILE)),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    v_tile = ct.load_advanced_indexing(
                        v,
                        (load_p, ct.Slice(0, PARTICLE_VECTOR_TILE)),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    C_tile = ct.load_advanced_indexing(
                        C,
                        (load_p, ct.Slice(0, PARTICLE_MATRIX_TILE)),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    stress_tile = ct.load_advanced_indexing(
                        stress,
                        (load_p, ct.Slice(0, PARTICLE_MATRIX_TILE)),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                    px0 = ct.reshape(
                        ct.extract(x_tile, (0, 0), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    ) * inv_dx
                    px1 = ct.reshape(
                        ct.extract(x_tile, (0, 1), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    ) * inv_dx
                    px2 = ct.reshape(
                        ct.extract(x_tile, (0, 2), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    ) * inv_dx
                else:
                    px0 = x_mem.load_offset(p * 3 + 0, mask=active) * inv_dx
                    px1 = x_mem.load_offset(p * 3 + 1, mask=active) * inv_dx
                    px2 = x_mem.load_offset(p * 3 + 2, mask=active) * inv_dx

                b0 = ct.astype(ct.floor(px0 - 0.5), ct.int32)
                b1 = ct.astype(ct.floor(px1 - 0.5), ct.int32)
                b2 = ct.astype(ct.floor(px2 - 0.5), ct.int32)

                fx0 = px0 - ct.astype(b0, ct.float32)
                fx1 = px1 - ct.astype(b1, ct.float32)
                fx2 = px2 - ct.astype(b2, ct.float32)

                if tile_loads:
                    vp0 = ct.reshape(
                        ct.extract(v_tile, (0, 0), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    )
                    vp1 = ct.reshape(
                        ct.extract(v_tile, (0, 1), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    )
                    vp2 = ct.reshape(
                        ct.extract(v_tile, (0, 2), shape=(SUPERCELL_PARTICLE_TILE, 1)),
                        (SUPERCELL_PARTICLE_TILE,),
                    )

                    C00 = ct.reshape(ct.extract(C_tile, (0, 0), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C01 = ct.reshape(ct.extract(C_tile, (0, 1), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C02 = ct.reshape(ct.extract(C_tile, (0, 2), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C10 = ct.reshape(ct.extract(C_tile, (0, 3), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C11 = ct.reshape(ct.extract(C_tile, (0, 4), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C12 = ct.reshape(ct.extract(C_tile, (0, 5), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C20 = ct.reshape(ct.extract(C_tile, (0, 6), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C21 = ct.reshape(ct.extract(C_tile, (0, 7), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    C22 = ct.reshape(ct.extract(C_tile, (0, 8), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))

                    s00 = ct.reshape(ct.extract(stress_tile, (0, 0), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s01 = ct.reshape(ct.extract(stress_tile, (0, 1), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s02 = ct.reshape(ct.extract(stress_tile, (0, 2), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s10 = ct.reshape(ct.extract(stress_tile, (0, 3), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s11 = ct.reshape(ct.extract(stress_tile, (0, 4), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s12 = ct.reshape(ct.extract(stress_tile, (0, 5), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s20 = ct.reshape(ct.extract(stress_tile, (0, 6), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s21 = ct.reshape(ct.extract(stress_tile, (0, 7), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                    s22 = ct.reshape(ct.extract(stress_tile, (0, 8), shape=(SUPERCELL_PARTICLE_TILE, 1)), (SUPERCELL_PARTICLE_TILE,))
                else:
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

                b0_col = ct.reshape(b0, (SUPERCELL_PARTICLE_TILE, 1))
                b1_col = ct.reshape(b1, (SUPERCELL_PARTICLE_TILE, 1))
                b2_col = ct.reshape(b2, (SUPERCELL_PARTICLE_TILE, 1))
                active_col = ct.reshape(active, (SUPERCELL_PARTICLE_TILE, 1))

                ox = gi_row - b0_col
                oy = gj_row - b1_col
                oz = gk_row - b2_col
                contributes = (
                    active_col
                    & node_mask
                    & (ox >= 0) & (ox < 3)
                    & (oy >= 0) & (oy < 3)
                    & (oz >= 0) & (oz < 3)
                )

                fx0_col = ct.reshape(fx0, (SUPERCELL_PARTICLE_TILE, 1))
                fx1_col = ct.reshape(fx1, (SUPERCELL_PARTICLE_TILE, 1))
                fx2_col = ct.reshape(fx2, (SUPERCELL_PARTICLE_TILE, 1))
                vp0_col = ct.reshape(vp0, (SUPERCELL_PARTICLE_TILE, 1))
                vp1_col = ct.reshape(vp1, (SUPERCELL_PARTICLE_TILE, 1))
                vp2_col = ct.reshape(vp2, (SUPERCELL_PARTICLE_TILE, 1))

                C00_col = ct.reshape(C00, (SUPERCELL_PARTICLE_TILE, 1))
                C01_col = ct.reshape(C01, (SUPERCELL_PARTICLE_TILE, 1))
                C02_col = ct.reshape(C02, (SUPERCELL_PARTICLE_TILE, 1))
                C10_col = ct.reshape(C10, (SUPERCELL_PARTICLE_TILE, 1))
                C11_col = ct.reshape(C11, (SUPERCELL_PARTICLE_TILE, 1))
                C12_col = ct.reshape(C12, (SUPERCELL_PARTICLE_TILE, 1))
                C20_col = ct.reshape(C20, (SUPERCELL_PARTICLE_TILE, 1))
                C21_col = ct.reshape(C21, (SUPERCELL_PARTICLE_TILE, 1))
                C22_col = ct.reshape(C22, (SUPERCELL_PARTICLE_TILE, 1))

                s00_col = ct.reshape(s00, (SUPERCELL_PARTICLE_TILE, 1))
                s01_col = ct.reshape(s01, (SUPERCELL_PARTICLE_TILE, 1))
                s02_col = ct.reshape(s02, (SUPERCELL_PARTICLE_TILE, 1))
                s10_col = ct.reshape(s10, (SUPERCELL_PARTICLE_TILE, 1))
                s11_col = ct.reshape(s11, (SUPERCELL_PARTICLE_TILE, 1))
                s12_col = ct.reshape(s12, (SUPERCELL_PARTICLE_TILE, 1))
                s20_col = ct.reshape(s20, (SUPERCELL_PARTICLE_TILE, 1))
                s21_col = ct.reshape(s21, (SUPERCELL_PARTICLE_TILE, 1))
                s22_col = ct.reshape(s22, (SUPERCELL_PARTICLE_TILE, 1))

                ox_f = ct.astype(ox, ct.float32)
                oy_f = ct.astype(oy, ct.float32)
                oz_f = ct.astype(oz, ct.float32)

                wx = ct.where(
                    ox == 0,
                    0.5 * (1.5 - fx0_col) * (1.5 - fx0_col),
                    ct.where(
                        ox == 1,
                        0.75 - (fx0_col - 1.0) * (fx0_col - 1.0),
                        0.5 * (fx0_col - 0.5) * (fx0_col - 0.5),
                    ),
                )
                wy = ct.where(
                    oy == 0,
                    0.5 * (1.5 - fx1_col) * (1.5 - fx1_col),
                    ct.where(
                        oy == 1,
                        0.75 - (fx1_col - 1.0) * (fx1_col - 1.0),
                        0.5 * (fx1_col - 0.5) * (fx1_col - 0.5),
                    ),
                )
                wz = ct.where(
                    oz == 0,
                    0.5 * (1.5 - fx2_col) * (1.5 - fx2_col),
                    ct.where(
                        oz == 1,
                        0.75 - (fx2_col - 1.0) * (fx2_col - 1.0),
                        0.5 * (fx2_col - 0.5) * (fx2_col - 0.5),
                    ),
                )
                dwx = ct.where(
                    ox == 0,
                    fx0_col - 1.5,
                    ct.where(ox == 1, -2.0 * (fx0_col - 1.0), fx0_col - 0.5),
                )
                dwy = ct.where(
                    oy == 0,
                    fx1_col - 1.5,
                    ct.where(oy == 1, -2.0 * (fx1_col - 1.0), fx1_col - 0.5),
                )
                dwz = ct.where(
                    oz == 0,
                    fx2_col - 1.5,
                    ct.where(oz == 1, -2.0 * (fx2_col - 1.0), fx2_col - 0.5),
                )

                weight = wx * wy * wz
                dw0 = inv_dx * dwx * wy * wz
                dw1 = inv_dx * wx * dwy * wz
                dw2 = inv_dx * wx * wy * dwz

                dpos0 = (ox_f - fx0_col) * dx
                dpos1 = (oy_f - fx1_col) * dx
                dpos2 = (oz_f - fx2_col) * dx

                affine0 = vp0_col + C00_col * dpos0 + C01_col * dpos1 + C02_col * dpos2
                affine1 = vp1_col + C10_col * dpos0 + C11_col * dpos1 + C12_col * dpos2
                affine2 = vp2_col + C20_col * dpos0 + C21_col * dpos1 + C22_col * dpos2

                stress_dw0 = s00_col * dw0 + s01_col * dw1 + s02_col * dw2
                stress_dw1 = s10_col * dw0 + s11_col * dw1 + s12_col * dw2
                stress_dw2 = s20_col * dw0 + s21_col * dw1 + s22_col * dw2

                mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
                mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
                mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
                mass = p_mass * weight

                contrib01 = ct.cat(
                    (
                        ct.reshape(mv0, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                        ct.reshape(mv1, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                    ),
                    axis=2,
                )
                contrib2m = ct.cat(
                    (
                        ct.reshape(mv2, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                        ct.reshape(mass, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                    ),
                    axis=2,
                )
                contrib = ct.cat((contrib01, contrib2m), axis=2)
                mask = ct.reshape(contributes, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1))
                acc += ct.sum(ct.where(mask, contrib, 0.0), axis=0)
                chunk_start += particle_tile

            grid_idx = gi * G * G + gj * G + gk
            acc0 = ct.reshape(ct.extract(acc, (0, 0), shape=(SUPERCELL_NODE_TILE, 1)), (SUPERCELL_NODE_TILE,))
            acc1 = ct.reshape(ct.extract(acc, (0, 1), shape=(SUPERCELL_NODE_TILE, 1)), (SUPERCELL_NODE_TILE,))
            acc2 = ct.reshape(ct.extract(acc, (0, 2), shape=(SUPERCELL_NODE_TILE, 1)), (SUPERCELL_NODE_TILE,))
            accm = ct.reshape(ct.extract(acc, (0, 3), shape=(SUPERCELL_NODE_TILE, 1)), (SUPERCELL_NODE_TILE,))
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 0, acc0, mask=in_physical_tile)
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 1, acc1, mask=in_physical_tile)
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 2, acc2, mask=in_physical_tile)
            grid_m_mem.atomic_add_offset(grid_idx, accm, mask=in_physical_tile)

    else:
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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
            chunk_start += particle_tile


@ct.kernel(occupancy=2, num_worker_warps=4)
def _p2g_sc4_tiledview_flush_kernel(
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
    color_id,
    colored_flush: ct.Constant[bool],
):
    super_id = ct.bid(0)
    x_mem = x.get_raw_memory()
    v_mem = v.get_raw_memory()
    C_mem = C.get_raw_memory()
    stress_mem = stress.get_raw_memory()
    cell_start_mem = cell_start.get_raw_memory()
    grid_mem = grid.get_raw_memory()

    Gs = G // SUPER_CELL_WIDTH
    Gp = G + 2
    if colored_flush:
        color_i = color_id // 4
        color_j = (color_id // 2) % 2
        color_k = color_id % 2
        color_grid = (Gs + 1) // 2
        li = super_id // (color_grid * color_grid)
        lj = (super_id // color_grid) % color_grid
        lk = super_id % color_grid
        si = li * 2 + color_i
        sj = lj * 2 + color_j
        sk = lk * 2 + color_k
        if si >= Gs or sj >= Gs or sk >= Gs:
            return
    else:
        si = super_id // (Gs * Gs)
        sj = (super_id // Gs) % Gs
        sk = super_id % Gs
    cell_id = si * Gs * Gs + sj * Gs + sk

    tile_i = si * SUPER_CELL_WIDTH - 1
    tile_j = sj * SUPER_CELL_WIDTH - 1
    tile_k = sk * SUPER_CELL_WIDTH - 1

    p_start = cell_start_mem.load_offset(cell_id)
    p_end = cell_start_mem.load_offset(cell_id + 1)
    if p_start == p_end:
        return
    interior_tile = (
        tile_i >= 0
        and tile_j >= 0
        and tile_k >= 0
        and tile_i + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_j + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_k + SUPER_TILE_PHYSICAL_DIM <= G
    )

    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    if interior_tile:
        grid_view_shape = (
            SC4_TILED_VIEW_TILE_DIM,
            SC4_TILED_VIEW_TILE_DIM,
            SC4_TILED_VIEW_TILE_DIM,
            GRID_CHANNELS,
        )
        grid_view_steps = (
            SUPER_CELL_WIDTH,
            SUPER_CELL_WIDTH,
            SUPER_CELL_WIDTH,
            GRID_CHANNELS,
        )
        if colored_flush:
            grid_view = grid.tiled_view(
                grid_view_shape,
                padding_mode=ct.PaddingMode.ZERO,
                traversal_steps=grid_view_steps,
            )
        else:
            grid_view = grid.tiled_view(
                grid_view_shape,
                traversal_steps=grid_view_steps,
            )

        node_lane = ct.arange(SC4_TILED_VIEW_TILE_NODES, dtype=ct.int32)
        ti = node_lane // (SC4_TILED_VIEW_TILE_DIM * SC4_TILED_VIEW_TILE_DIM)
        tj = (node_lane // SC4_TILED_VIEW_TILE_DIM) % SC4_TILED_VIEW_TILE_DIM
        tk = node_lane % SC4_TILED_VIEW_TILE_DIM
        gi = tile_i + ti
        gj = tile_j + tj
        gk = tile_k + tk
        physical_node = (
            (ti < SUPER_TILE_PHYSICAL_DIM)
            & (tj < SUPER_TILE_PHYSICAL_DIM)
            & (tk < SUPER_TILE_PHYSICAL_DIM)
        )

        gi_row = ct.reshape(gi, (1, SC4_TILED_VIEW_TILE_NODES))
        gj_row = ct.reshape(gj, (1, SC4_TILED_VIEW_TILE_NODES))
        gk_row = ct.reshape(gk, (1, SC4_TILED_VIEW_TILE_NODES))
        node_mask = ct.reshape(physical_node, (1, SC4_TILED_VIEW_TILE_NODES))

        acc = ct.zeros((SC4_TILED_VIEW_TILE_NODES, GRID_CHANNELS), ct.float32)

        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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

            b0_col = ct.reshape(b0, (SUPERCELL_PARTICLE_TILE, 1))
            b1_col = ct.reshape(b1, (SUPERCELL_PARTICLE_TILE, 1))
            b2_col = ct.reshape(b2, (SUPERCELL_PARTICLE_TILE, 1))
            active_col = ct.reshape(active, (SUPERCELL_PARTICLE_TILE, 1))

            ox = gi_row - b0_col
            oy = gj_row - b1_col
            oz = gk_row - b2_col
            contributes = (
                active_col
                & node_mask
                & (ox >= 0)
                & (ox < 3)
                & (oy >= 0)
                & (oy < 3)
                & (oz >= 0)
                & (oz < 3)
            )

            fx0_col = ct.reshape(fx0, (SUPERCELL_PARTICLE_TILE, 1))
            fx1_col = ct.reshape(fx1, (SUPERCELL_PARTICLE_TILE, 1))
            fx2_col = ct.reshape(fx2, (SUPERCELL_PARTICLE_TILE, 1))
            vp0_col = ct.reshape(vp0, (SUPERCELL_PARTICLE_TILE, 1))
            vp1_col = ct.reshape(vp1, (SUPERCELL_PARTICLE_TILE, 1))
            vp2_col = ct.reshape(vp2, (SUPERCELL_PARTICLE_TILE, 1))

            C00_col = ct.reshape(C00, (SUPERCELL_PARTICLE_TILE, 1))
            C01_col = ct.reshape(C01, (SUPERCELL_PARTICLE_TILE, 1))
            C02_col = ct.reshape(C02, (SUPERCELL_PARTICLE_TILE, 1))
            C10_col = ct.reshape(C10, (SUPERCELL_PARTICLE_TILE, 1))
            C11_col = ct.reshape(C11, (SUPERCELL_PARTICLE_TILE, 1))
            C12_col = ct.reshape(C12, (SUPERCELL_PARTICLE_TILE, 1))
            C20_col = ct.reshape(C20, (SUPERCELL_PARTICLE_TILE, 1))
            C21_col = ct.reshape(C21, (SUPERCELL_PARTICLE_TILE, 1))
            C22_col = ct.reshape(C22, (SUPERCELL_PARTICLE_TILE, 1))

            s00_col = ct.reshape(s00, (SUPERCELL_PARTICLE_TILE, 1))
            s01_col = ct.reshape(s01, (SUPERCELL_PARTICLE_TILE, 1))
            s02_col = ct.reshape(s02, (SUPERCELL_PARTICLE_TILE, 1))
            s10_col = ct.reshape(s10, (SUPERCELL_PARTICLE_TILE, 1))
            s11_col = ct.reshape(s11, (SUPERCELL_PARTICLE_TILE, 1))
            s12_col = ct.reshape(s12, (SUPERCELL_PARTICLE_TILE, 1))
            s20_col = ct.reshape(s20, (SUPERCELL_PARTICLE_TILE, 1))
            s21_col = ct.reshape(s21, (SUPERCELL_PARTICLE_TILE, 1))
            s22_col = ct.reshape(s22, (SUPERCELL_PARTICLE_TILE, 1))

            ox_f = ct.astype(ox, ct.float32)
            oy_f = ct.astype(oy, ct.float32)
            oz_f = ct.astype(oz, ct.float32)

            wx = ct.where(
                ox == 0,
                0.5 * (1.5 - fx0_col) * (1.5 - fx0_col),
                ct.where(
                    ox == 1,
                    0.75 - (fx0_col - 1.0) * (fx0_col - 1.0),
                    0.5 * (fx0_col - 0.5) * (fx0_col - 0.5),
                ),
            )
            wy = ct.where(
                oy == 0,
                0.5 * (1.5 - fx1_col) * (1.5 - fx1_col),
                ct.where(
                    oy == 1,
                    0.75 - (fx1_col - 1.0) * (fx1_col - 1.0),
                    0.5 * (fx1_col - 0.5) * (fx1_col - 0.5),
                ),
            )
            wz = ct.where(
                oz == 0,
                0.5 * (1.5 - fx2_col) * (1.5 - fx2_col),
                ct.where(
                    oz == 1,
                    0.75 - (fx2_col - 1.0) * (fx2_col - 1.0),
                    0.5 * (fx2_col - 0.5) * (fx2_col - 0.5),
                ),
            )
            dwx = ct.where(
                ox == 0,
                fx0_col - 1.5,
                ct.where(ox == 1, -2.0 * (fx0_col - 1.0), fx0_col - 0.5),
            )
            dwy = ct.where(
                oy == 0,
                fx1_col - 1.5,
                ct.where(oy == 1, -2.0 * (fx1_col - 1.0), fx1_col - 0.5),
            )
            dwz = ct.where(
                oz == 0,
                fx2_col - 1.5,
                ct.where(oz == 1, -2.0 * (fx2_col - 1.0), fx2_col - 0.5),
            )

            weight = wx * wy * wz
            dw0 = inv_dx * dwx * wy * wz
            dw1 = inv_dx * wx * dwy * wz
            dw2 = inv_dx * wx * wy * dwz

            dpos0 = (ox_f - fx0_col) * dx
            dpos1 = (oy_f - fx1_col) * dx
            dpos2 = (oz_f - fx2_col) * dx

            affine0 = vp0_col + C00_col * dpos0 + C01_col * dpos1 + C02_col * dpos2
            affine1 = vp1_col + C10_col * dpos0 + C11_col * dpos1 + C12_col * dpos2
            affine2 = vp2_col + C20_col * dpos0 + C21_col * dpos1 + C22_col * dpos2

            stress_dw0 = s00_col * dw0 + s01_col * dw1 + s02_col * dw2
            stress_dw1 = s10_col * dw0 + s11_col * dw1 + s12_col * dw2
            stress_dw2 = s20_col * dw0 + s21_col * dw1 + s22_col * dw2

            mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
            mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
            mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
            mass = p_mass * weight

            contrib01 = ct.cat(
                (
                    ct.reshape(mv0, (SUPERCELL_PARTICLE_TILE, SC4_TILED_VIEW_TILE_NODES, 1)),
                    ct.reshape(mv1, (SUPERCELL_PARTICLE_TILE, SC4_TILED_VIEW_TILE_NODES, 1)),
                ),
                axis=2,
            )
            contrib2m = ct.cat(
                (
                    ct.reshape(mv2, (SUPERCELL_PARTICLE_TILE, SC4_TILED_VIEW_TILE_NODES, 1)),
                    ct.reshape(mass, (SUPERCELL_PARTICLE_TILE, SC4_TILED_VIEW_TILE_NODES, 1)),
                ),
                axis=2,
            )
            contrib = ct.cat((contrib01, contrib2m), axis=2)
            mask = ct.reshape(
                contributes,
                (SUPERCELL_PARTICLE_TILE, SC4_TILED_VIEW_TILE_NODES, 1),
            )
            acc += ct.sum(ct.where(mask, contrib, 0.0), axis=0)
            chunk_start += particle_tile

        update = ct.reshape(
            acc,
            (
                SC4_TILED_VIEW_TILE_DIM,
                SC4_TILED_VIEW_TILE_DIM,
                SC4_TILED_VIEW_TILE_DIM,
                GRID_CHANNELS,
            ),
        )
        if colored_flush:
            old = grid_view.load((si, sj, sk, 0))
            grid_view.store((si, sj, sk, 0), old + update)
        else:
            grid_view.atomic_store_add((si, sj, sk, 0), update)

    else:
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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

                        gi = ct.minimum(ct.maximum(b0 + ox, 0), G - 1) + 1
                        gj = ct.minimum(ct.maximum(b1 + oy, 0), G - 1) + 1
                        gk = ct.minimum(ct.maximum(b2 + oz, 0), G - 1) + 1
                        grid_idx = gi * Gp * Gp + gj * Gp + gk

                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 0, mv0, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 1, mv1, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 2, mv2, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 3, mass, mask=active)
            chunk_start += particle_tile


@ct.kernel(occupancy=2, num_worker_warps=4)
def _p2g_sc4_colored_arena256_kernel(
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
    color_id,
):
    compact_id = ct.bid(0)
    x_mem = x.get_raw_memory()
    v_mem = v.get_raw_memory()
    C_mem = C.get_raw_memory()
    stress_mem = stress.get_raw_memory()
    cell_start_mem = cell_start.get_raw_memory()
    grid_mem = grid.get_raw_memory()

    Gs = G // SUPER_CELL_WIDTH
    Gp = G + 2
    color_i = color_id // 4
    color_j = (color_id // 2) % 2
    color_k = color_id % 2
    color_grid = (Gs + 1) // 2
    li = compact_id // (color_grid * color_grid)
    lj = (compact_id // color_grid) % color_grid
    lk = compact_id % color_grid
    si = li * 2 + color_i
    sj = lj * 2 + color_j
    sk = lk * 2 + color_k
    if si >= Gs or sj >= Gs or sk >= Gs:
        return

    cell_id = si * Gs * Gs + sj * Gs + sk
    p_start = cell_start_mem.load_offset(cell_id)
    p_end = cell_start_mem.load_offset(cell_id + 1)
    if p_start == p_end:
        return

    tile_i = si * SUPER_CELL_WIDTH - 1
    tile_j = sj * SUPER_CELL_WIDTH - 1
    tile_k = sk * SUPER_CELL_WIDTH - 1
    interior_tile = (
        tile_i >= 0
        and tile_j >= 0
        and tile_k >= 0
        and tile_i + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_j + SUPER_TILE_PHYSICAL_DIM <= G
        and tile_k + SUPER_TILE_PHYSICAL_DIM <= G
    )

    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    if interior_tile:
        node_lane = ct.arange(SUPERCELL_NODE_TILE, dtype=ct.int32)
        local_node = node_lane
        node_active = local_node < SUPER_TILE_NODES

        ti = local_node // (SUPER_TILE_DIM * SUPER_TILE_DIM)
        tj = (local_node // SUPER_TILE_DIM) % SUPER_TILE_DIM
        tk = local_node % SUPER_TILE_DIM
        gi = tile_i + ti
        gj = tile_j + tj
        gk = tile_k + tk
        in_physical_tile = (
            node_active
            & (ti < SUPER_TILE_PHYSICAL_DIM)
            & (tj < SUPER_TILE_PHYSICAL_DIM)
            & (tk < SUPER_TILE_PHYSICAL_DIM)
        )

        gi_row = ct.reshape(gi, (1, SUPERCELL_NODE_TILE))
        gj_row = ct.reshape(gj, (1, SUPERCELL_NODE_TILE))
        gk_row = ct.reshape(gk, (1, SUPERCELL_NODE_TILE))
        node_mask = ct.reshape(in_physical_tile, (1, SUPERCELL_NODE_TILE))

        acc = ct.zeros((SUPERCELL_NODE_TILE, GRID_CHANNELS), ct.float32)

        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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

            b0_col = ct.reshape(b0, (SUPERCELL_PARTICLE_TILE, 1))
            b1_col = ct.reshape(b1, (SUPERCELL_PARTICLE_TILE, 1))
            b2_col = ct.reshape(b2, (SUPERCELL_PARTICLE_TILE, 1))
            active_col = ct.reshape(active, (SUPERCELL_PARTICLE_TILE, 1))

            ox = gi_row - b0_col
            oy = gj_row - b1_col
            oz = gk_row - b2_col
            contributes = (
                active_col
                & node_mask
                & (ox >= 0)
                & (ox < 3)
                & (oy >= 0)
                & (oy < 3)
                & (oz >= 0)
                & (oz < 3)
            )

            fx0_col = ct.reshape(fx0, (SUPERCELL_PARTICLE_TILE, 1))
            fx1_col = ct.reshape(fx1, (SUPERCELL_PARTICLE_TILE, 1))
            fx2_col = ct.reshape(fx2, (SUPERCELL_PARTICLE_TILE, 1))
            vp0_col = ct.reshape(vp0, (SUPERCELL_PARTICLE_TILE, 1))
            vp1_col = ct.reshape(vp1, (SUPERCELL_PARTICLE_TILE, 1))
            vp2_col = ct.reshape(vp2, (SUPERCELL_PARTICLE_TILE, 1))

            C00_col = ct.reshape(C00, (SUPERCELL_PARTICLE_TILE, 1))
            C01_col = ct.reshape(C01, (SUPERCELL_PARTICLE_TILE, 1))
            C02_col = ct.reshape(C02, (SUPERCELL_PARTICLE_TILE, 1))
            C10_col = ct.reshape(C10, (SUPERCELL_PARTICLE_TILE, 1))
            C11_col = ct.reshape(C11, (SUPERCELL_PARTICLE_TILE, 1))
            C12_col = ct.reshape(C12, (SUPERCELL_PARTICLE_TILE, 1))
            C20_col = ct.reshape(C20, (SUPERCELL_PARTICLE_TILE, 1))
            C21_col = ct.reshape(C21, (SUPERCELL_PARTICLE_TILE, 1))
            C22_col = ct.reshape(C22, (SUPERCELL_PARTICLE_TILE, 1))

            s00_col = ct.reshape(s00, (SUPERCELL_PARTICLE_TILE, 1))
            s01_col = ct.reshape(s01, (SUPERCELL_PARTICLE_TILE, 1))
            s02_col = ct.reshape(s02, (SUPERCELL_PARTICLE_TILE, 1))
            s10_col = ct.reshape(s10, (SUPERCELL_PARTICLE_TILE, 1))
            s11_col = ct.reshape(s11, (SUPERCELL_PARTICLE_TILE, 1))
            s12_col = ct.reshape(s12, (SUPERCELL_PARTICLE_TILE, 1))
            s20_col = ct.reshape(s20, (SUPERCELL_PARTICLE_TILE, 1))
            s21_col = ct.reshape(s21, (SUPERCELL_PARTICLE_TILE, 1))
            s22_col = ct.reshape(s22, (SUPERCELL_PARTICLE_TILE, 1))

            ox_f = ct.astype(ox, ct.float32)
            oy_f = ct.astype(oy, ct.float32)
            oz_f = ct.astype(oz, ct.float32)

            wx = ct.where(
                ox == 0,
                0.5 * (1.5 - fx0_col) * (1.5 - fx0_col),
                ct.where(
                    ox == 1,
                    0.75 - (fx0_col - 1.0) * (fx0_col - 1.0),
                    0.5 * (fx0_col - 0.5) * (fx0_col - 0.5),
                ),
            )
            wy = ct.where(
                oy == 0,
                0.5 * (1.5 - fx1_col) * (1.5 - fx1_col),
                ct.where(
                    oy == 1,
                    0.75 - (fx1_col - 1.0) * (fx1_col - 1.0),
                    0.5 * (fx1_col - 0.5) * (fx1_col - 0.5),
                ),
            )
            wz = ct.where(
                oz == 0,
                0.5 * (1.5 - fx2_col) * (1.5 - fx2_col),
                ct.where(
                    oz == 1,
                    0.75 - (fx2_col - 1.0) * (fx2_col - 1.0),
                    0.5 * (fx2_col - 0.5) * (fx2_col - 0.5),
                ),
            )
            dwx = ct.where(
                ox == 0,
                fx0_col - 1.5,
                ct.where(ox == 1, -2.0 * (fx0_col - 1.0), fx0_col - 0.5),
            )
            dwy = ct.where(
                oy == 0,
                fx1_col - 1.5,
                ct.where(oy == 1, -2.0 * (fx1_col - 1.0), fx1_col - 0.5),
            )
            dwz = ct.where(
                oz == 0,
                fx2_col - 1.5,
                ct.where(oz == 1, -2.0 * (fx2_col - 1.0), fx2_col - 0.5),
            )

            weight = wx * wy * wz
            dw0 = inv_dx * dwx * wy * wz
            dw1 = inv_dx * wx * dwy * wz
            dw2 = inv_dx * wx * wy * dwz

            dpos0 = (ox_f - fx0_col) * dx
            dpos1 = (oy_f - fx1_col) * dx
            dpos2 = (oz_f - fx2_col) * dx

            affine0 = vp0_col + C00_col * dpos0 + C01_col * dpos1 + C02_col * dpos2
            affine1 = vp1_col + C10_col * dpos0 + C11_col * dpos1 + C12_col * dpos2
            affine2 = vp2_col + C20_col * dpos0 + C21_col * dpos1 + C22_col * dpos2

            stress_dw0 = s00_col * dw0 + s01_col * dw1 + s02_col * dw2
            stress_dw1 = s10_col * dw0 + s11_col * dw1 + s12_col * dw2
            stress_dw2 = s20_col * dw0 + s21_col * dw1 + s22_col * dw2

            mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
            mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
            mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
            mass = p_mass * weight

            contrib01 = ct.cat(
                (
                    ct.reshape(mv0, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                    ct.reshape(mv1, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                ),
                axis=2,
            )
            contrib2m = ct.cat(
                (
                    ct.reshape(mv2, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                    ct.reshape(mass, (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1)),
                ),
                axis=2,
            )
            contrib = ct.cat((contrib01, contrib2m), axis=2)
            mask = ct.reshape(
                contributes,
                (SUPERCELL_PARTICLE_TILE, SUPERCELL_NODE_TILE, 1),
            )
            acc += ct.sum(ct.where(mask, contrib, 0.0), axis=0)
            chunk_start += particle_tile

        gpi = gi + 1
        gpj = gj + 1
        gpk = gk + 1
        grid_idx = gpi * Gp * Gp + gpj * Gp + gpk
        acc0 = ct.reshape(
            ct.extract(acc, (0, 0), shape=(SUPERCELL_NODE_TILE, 1)),
            (SUPERCELL_NODE_TILE,),
        )
        acc1 = ct.reshape(
            ct.extract(acc, (0, 1), shape=(SUPERCELL_NODE_TILE, 1)),
            (SUPERCELL_NODE_TILE,),
        )
        acc2 = ct.reshape(
            ct.extract(acc, (0, 2), shape=(SUPERCELL_NODE_TILE, 1)),
            (SUPERCELL_NODE_TILE,),
        )
        accm = ct.reshape(
            ct.extract(acc, (0, 3), shape=(SUPERCELL_NODE_TILE, 1)),
            (SUPERCELL_NODE_TILE,),
        )
        old0 = grid_mem.load_offset(grid_idx * GRID_CHANNELS + 0, mask=in_physical_tile)
        old1 = grid_mem.load_offset(grid_idx * GRID_CHANNELS + 1, mask=in_physical_tile)
        old2 = grid_mem.load_offset(grid_idx * GRID_CHANNELS + 2, mask=in_physical_tile)
        oldm = grid_mem.load_offset(grid_idx * GRID_CHANNELS + 3, mask=in_physical_tile)
        grid_mem.store_offset(grid_idx * GRID_CHANNELS + 0, old0 + acc0, mask=in_physical_tile)
        grid_mem.store_offset(grid_idx * GRID_CHANNELS + 1, old1 + acc1, mask=in_physical_tile)
        grid_mem.store_offset(grid_idx * GRID_CHANNELS + 2, old2 + acc2, mask=in_physical_tile)
        grid_mem.store_offset(grid_idx * GRID_CHANNELS + 3, oldm + accm, mask=in_physical_tile)

    else:
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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

                        gi_fallback = ct.minimum(ct.maximum(b0 + ox, 0), G - 1) + 1
                        gj_fallback = ct.minimum(ct.maximum(b1 + oy, 0), G - 1) + 1
                        gk_fallback = ct.minimum(ct.maximum(b2 + oz, 0), G - 1) + 1
                        grid_idx = gi_fallback * Gp * Gp + gj_fallback * Gp + gk_fallback

                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 0, mv0, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 1, mv1, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 2, mv2, mask=active)
                        grid_mem.atomic_add_offset(grid_idx * GRID_CHANNELS + 3, mass, mask=active)
            chunk_start += particle_tile


@ct.kernel
def _p2g_native4_kernel(
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
):
    super_id = ct.bid(0)
    x_mem = x.get_raw_memory()
    v_mem = v.get_raw_memory()
    C_mem = C.get_raw_memory()
    stress_mem = stress.get_raw_memory()
    cell_start_mem = cell_start.get_raw_memory()
    grid_mv_mem = grid_mv.get_raw_memory()
    grid_m_mem = grid_m.get_raw_memory()

    Gs = G // NATIVE_SUPER_CELL_WIDTH
    si = super_id // (Gs * Gs)
    sj = (super_id // Gs) % Gs
    sk = super_id % Gs

    tile_i = si * NATIVE_SUPER_CELL_WIDTH - 1
    tile_j = sj * NATIVE_SUPER_CELL_WIDTH - 1
    tile_k = sk * NATIVE_SUPER_CELL_WIDTH - 1

    p_start = cell_start_mem.load_offset(super_id)
    p_end = cell_start_mem.load_offset(super_id + 1)
    if p_start == p_end:
        return
    interior_tile = (
        tile_i >= 0
        and tile_j >= 0
        and tile_k >= 0
        and tile_i + NATIVE_NODE_TILE_DIM <= G
        and tile_j + NATIVE_NODE_TILE_DIM <= G
        and tile_k + NATIVE_NODE_TILE_DIM <= G
    )

    p_lane = ct.arange(particle_tile, dtype=ct.int32)

    if interior_tile:
        node_lane = ct.arange(NATIVE_NODE_TILE_DIM, dtype=ct.int32)
        ti = ct.reshape(node_lane, (1, NATIVE_NODE_TILE_DIM, 1, 1))
        tj = ct.reshape(node_lane, (1, 1, NATIVE_NODE_TILE_DIM, 1))
        tk = ct.reshape(node_lane, (1, 1, 1, NATIVE_NODE_TILE_DIM))
        gi = tile_i + ti
        gj = tile_j + tj
        gk = tile_k + tk

        acc0 = ct.zeros(
            (NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM),
            ct.float32,
        )
        acc1 = ct.zeros(
            (NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM),
            ct.float32,
        )
        acc2 = ct.zeros(
            (NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM),
            ct.float32,
        )
        accm = ct.zeros(
            (NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM, NATIVE_NODE_TILE_DIM),
            ct.float32,
        )

        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end
            load_p = ct.minimum(p, p_end - 1)

            x_tile = ct.load_advanced_indexing(
                x,
                (load_p, ct.Slice(0, PARTICLE_VECTOR_TILE)),
                padding_mode=ct.PaddingMode.ZERO,
            )
            v_tile = ct.load_advanced_indexing(
                v,
                (load_p, ct.Slice(0, PARTICLE_VECTOR_TILE)),
                padding_mode=ct.PaddingMode.ZERO,
            )
            C_tile = ct.load_advanced_indexing(
                C,
                (load_p, ct.Slice(0, PARTICLE_MATRIX_TILE)),
                padding_mode=ct.PaddingMode.ZERO,
            )
            stress_tile = ct.load_advanced_indexing(
                stress,
                (load_p, ct.Slice(0, PARTICLE_MATRIX_TILE)),
                padding_mode=ct.PaddingMode.ZERO,
            )

            px0 = ct.reshape(
                ct.extract(x_tile, (0, 0), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            ) * inv_dx
            px1 = ct.reshape(
                ct.extract(x_tile, (0, 1), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            ) * inv_dx
            px2 = ct.reshape(
                ct.extract(x_tile, (0, 2), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            ) * inv_dx

            b0 = ct.astype(ct.floor(px0 - 0.5), ct.int32)
            b1 = ct.astype(ct.floor(px1 - 0.5), ct.int32)
            b2 = ct.astype(ct.floor(px2 - 0.5), ct.int32)

            fx0 = px0 - ct.astype(b0, ct.float32)
            fx1 = px1 - ct.astype(b1, ct.float32)
            fx2 = px2 - ct.astype(b2, ct.float32)

            vp0 = ct.reshape(
                ct.extract(v_tile, (0, 0), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            vp1 = ct.reshape(
                ct.extract(v_tile, (0, 1), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            vp2 = ct.reshape(
                ct.extract(v_tile, (0, 2), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )

            C00 = ct.reshape(
                ct.extract(C_tile, (0, 0), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C01 = ct.reshape(
                ct.extract(C_tile, (0, 1), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C02 = ct.reshape(
                ct.extract(C_tile, (0, 2), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C10 = ct.reshape(
                ct.extract(C_tile, (0, 3), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C11 = ct.reshape(
                ct.extract(C_tile, (0, 4), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C12 = ct.reshape(
                ct.extract(C_tile, (0, 5), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C20 = ct.reshape(
                ct.extract(C_tile, (0, 6), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C21 = ct.reshape(
                ct.extract(C_tile, (0, 7), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            C22 = ct.reshape(
                ct.extract(C_tile, (0, 8), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )

            s00 = ct.reshape(
                ct.extract(stress_tile, (0, 0), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s01 = ct.reshape(
                ct.extract(stress_tile, (0, 1), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s02 = ct.reshape(
                ct.extract(stress_tile, (0, 2), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s10 = ct.reshape(
                ct.extract(stress_tile, (0, 3), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s11 = ct.reshape(
                ct.extract(stress_tile, (0, 4), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s12 = ct.reshape(
                ct.extract(stress_tile, (0, 5), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s20 = ct.reshape(
                ct.extract(stress_tile, (0, 6), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s21 = ct.reshape(
                ct.extract(stress_tile, (0, 7), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )
            s22 = ct.reshape(
                ct.extract(stress_tile, (0, 8), shape=(NATIVE_PARTICLE_TILE, 1)),
                (NATIVE_PARTICLE_TILE,),
            )

            b0_col = ct.reshape(b0, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            b1_col = ct.reshape(b1, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            b2_col = ct.reshape(b2, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            active_col = ct.reshape(active, (NATIVE_PARTICLE_TILE, 1, 1, 1))

            ox = gi - b0_col
            oy = gj - b1_col
            oz = gk - b2_col
            contributes = (
                active_col
                & (ox >= 0) & (ox < 3)
                & (oy >= 0) & (oy < 3)
                & (oz >= 0) & (oz < 3)
            )

            fx0_col = ct.reshape(fx0, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            fx1_col = ct.reshape(fx1, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            fx2_col = ct.reshape(fx2, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            vp0_col = ct.reshape(vp0, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            vp1_col = ct.reshape(vp1, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            vp2_col = ct.reshape(vp2, (NATIVE_PARTICLE_TILE, 1, 1, 1))

            C00_col = ct.reshape(C00, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C01_col = ct.reshape(C01, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C02_col = ct.reshape(C02, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C10_col = ct.reshape(C10, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C11_col = ct.reshape(C11, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C12_col = ct.reshape(C12, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C20_col = ct.reshape(C20, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C21_col = ct.reshape(C21, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            C22_col = ct.reshape(C22, (NATIVE_PARTICLE_TILE, 1, 1, 1))

            s00_col = ct.reshape(s00, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s01_col = ct.reshape(s01, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s02_col = ct.reshape(s02, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s10_col = ct.reshape(s10, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s11_col = ct.reshape(s11, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s12_col = ct.reshape(s12, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s20_col = ct.reshape(s20, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s21_col = ct.reshape(s21, (NATIVE_PARTICLE_TILE, 1, 1, 1))
            s22_col = ct.reshape(s22, (NATIVE_PARTICLE_TILE, 1, 1, 1))

            ox_f = ct.astype(ox, ct.float32)
            oy_f = ct.astype(oy, ct.float32)
            oz_f = ct.astype(oz, ct.float32)

            wx = ct.where(
                ox == 0,
                0.5 * (1.5 - fx0_col) * (1.5 - fx0_col),
                ct.where(
                    ox == 1,
                    0.75 - (fx0_col - 1.0) * (fx0_col - 1.0),
                    0.5 * (fx0_col - 0.5) * (fx0_col - 0.5),
                ),
            )
            wy = ct.where(
                oy == 0,
                0.5 * (1.5 - fx1_col) * (1.5 - fx1_col),
                ct.where(
                    oy == 1,
                    0.75 - (fx1_col - 1.0) * (fx1_col - 1.0),
                    0.5 * (fx1_col - 0.5) * (fx1_col - 0.5),
                ),
            )
            wz = ct.where(
                oz == 0,
                0.5 * (1.5 - fx2_col) * (1.5 - fx2_col),
                ct.where(
                    oz == 1,
                    0.75 - (fx2_col - 1.0) * (fx2_col - 1.0),
                    0.5 * (fx2_col - 0.5) * (fx2_col - 0.5),
                ),
            )
            dwx = ct.where(
                ox == 0,
                fx0_col - 1.5,
                ct.where(ox == 1, -2.0 * (fx0_col - 1.0), fx0_col - 0.5),
            )
            dwy = ct.where(
                oy == 0,
                fx1_col - 1.5,
                ct.where(oy == 1, -2.0 * (fx1_col - 1.0), fx1_col - 0.5),
            )
            dwz = ct.where(
                oz == 0,
                fx2_col - 1.5,
                ct.where(oz == 1, -2.0 * (fx2_col - 1.0), fx2_col - 0.5),
            )

            weight = wx * wy * wz
            dw0 = inv_dx * dwx * wy * wz
            dw1 = inv_dx * wx * dwy * wz
            dw2 = inv_dx * wx * wy * dwz

            dpos0 = (ox_f - fx0_col) * dx
            dpos1 = (oy_f - fx1_col) * dx
            dpos2 = (oz_f - fx2_col) * dx

            affine0 = vp0_col + C00_col * dpos0 + C01_col * dpos1 + C02_col * dpos2
            affine1 = vp1_col + C10_col * dpos0 + C11_col * dpos1 + C12_col * dpos2
            affine2 = vp2_col + C20_col * dpos0 + C21_col * dpos1 + C22_col * dpos2

            stress_dw0 = s00_col * dw0 + s01_col * dw1 + s02_col * dw2
            stress_dw1 = s10_col * dw0 + s11_col * dw1 + s12_col * dw2
            stress_dw2 = s20_col * dw0 + s21_col * dw1 + s22_col * dw2

            mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
            mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
            mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
            mass = p_mass * weight

            acc0 += ct.sum(ct.where(contributes, mv0, 0.0), axis=0)
            acc1 += ct.sum(ct.where(contributes, mv1, 0.0), axis=0)
            acc2 += ct.sum(ct.where(contributes, mv2, 0.0), axis=0)
            accm += ct.sum(ct.where(contributes, mass, 0.0), axis=0)
            chunk_start += particle_tile

        grid_idx = gi * G * G + gj * G + gk
        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 0, acc0)
        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 1, acc1)
        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 2, acc2)
        grid_m_mem.atomic_add_offset(grid_idx, accm)

    else:
        chunk_start = p_start
        while chunk_start < p_end:
            p = chunk_start + p_lane
            active = p < p_end

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

                        gi_fallback = ct.minimum(ct.maximum(b0 + ox, 0), G - 1)
                        gj_fallback = ct.minimum(ct.maximum(b1 + oy, 0), G - 1)
                        gk_fallback = ct.minimum(ct.maximum(b2 + oz, 0), G - 1)
                        grid_idx = gi_fallback * G * G + gj_fallback * G + gk_fallback

                        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 0, mv0, mask=active)
                        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 1, mv1, mask=active)
                        grid_mv_mem.atomic_add_offset(grid_idx * 3 + 2, mv2, mask=active)
                        grid_m_mem.atomic_add_offset(grid_idx, mass, mask=active)
            chunk_start += particle_tile


def _cutile_p2g_supercell_reduce(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx, *, tile_loads
):
    n = x.shape[0]
    g = int(num_grids)
    g3 = g ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = cutile_call(
        ((g // SUPER_CELL_WIDTH) ** 3,),
        _p2g_supercell_reduce_kernel,
        (
            x,
            v,
            C_flat,
            stress_flat,
            cell_start,
            InputOutput(grid_mv_flat),
            InputOutput(grid_m),
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            SUPERCELL_PARTICLE_TILE,
            SUPERCELL_NODE_TILE,
            bool(tile_loads),
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


def cutile_p2g_supercell_reduce(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Super-cell-local cuTile reduction before the final global scatter."""
    return _cutile_p2g_supercell_reduce(
        x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx,
        tile_loads=False,
    )


def cutile_p2g_supercell_reduce_tiled_loads(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Super-cell cuTile reduction with tile-shaped particle attribute loads."""
    return _cutile_p2g_supercell_reduce(
        x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx,
        tile_loads=True,
    )


def cutile_p2g_sc4_tiledview_flush(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """SC=4 super-cell P2G with a padded 4-channel tiled-view flush."""
    n = x.shape[0]
    g = int(num_grids)
    gp = g + 2
    g3 = g ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid = jnp.zeros((gp, gp, gp, GRID_CHANNELS), dtype=jnp.float32)

    grid = cutile_call(
        ((g // SUPER_CELL_WIDTH) ** 3,),
        _p2g_sc4_tiledview_flush_kernel,
        (
            x,
            v,
            C_flat,
            stress_flat,
            cell_start,
            InputOutput(grid),
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            SUPERCELL_PARTICLE_TILE,
            0,
            False,
        ),
    )
    grid = grid[1 : g + 1, 1 : g + 1, 1 : g + 1, :]
    grid_flat = grid.reshape((g3, GRID_CHANNELS))
    return grid_flat[:, :3], grid_flat[:, 3]


def cutile_p2g_sc4_colored_tiledview_store(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """SC=4 super-cell P2G with 8-color non-atomic tiled-view writeback."""
    n = x.shape[0]
    g = int(num_grids)
    gp = g + 2
    g3 = g ** 3
    gs = g // SUPER_CELL_WIDTH
    color_grid = (gs + 1) // 2
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid = jnp.zeros((gp, gp, gp, GRID_CHANNELS), dtype=jnp.float32)

    for color in range(8):
        grid = cutile_call(
            (color_grid ** 3,),
            _p2g_sc4_tiledview_flush_kernel,
            (
                x,
                v,
                C_flat,
                stress_flat,
                cell_start,
                InputOutput(grid),
                g,
                float(dt),
                float(vol),
                float(p_mass),
                float(inv_dx),
                float(dx),
                SUPERCELL_PARTICLE_TILE,
                color,
                True,
            ),
        )
    grid = grid[1 : g + 1, 1 : g + 1, 1 : g + 1, :]
    grid_flat = grid.reshape((g3, GRID_CHANNELS))
    return grid_flat[:, :3], grid_flat[:, 3]


def cutile_p2g_sc4_colored_arena256_store(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """SC=4 P2G with a 256-lane local arena and colored non-atomic writeback."""
    n = x.shape[0]
    g = int(num_grids)
    gp = g + 2
    g3 = g ** 3
    gs = g // SUPER_CELL_WIDTH
    color_grid = (gs + 1) // 2
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid = jnp.zeros((gp, gp, gp, GRID_CHANNELS), dtype=jnp.float32)

    for color in range(SC4_COLOR_COUNT):
        grid = cutile_call(
            (color_grid ** 3,),
            _p2g_sc4_colored_arena256_kernel,
            (
                x,
                v,
                C_flat,
                stress_flat,
                cell_start,
                InputOutput(grid),
                g,
                float(dt),
                float(vol),
                float(p_mass),
                float(inv_dx),
                float(dx),
                SUPERCELL_PARTICLE_TILE,
                color,
            ),
        )
    grid = grid[1 : g + 1, 1 : g + 1, 1 : g + 1, :]
    grid_flat = grid.reshape((g3, GRID_CHANNELS))
    return grid_flat[:, :3], grid_flat[:, 3]


def cutile_p2g_native4(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Native 4x4x4 node-tile P2G from 2x2x2 sorted grid supercells."""
    n = x.shape[0]
    g = int(num_grids)
    g3 = g ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid_mv_flat = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = cutile_call(
        ((g // NATIVE_SUPER_CELL_WIDTH) ** 3,),
        _p2g_native4_kernel,
        (
            x,
            v,
            C_flat,
            stress_flat,
            cell_start,
            InputOutput(grid_mv_flat),
            InputOutput(grid_m),
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            NATIVE_PARTICLE_TILE,
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m
