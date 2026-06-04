"""cuTile P2G kernels called from the shared JAX-owned frame loop."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


P2G_TILE_SIZE = 256
SUPER_CELL_WIDTH = 4
SUPER_TILE_DIM = SUPER_CELL_WIDTH + 2
SUPER_TILE_NODES = SUPER_TILE_DIM * SUPER_TILE_DIM * SUPER_TILE_DIM
SUPERCELL_PARTICLE_TILE = 8
SUPERCELL_NODE_TILE = 64


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


@ct.kernel
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
    interior_tile = (
        tile_i >= 0
        and tile_j >= 0
        and tile_k >= 0
        and tile_i + SUPER_TILE_DIM <= G
        and tile_j + SUPER_TILE_DIM <= G
        and tile_k + SUPER_TILE_DIM <= G
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

            gi_row = ct.reshape(gi, (1, SUPERCELL_NODE_TILE))
            gj_row = ct.reshape(gj, (1, SUPERCELL_NODE_TILE))
            gk_row = ct.reshape(gk, (1, SUPERCELL_NODE_TILE))
            node_mask = ct.reshape(node_active, (1, SUPERCELL_NODE_TILE))

            acc0 = ct.zeros((SUPERCELL_NODE_TILE,), ct.float32)
            acc1 = ct.zeros((SUPERCELL_NODE_TILE,), ct.float32)
            acc2 = ct.zeros((SUPERCELL_NODE_TILE,), ct.float32)
            accm = ct.zeros((SUPERCELL_NODE_TILE,), ct.float32)

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

                acc0 += ct.sum(ct.where(contributes, mv0, 0.0), axis=0)
                acc1 += ct.sum(ct.where(contributes, mv1, 0.0), axis=0)
                acc2 += ct.sum(ct.where(contributes, mv2, 0.0), axis=0)
                accm += ct.sum(ct.where(contributes, mass, 0.0), axis=0)
                chunk_start += particle_tile

            grid_idx = gi * G * G + gj * G + gk
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 0, acc0, mask=node_active)
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 1, acc1, mask=node_active)
            grid_mv_mem.atomic_add_offset(grid_idx * 3 + 2, acc2, mask=node_active)
            grid_m_mem.atomic_add_offset(grid_idx, accm, mask=node_active)

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


def cutile_p2g_supercell_reduce(
    x, v, C, stress, cell_start, num_grids, dt, vol, p_mass, inv_dx, dx
):
    """Super-cell-local cuTile reduction before the final global scatter."""
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
        ),
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m
