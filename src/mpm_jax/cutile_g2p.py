"""cuTile tiled G2P (grid-to-particle gather).

G2P is the *easy* direction for tile programming: each particle reads its 27
stencil nodes and writes only its own (x, v, C, F), so there is no write conflict
-- no atomics, no coloring, no privatization. One tile of particles loops the 27
offsets, gathers grid velocity, and accumulates the unified MLS-MPM update in
registers, exactly matching ``g2p_scan._g2p_scan_mls`` (so it passes the 1e-5
equivalence test). This is the first cuTile variant whose G2P is also tiled
(every other variant reuses the shared JAX baseline G2P).
"""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call


G2P_TILE = 32  # particles per block (occupancy; G2P is register-heavy)


@ct.kernel
def _g2p_kernel(
    x, F, grid_v, out_x, out_v, out_C, out_F,
    N: ct.Constant[int], G: ct.Constant[int],
    dt: ct.Constant[float], inv_dx: ct.Constant[float], dx: ct.Constant[float],
    clip_bound: ct.Constant[float], tile_size: ct.Constant[int],
):
    p = ct.bid(0) * tile_size + ct.arange(tile_size, dtype=ct.int32)
    active = p < N

    xr0 = ct.gather(x, p * 3 + 0, mask=active, padding_value=0.0)  # raw position
    xr1 = ct.gather(x, p * 3 + 1, mask=active, padding_value=0.0)
    xr2 = ct.gather(x, p * 3 + 2, mask=active, padding_value=0.0)
    px0 = xr0 * inv_dx
    px1 = xr1 * inv_dx
    px2 = xr2 * inv_dx
    b0 = ct.astype(ct.floor(px0 - 0.5), ct.int32)
    b1 = ct.astype(ct.floor(px1 - 0.5), ct.int32)
    b2 = ct.astype(ct.floor(px2 - 0.5), ct.int32)
    fx0 = px0 - ct.astype(b0, ct.float32)
    fx1 = px1 - ct.astype(b1, ct.float32)
    fx2 = px2 - ct.astype(b2, ct.float32)

    F00 = ct.gather(F, p * 9 + 0, mask=active, padding_value=0.0)
    F01 = ct.gather(F, p * 9 + 1, mask=active, padding_value=0.0)
    F02 = ct.gather(F, p * 9 + 2, mask=active, padding_value=0.0)
    F10 = ct.gather(F, p * 9 + 3, mask=active, padding_value=0.0)
    F11 = ct.gather(F, p * 9 + 4, mask=active, padding_value=0.0)
    F12 = ct.gather(F, p * 9 + 5, mask=active, padding_value=0.0)
    F20 = ct.gather(F, p * 9 + 6, mask=active, padding_value=0.0)
    F21 = ct.gather(F, p * 9 + 7, mask=active, padding_value=0.0)
    F22 = ct.gather(F, p * 9 + 8, mask=active, padding_value=0.0)

    v0 = ct.zeros((tile_size,), ct.float32)
    v1 = ct.zeros((tile_size,), ct.float32)
    v2 = ct.zeros((tile_size,), ct.float32)
    C00 = ct.zeros((tile_size,), ct.float32)
    C01 = ct.zeros((tile_size,), ct.float32)
    C02 = ct.zeros((tile_size,), ct.float32)
    C10 = ct.zeros((tile_size,), ct.float32)
    C11 = ct.zeros((tile_size,), ct.float32)
    C12 = ct.zeros((tile_size,), ct.float32)
    C20 = ct.zeros((tile_size,), ct.float32)
    C21 = ct.zeros((tile_size,), ct.float32)
    C22 = ct.zeros((tile_size,), ct.float32)

    for ox in ct.static_iter(range(3)):
        if ox == 0:
            wx = 0.5 * (1.5 - fx0) * (1.5 - fx0)
        elif ox == 1:
            wx = 0.75 - (fx0 - 1.0) * (fx0 - 1.0)
        else:
            wx = 0.5 * (fx0 - 0.5) * (fx0 - 0.5)
        for oy in ct.static_iter(range(3)):
            if oy == 0:
                wy = 0.5 * (1.5 - fx1) * (1.5 - fx1)
            elif oy == 1:
                wy = 0.75 - (fx1 - 1.0) * (fx1 - 1.0)
            else:
                wy = 0.5 * (fx1 - 0.5) * (fx1 - 0.5)
            for oz in ct.static_iter(range(3)):
                if oz == 0:
                    wz = 0.5 * (1.5 - fx2) * (1.5 - fx2)
                elif oz == 1:
                    wz = 0.75 - (fx2 - 1.0) * (fx2 - 1.0)
                else:
                    wz = 0.5 * (fx2 - 0.5) * (fx2 - 0.5)

                weight = wx * wy * wz
                dpos0 = (float(ox) - fx0) * dx
                dpos1 = (float(oy) - fx1) * dx
                dpos2 = (float(oz) - fx2) * dx

                # Flat node index, clamped on the FLAT id (matches g2p_scan).
                idx = (b0 + ox) * G * G + (b1 + oy) * G + (b2 + oz)
                idx = ct.minimum(ct.maximum(idx, 0), G * G * G - 1)
                gv0 = ct.gather(grid_v, idx * 3 + 0, mask=active, padding_value=0.0)
                gv1 = ct.gather(grid_v, idx * 3 + 1, mask=active, padding_value=0.0)
                gv2 = ct.gather(grid_v, idx * 3 + 2, mask=active, padding_value=0.0)

                v0 = v0 + weight * gv0
                v1 = v1 + weight * gv1
                v2 = v2 + weight * gv2
                C00 = C00 + weight * gv0 * dpos0
                C01 = C01 + weight * gv0 * dpos1
                C02 = C02 + weight * gv0 * dpos2
                C10 = C10 + weight * gv1 * dpos0
                C11 = C11 + weight * gv1 * dpos1
                C12 = C12 + weight * gv1 * dpos2
                C20 = C20 + weight * gv2 * dpos0
                C21 = C21 + weight * gv2 * dpos1
                C22 = C22 + weight * gv2 * dpos2

    s = 4.0 * inv_dx * inv_dx
    nC00, nC01, nC02 = s * C00, s * C01, s * C02
    nC10, nC11, nC12 = s * C10, s * C11, s * C12
    nC20, nC21, nC22 = s * C20, s * C21, s * C22

    lo, hi = clip_bound, 1.0 - clip_bound
    nx0 = ct.minimum(ct.maximum(xr0 + v0 * dt, lo), hi)
    nx1 = ct.minimum(ct.maximum(xr1 + v1 * dt, lo), hi)
    nx2 = ct.minimum(ct.maximum(xr2 + v2 * dt, lo), hi)

    # F_new = clip(F + dt * (newC @ F), -2, 2)
    M00 = nC00 * F00 + nC01 * F10 + nC02 * F20
    M01 = nC00 * F01 + nC01 * F11 + nC02 * F21
    M02 = nC00 * F02 + nC01 * F12 + nC02 * F22
    M10 = nC10 * F00 + nC11 * F10 + nC12 * F20
    M11 = nC10 * F01 + nC11 * F11 + nC12 * F21
    M12 = nC10 * F02 + nC11 * F12 + nC12 * F22
    M20 = nC20 * F00 + nC21 * F10 + nC22 * F20
    M21 = nC20 * F01 + nC21 * F11 + nC22 * F21
    M22 = nC20 * F02 + nC21 * F12 + nC22 * F22

    nF00 = ct.minimum(ct.maximum(F00 + dt * M00, -2.0), 2.0)
    nF01 = ct.minimum(ct.maximum(F01 + dt * M01, -2.0), 2.0)
    nF02 = ct.minimum(ct.maximum(F02 + dt * M02, -2.0), 2.0)
    nF10 = ct.minimum(ct.maximum(F10 + dt * M10, -2.0), 2.0)
    nF11 = ct.minimum(ct.maximum(F11 + dt * M11, -2.0), 2.0)
    nF12 = ct.minimum(ct.maximum(F12 + dt * M12, -2.0), 2.0)
    nF20 = ct.minimum(ct.maximum(F20 + dt * M20, -2.0), 2.0)
    nF21 = ct.minimum(ct.maximum(F21 + dt * M21, -2.0), 2.0)
    nF22 = ct.minimum(ct.maximum(F22 + dt * M22, -2.0), 2.0)

    ct.scatter(out_x, (p * 3 + 0,), nx0, mask=active)
    ct.scatter(out_x, (p * 3 + 1,), nx1, mask=active)
    ct.scatter(out_x, (p * 3 + 2,), nx2, mask=active)
    ct.scatter(out_v, (p * 3 + 0,), v0, mask=active)
    ct.scatter(out_v, (p * 3 + 1,), v1, mask=active)
    ct.scatter(out_v, (p * 3 + 2,), v2, mask=active)
    ct.scatter(out_C, (p * 9 + 0,), nC00, mask=active)
    ct.scatter(out_C, (p * 9 + 1,), nC01, mask=active)
    ct.scatter(out_C, (p * 9 + 2,), nC02, mask=active)
    ct.scatter(out_C, (p * 9 + 3,), nC10, mask=active)
    ct.scatter(out_C, (p * 9 + 4,), nC11, mask=active)
    ct.scatter(out_C, (p * 9 + 5,), nC12, mask=active)
    ct.scatter(out_C, (p * 9 + 6,), nC20, mask=active)
    ct.scatter(out_C, (p * 9 + 7,), nC21, mask=active)
    ct.scatter(out_C, (p * 9 + 8,), nC22, mask=active)
    ct.scatter(out_F, (p * 9 + 0,), nF00, mask=active)
    ct.scatter(out_F, (p * 9 + 1,), nF01, mask=active)
    ct.scatter(out_F, (p * 9 + 2,), nF02, mask=active)
    ct.scatter(out_F, (p * 9 + 3,), nF10, mask=active)
    ct.scatter(out_F, (p * 9 + 4,), nF11, mask=active)
    ct.scatter(out_F, (p * 9 + 5,), nF12, mask=active)
    ct.scatter(out_F, (p * 9 + 6,), nF20, mask=active)
    ct.scatter(out_F, (p * 9 + 7,), nF21, mask=active)
    ct.scatter(out_F, (p * 9 + 8,), nF22, mask=active)


def cutile_g2p(grid_v, x, F, num_grids, dt, inv_dx, dx, clip_bound):
    """Tiled cuTile G2P; returns (new_x, new_v, new_C, new_F) matching the baseline."""
    n = x.shape[0]
    g = int(num_grids)
    x_flat = x.reshape(-1)
    F_flat = F.reshape(-1)
    grid_v_flat = grid_v.reshape(-1)

    out_x = jnp.zeros((n * 3,), jnp.float32)
    out_v = jnp.zeros((n * 3,), jnp.float32)
    out_C = jnp.zeros((n * 9,), jnp.float32)
    out_F = jnp.zeros((n * 9,), jnp.float32)

    out_x, out_v, out_C, out_F = cutile_call(
        ((n + G2P_TILE - 1) // G2P_TILE,),
        _g2p_kernel,
        (
            x_flat, F_flat, grid_v_flat,
            InputOutput(out_x), InputOutput(out_v),
            InputOutput(out_C), InputOutput(out_F),
            int(n), g, float(dt), float(inv_dx), float(dx),
            float(clip_bound), G2P_TILE,
        ),
    )
    return (
        out_x.reshape(n, 3), out_v.reshape(n, 3),
        out_C.reshape(n, 3, 3), out_F.reshape(n, 3, 3),
    )
