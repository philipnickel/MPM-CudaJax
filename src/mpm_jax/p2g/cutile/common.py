"""Shared cuTile MLS-MPM math helpers."""

import cuda.tile as ct


VEC_USED = 3
MAT_USED = 9
GRID_CHANNELS = 4
STENCIL_NODES = 3**3


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


VEC_PAD = _next_power_of_two(VEC_USED)
MAT_PAD = _next_power_of_two(MAT_USED)
STENCIL_LANES = _next_power_of_two(STENCIL_NODES)


def _quad_w(o, fx):
    """Quadratic B-spline weight for an offset tile and fractional position."""
    return ct.where(
        o == 0,
        0.5 * (1.5 - fx) * (1.5 - fx),
        ct.where(o == 1, 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5)),
    )


def _quad_dw(o, fx):
    return ct.where(o == 0, fx - 1.5, ct.where(o == 1, -2.0 * (fx - 1.0), fx - 0.5))


def _columns(tile, count, particle_tile):
    columns = ()
    for column_id in ct.static_iter(range(count)):
        column = ct.extract(tile, (0, column_id), shape=(particle_tile, 1))
        columns += (column,)
    return columns


def _load_particle_columns(x, v, C, stress, p, inv_dx, particle_tile):
    x_p = ct.load_advanced_indexing(
        x,
        (p, ct.Slice(0, VEC_PAD)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    v_p = ct.load_advanced_indexing(
        v,
        (p, ct.Slice(0, VEC_PAD)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    C_p = ct.load_advanced_indexing(
        C,
        (p, ct.Slice(0, MAT_PAD)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    stress_p = ct.load_advanced_indexing(
        stress,
        (p, ct.Slice(0, MAT_PAD)),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x_p = x_p * inv_dx
    b = ct.astype(ct.floor(x_p - 0.5), ct.int32)
    fx = x_p - ct.astype(b, ct.float32)
    return (
        _columns(b, VEC_USED, particle_tile),
        _columns(fx, VEC_USED, particle_tile),
        _columns(v_p, VEC_USED, particle_tile),
        _columns(C_p, MAT_USED, particle_tile),
        _columns(stress_p, MAT_USED, particle_tile),
    )


def _contribution_from_offsets(fx, velocity, C, stress, offsets, dt, vol, p_mass, inv_dx, dx):
    fx0, fx1, fx2 = fx
    vp0, vp1, vp2 = velocity
    C00, C01, C02, C10, C11, C12, C20, C21, C22 = C
    s00, s01, s02, s10, s11, s12, s20, s21, s22 = stress
    ox, oy, oz = offsets

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
    )


def _stencil_contribution_columns(pcols, offsets, dt, vol, p_mass, inv_dx, dx):
    _, fx, velocity, C, stress = pcols
    return _contribution_from_offsets(
        fx, velocity, C, stress, offsets, dt, vol, p_mass, inv_dx, dx
    )


def _channel_tile(c0, c1, c2, c3, shape, axis):
    return ct.cat(
        (
            ct.cat((ct.reshape(c0, shape), ct.reshape(c1, shape)), axis),
            ct.cat((ct.reshape(c2, shape), ct.reshape(c3, shape)), axis),
        ),
        axis,
    )


def _masked_channel_sum(mask, mv0, mv1, mv2, mass, shape, axis):
    channels = _channel_tile(mv0, mv1, mv2, mass, shape, axis)
    mask = ct.reshape(mask, shape)
    return ct.sum(ct.where(mask, channels, 0.0), axis=0)


def _clip_grid_nodes(nodes, G):
    gi, gj, gk = nodes
    flat = gi * (G * G) + gj * G + gk
    flat = ct.maximum(0, ct.minimum(flat, G * G * G - 1))
    return flat // (G * G), (flat // G) % G, flat % G


def _atomic_add_grid_channels(
    grid, nodes, values, G, index_shape, channel_shape, mask=None
):
    gi, gj, gk = _clip_grid_nodes(nodes, G)
    channel = ct.reshape(
        ct.arange(GRID_CHANNELS, dtype=ct.int32),
        channel_shape,
    )
    update = values
    if mask is not None:
        update = ct.where(ct.reshape(mask, index_shape), values, 0.0)
    ct.atomic_add(
        grid,
        (
            ct.reshape(gi, index_shape),
            ct.reshape(gj, index_shape),
            ct.reshape(gk, index_shape),
            channel,
        ),
        update,
        check_bounds=False,
    )
