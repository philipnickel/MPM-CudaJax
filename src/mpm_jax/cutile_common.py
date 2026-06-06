"""Shared cuTile MLS-MPM math helpers."""

import cuda.tile as ct


def _quad_w(o, fx):
    """Quadratic B-spline weight for offset tile ``o`` and fractional ``fx``."""
    return ct.where(
        o == 0,
        0.5 * (1.5 - fx) * (1.5 - fx),
        ct.where(o == 1, 0.75 - (fx - 1.0) * (fx - 1.0), 0.5 * (fx - 0.5) * (fx - 0.5)),
    )


def _quad_dw(o, fx):
    return ct.where(o == 0, fx - 1.5, ct.where(o == 1, -2.0 * (fx - 1.0), fx - 0.5))


def _vector_columns(tile, particle_tile):
    columns = ()
    for axis in ct.static_iter(range(3)):
        column = ct.extract(tile, (0, axis), shape=(particle_tile, 1))
        columns += (column,)
    return columns


def _matrix_columns(tile, particle_tile):
    columns = ()
    for column_id in ct.static_iter(range(9)):
        column = ct.extract(tile, (0, column_id), shape=(particle_tile, 1))
        columns += (column,)
    return columns


def _load_particle_columns(x, v, C, stress, p, active, inv_dx, particle_tile):
    active_v = ct.reshape(active, (particle_tile, 1))

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
        (p, ct.Slice(0, 16)),
        padding_mode=ct.PaddingMode.ZERO,
    )
    stress_p = ct.load_advanced_indexing(
        stress,
        (p, ct.Slice(0, 16)),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x_p = ct.where(active_v, x_p, 0.0) * inv_dx
    v_p = ct.where(active_v, v_p, 0.0)
    C_p = ct.where(active_v, C_p, 0.0)
    stress_p = ct.where(active_v, stress_p, 0.0)

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
