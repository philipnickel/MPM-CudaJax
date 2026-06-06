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
    """Per (particle, node, channel) MLS-MPM contribution tile."""
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
