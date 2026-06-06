"""Shared quadratic B-spline stencil math for P2G and G2P."""

import jax.numpy as jnp


OFFSET_27 = jnp.indices((3, 3, 3)).reshape(3, -1).T.astype(jnp.int32)


def base_fx(x, inv_dx):
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    return base, px - base.astype(jnp.float32)


def quadratic_weight_tables(fx):
    weights = jnp.stack(
        (
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ),
        axis=-2,
    )
    weight_grads = jnp.stack(
        (
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5,
        ),
        axis=-2,
    )
    return weights, weight_grads


def offset_axis_values(table, offset):
    return table[..., offset, jnp.arange(3)]


def flat_grid_index(base, offset, num_grids):
    idx_3d = base + offset
    idx = (
        idx_3d[..., 0] * num_grids * num_grids
        + idx_3d[..., 1] * num_grids
        + idx_3d[..., 2]
    )
    return jnp.clip(idx, 0, num_grids**3 - 1)


def weight_dpos_index(x, offset, inv_dx, dx, num_grids):
    base, fx = base_fx(x, inv_dx)
    weights, _ = quadratic_weight_tables(fx)
    w_axis = offset_axis_values(weights, offset)
    weight = jnp.prod(w_axis, axis=-1)
    dpos = (offset.astype(jnp.float32) - fx) * dx
    return flat_grid_index(base, offset, num_grids), weight, dpos


def weight_dweight_dpos_index(x, offset, inv_dx, dx, num_grids):
    base, fx = base_fx(x, inv_dx)
    weights, weight_grads = quadratic_weight_tables(fx)
    w_axis = offset_axis_values(weights, offset)
    dw_axis = offset_axis_values(weight_grads, offset)
    weight = jnp.prod(w_axis, axis=-1)
    dweight = inv_dx * jnp.stack(
        (
            dw_axis[..., 0] * w_axis[..., 1] * w_axis[..., 2],
            w_axis[..., 0] * dw_axis[..., 1] * w_axis[..., 2],
            w_axis[..., 0] * w_axis[..., 1] * dw_axis[..., 2],
        ),
        axis=-1,
    )
    dpos = (offset.astype(jnp.float32) - fx) * dx
    return flat_grid_index(base, offset, num_grids), weight, dweight, dpos
