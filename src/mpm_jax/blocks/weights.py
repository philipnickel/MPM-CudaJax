import jax
import jax.numpy as jnp


# 27 offsets for the 3x3x3 quadratic B-spline support around each particle.
OFFSET_27 = jnp.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float32,
)  # (27, 3)


def _single_particle_weights(x_p, inv_dx, dx, num_grids):
    """Compute B-spline weights, gradients, dpos, and indices for one particle.

    Args:
        x_p: (3,) particle position

    Returns:
        weight:  (27,)   scalar weight per stencil node
        dweight: (27, 3) weight gradient per stencil node
        dpos:    (27, 3) offset from particle to each stencil node
        index:   (27,)   flat grid index per stencil node
    """
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)  # (3,)
    fx = px - base.astype(jnp.float32)       # (3,)

    # Quadratic B-spline weights per dimension: (3, 3) -> offset x dim
    w = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])  # (3, 3): [offset_idx, spatial_dim]

    dw = jnp.stack([
        fx - 1.5,
        -2.0 * (fx - 1.0),
        fx - 0.5,
    ])  # (3, 3)

    # 3D tensor product over 27 nodes
    offsets = OFFSET_27.astype(int)  # (27, 3)
    weight = w[offsets[:, 0], 0] * w[offsets[:, 1], 1] * w[offsets[:, 2], 2]  # (27,)

    dweight = inv_dx * jnp.stack([
        dw[offsets[:, 0], 0] *  w[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] * dw[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] *  w[offsets[:, 1], 1] * dw[offsets[:, 2], 2],
    ], axis=-1)  # (27, 3)

    dpos = (OFFSET_27 - fx[None, :]) * dx  # (27, 3)

    # Flat grid indices
    idx_3d = base[None, :] + offsets  # (27, 3)
    index = idx_3d[:, 0] * num_grids * num_grids + idx_3d[:, 1] * num_grids + idx_3d[:, 2]
    index = jnp.clip(index, 0, num_grids ** 3 - 1)  # (27,)

    return weight, dweight, dpos, index


# vmap over particles (axis 0), keep grid params as scalars
compute_weights_and_indices = jax.vmap(
    _single_particle_weights,
    in_axes=(0, None, None, None),
)  # (N, 3) -> (N, 27), (N, 27, 3), (N, 27, 3), (N, 27)
