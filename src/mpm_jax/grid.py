import jax
import jax.numpy as jnp


@jax.jit
def grid_update(grid_mv, grid_m, gravity, dt, damping):
    """Normalize momentum by mass, apply gravity and damping.

    Embarrassingly parallel over grid nodes.
    """
    valid = grid_m > 1e-15
    grid_mv = jnp.where(valid[:, None], grid_mv / grid_m[:, None], grid_mv)
    grid_mv = damping * (grid_mv + dt * gravity)
    return grid_mv


def build_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
