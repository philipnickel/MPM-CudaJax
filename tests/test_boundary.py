import jax.numpy as jnp

from mpm_jax.solver import _apply_sticky_floor, _sticky_floor_mask


def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def test_sticky_boundary_zeroes_velocity_below_surface():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    sticky_floor = _sticky_floor_mask(grid_x, dx)
    grid_mv = jnp.ones((num_grids**3, 3))
    result = _apply_sticky_floor(grid_mv, sticky_floor)
    below = grid_x[:, 2] * dx < 0.02
    assert jnp.allclose(result[below], 0.0)
    assert jnp.allclose(result[~below], 1.0)


def test_sticky_boundary_is_noop_when_mask_is_empty():
    num_grids = 5
    grid_mv = jnp.ones((num_grids**3, 3))
    sticky_floor = jnp.zeros((num_grids**3,), dtype=bool)
    assert jnp.allclose(_apply_sticky_floor(grid_mv, sticky_floor), grid_mv)
