import jax.numpy as jnp

from mpm_jax.boundary import StickyPlane


def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def test_sticky_boundary_zeroes_velocity_below_surface():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    boundary = StickyPlane(
        point=(1.0, 1.0, 0.02),
        normal=(0.0, 0.0, 1.0),
        start_time=0.0,
        end_time=1e3,
    )
    post_fn = boundary.bind_grid(grid_x, dx)
    grid_mv = jnp.ones((num_grids**3, 3))
    grid_m = jnp.ones((num_grids**3,))
    result = post_fn(grid_mv, grid_m, 0.0)
    point = jnp.array([1.0, 1.0, 0.02])
    normal = jnp.array([0.0, 0.0, 1.0])
    offset = grid_x * dx - point
    below = jnp.sum(offset * normal, axis=1) < 0
    assert jnp.allclose(result[below], 0.0)
    assert jnp.allclose(result[~below], 1.0)


def test_inactive_sticky_boundary_is_noop():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    boundary = StickyPlane(
        point=(1.0, 1.0, 0.02),
        normal=(0.0, 0.0, 1.0),
        start_time=1.0,
        end_time=2.0,
    )
    post_fn = boundary.bind_grid(grid_x, dx)
    grid_mv = jnp.ones((num_grids**3, 3))
    grid_m = jnp.ones((num_grids**3,))
    assert jnp.allclose(post_fn(grid_mv, grid_m, 0.0), grid_mv)
