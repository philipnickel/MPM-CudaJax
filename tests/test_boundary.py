import jax.numpy as jnp
import pytest

from mpm_jax.boundary import build_boundary_fns


def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def test_sticky_boundary_zeroes_velocity_below_surface():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {
            "point": [1.0, 1.0, 0.02],
            "normal": [0.0, 0.0, 1.0],
            "start_time": 0.0,
            "end_time": 1e3,
        },
    ]
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, dx)
    grid_mv = jnp.ones((num_grids**3, 3))
    grid_m = jnp.ones((num_grids**3,))
    result = post_fn(grid_mv, grid_m, 0.0)
    point = jnp.array([1.0, 1.0, 0.02])
    normal = jnp.array([0.0, 0.0, 1.0])
    offset = grid_x * dx - point
    below = jnp.sum(offset * normal, axis=1) < 0
    assert jnp.allclose(result[below], 0.0)
    assert jnp.allclose(result[~below], 1.0)


def test_noop_when_no_bcs():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    pre_fn, post_fn = build_boundary_fns([], grid_x, dx)
    x = jnp.ones((10, 3)) * 0.5
    v = jnp.ones((10, 3))
    x2, v2 = pre_fn(x, v, 0.0)
    assert jnp.allclose(x2, x)
    assert jnp.allclose(v2, v)


def test_rejects_stale_boundary_choice_fields():
    num_grids = 5
    dx = 1.0 / num_grids
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {
            "type": "surface_collider",
            "point": [1.0, 1.0, 0.02],
            "normal": [0.0, 0.0, 1.0],
            "surface": "sticky",
            "start_time": 0.0,
            "end_time": 1e3,
        },
    ]
    with pytest.raises(ValueError, match="unexpected surface, type"):
        build_boundary_fns(bc_configs, grid_x, dx)
