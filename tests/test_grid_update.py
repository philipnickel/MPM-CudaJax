import jax.numpy as jnp
from mpm_jax.state import make_params
from mpm_jax.grid_update import grid_update


def test_grid_update_normalizes_momentum():
    """momentum / mass = velocity, then apply gravity."""
    params = make_params(n_particles=100, num_grids=4, dt=1e-3)
    G = 4
    grid_mv = jnp.ones((G**3, 3))
    grid_m = jnp.full((G**3,), 2.0)

    grid_v = grid_update(grid_mv, grid_m, params)

    # v = damping * (mv/m + gravity*dt) = 1.0 * (0.5 + gravity*dt)
    expected_z = 0.5 + params.gravity[2] * params.dt
    assert grid_v.shape == (G**3, 3)
    assert jnp.allclose(grid_v[:, 2], expected_z, atol=1e-5)


def test_grid_update_zero_mass_keeps_momentum():
    """Zero-mass cells keep original momentum (not zeroed), then damping+gravity applied."""
    params = make_params(n_particles=100, num_grids=4, dt=1e-3)
    G = 4
    grid_mv = jnp.ones((G**3, 3))
    grid_m = jnp.zeros((G**3,))

    grid_v = grid_update(grid_mv, grid_m, params)
    # Invalid cells keep grid_mv, then damping*(grid_mv + dt*gravity) is applied
    expected_z = params.damping * (1.0 + params.dt * params.gravity[2])
    assert jnp.allclose(grid_v[:, 2], expected_z, atol=1e-5)
