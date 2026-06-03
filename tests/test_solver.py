import jax.numpy as jnp
import numpy as np
from mpm_jax.types import MPMState, MPMParams, make_params
from mpm_jax.blocks.grid import grid_update

def test_mpm_state_is_namedtuple():
    N = 10
    state = MPMState(
        x=jnp.zeros((N, 3)),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    assert state.x.shape == (N, 3)
    assert state.F.shape == (N, 3, 3)

def test_make_params():
    N = 1000
    params = make_params(
        n_particles=N,
        num_grids=25,
        dt=3e-4,
        gravity=[0.0, 0.0, -9.8],
        rho=1000.0,
        clip_bound=0.5,
        damping=1.0,
        center=[0.5, 0.5, 0.5],
        size=[1.0, 1.0, 1.0],
    )
    assert params.num_grids == 25
    assert params.dx == 1.0 / 25
    assert params.inv_dx == 25.0
    assert params.clip_bound == 0.5 / 25
    expected_vol = 1.0 / N
    assert np.isclose(params.vol, expected_vol)
    assert np.isclose(params.p_mass, 1000.0 * expected_vol)
    assert params.gravity.shape == (3,)

def test_grid_update_divides_momentum_by_mass():
    grid_mv = jnp.array([[3.0, 6.0, 9.0], [0.0, 0.0, 0.0]])
    grid_m = jnp.array([3.0, 0.0])
    gravity = jnp.array([0.0, 0.0, -9.8])
    result = grid_update(grid_mv, grid_m, gravity, dt=1.0, damping=1.0)
    assert jnp.allclose(result[0], jnp.array([1.0, 2.0, -6.8]), atol=1e-5)
    assert jnp.allclose(result[1], jnp.array([0.0, 0.0, -9.8]), atol=1e-5)
