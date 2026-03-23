import jax
import jax.numpy as jnp
import numpy as np
from mpm_jax.solver import MPMState, MPMParams, make_params
from mpm_jax.solver import grid_update, step

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

def _noop_pre_particle(x, v, time):
    return x, v

def _noop_post_grid(grid_mv, grid_m, time):
    return grid_mv

def test_grid_update_divides_momentum_by_mass():
    grid_mv = jnp.array([[3.0, 6.0, 9.0], [0.0, 0.0, 0.0]])
    grid_m = jnp.array([3.0, 0.0])
    gravity = jnp.array([0.0, 0.0, -9.8])
    result = grid_update(grid_mv, grid_m, gravity, dt=1.0, damping=1.0)
    assert jnp.allclose(result[0], jnp.array([1.0, 2.0, -6.8]), atol=1e-5)
    assert jnp.allclose(result[1], jnp.array([0.0, 0.0, -9.8]), atol=1e-5)

def test_step_shapes_and_runs():
    N = 100
    params = make_params(n_particles=N, num_grids=10)
    state = MPMState(
        x=jnp.ones((N, 3)) * 0.5,
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    stress = jnp.zeros((N, 3, 3))
    new_state = step(params, state, stress, _noop_pre_particle, _noop_post_grid, 0.0)
    assert new_state.x.shape == (N, 3)
    assert new_state.v.shape == (N, 3)
    assert new_state.C.shape == (N, 3, 3)
    assert new_state.F.shape == (N, 3, 3)
    assert not jnp.allclose(new_state.v, 0.0)

def test_step_gravity_pulls_down():
    N = 50
    params = make_params(n_particles=N, num_grids=10, gravity=[0.0, 0.0, -9.8])
    state = MPMState(
        x=jnp.ones((N, 3)) * 0.5,
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    stress = jnp.zeros((N, 3, 3))
    new_state = step(params, state, stress, _noop_pre_particle, _noop_post_grid, 0.0)
    assert jnp.all(new_state.v[:, 2] < 0)

from mpm_jax.solver import simulate_frame

def test_simulate_frame_runs_multiple_substeps():
    N = 50
    params = make_params(n_particles=N, num_grids=10)
    state = MPMState(
        x=jnp.ones((N, 3)) * 0.5,
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    elasticity_fn = lambda F: jnp.zeros_like(F)
    plasticity_fn = lambda F: F

    new_state, new_time = simulate_frame(
        params, state, elasticity_fn, plasticity_fn,
        _noop_pre_particle, _noop_post_grid,
        steps_per_frame=5, time=0.0,
    )
    assert new_state.x.shape == (N, 3)
    assert jnp.isclose(new_time, 5 * params.dt)
    assert jnp.all(new_state.x[:, 2] < 0.5)
