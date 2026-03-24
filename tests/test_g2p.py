import jax.numpy as jnp
from mpm_jax.state import MPMState, make_params
from mpm_jax.g2p import g2p


def test_g2p_returns_mpm_state():
    N = 10
    params = make_params(n_particles=N, num_grids=4)
    G = params.num_grids
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    grid_v = jnp.zeros((G**3, 3))

    new_state = g2p(state, grid_v, params)

    assert isinstance(new_state, MPMState)
    assert new_state.x.shape == (N, 3)
    assert new_state.v.shape == (N, 3)
    assert new_state.C.shape == (N, 3, 3)
    assert new_state.F.shape == (N, 3, 3)


def test_g2p_updates_position():
    """With nonzero grid velocity, particles should move."""
    N = 5
    params = make_params(n_particles=N, num_grids=4, dt=1e-2)
    G = params.num_grids
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    # Uniform downward grid velocity
    grid_v = jnp.zeros((G**3, 3)).at[:, 2].set(-1.0)

    new_state = g2p(state, grid_v, params)

    # Particles should have moved down
    assert jnp.all(new_state.x[:, 2] < 0.5)
