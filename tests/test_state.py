import jax.numpy as jnp
import numpy as np
from mpm_jax.state import MPMState, MPMParams, make_params


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
    p = make_params(n_particles=100, num_grids=25)
    assert p.dx == 1.0 / 25
    assert p.inv_dx == 25.0
    assert p.vol > 0
    assert p.p_mass > 0
    assert p.n_particles == 100
    # clip_bound is scaled by dx
    assert p.clip_bound == 0.5 * p.dx


def test_make_params_volume_formula():
    """vol = prod(size) / n_particles — matches original solver."""
    p = make_params(n_particles=100, size=[1.0, 1.0, 1.0])
    assert np.isclose(p.vol, 1.0 / 100)
