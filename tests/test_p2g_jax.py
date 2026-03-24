import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g


def _make_state(N, params):
    return MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )


def test_jax_p2g_returns_correct_shapes():
    N = 50
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "IdentityPlasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)
    grid_mv, grid_m = p2g_fn(state, params)
    G = params.num_grids
    assert grid_mv.shape == (G**3, 3)
    assert grid_m.shape == (G**3,)


def test_jax_p2g_nonzero_mass():
    N = 50
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "IdentityPlasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)
    grid_mv, grid_m = p2g_fn(state, params)
    assert jnp.any(grid_m > 0)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))


def test_jax_p2g_mass_conservation():
    N = 100
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "IdentityPlasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)
    grid_mv, grid_m = p2g_fn(state, params)
    total_mass = jnp.sum(grid_m)
    expected_mass = N * params.p_mass
    assert jnp.allclose(total_mass, expected_mass, rtol=1e-4)
