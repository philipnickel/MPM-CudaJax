import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p


def _make_cfg():
    return OmegaConf.create({
        "material": {
            "elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "IdentityPlasticity"},
        },
        "kernel": {"name": "jax"},
    })


def test_full_timestep_jax():
    """One full P2G -> grid_update -> G2P cycle produces valid state."""
    N = 100
    params = make_params(n_particles=N, num_grids=10, dt=3e-4)
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    cfg = _make_cfg()
    p2g_fn = make_jax_p2g(cfg)

    grid_mv, grid_m = p2g_fn(state, params)
    grid_v = grid_update(grid_mv, grid_m, params)
    new_state = g2p(state, grid_v, params)

    assert jnp.all(jnp.isfinite(new_state.x))
    assert jnp.all(jnp.isfinite(new_state.v))
    assert jnp.all(jnp.isfinite(new_state.F))


def test_jelly_10_frames():
    """10 frames of jelly simulation: particles should fall under gravity."""
    N = 200
    params = make_params(n_particles=N, num_grids=15, dt=3e-4)
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    cfg = _make_cfg()
    p2g_fn = make_jax_p2g(cfg)

    initial_z = state.x[:, 2].mean()

    for frame in range(10):
        for step in range(10):
            grid_mv, grid_m = p2g_fn(state, params)
            grid_v = grid_update(grid_mv, grid_m, params)
            state = g2p(state, grid_v, params)

    # Particles should have fallen
    final_z = state.x[:, 2].mean()
    assert final_z < initial_z
    assert jnp.all(jnp.isfinite(state.x))
    assert jnp.all(jnp.isfinite(state.v))
