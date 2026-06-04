import jax.numpy as jnp
import jax
from mpm_jax.types import MPMState, make_params
from mpm_jax.backends import build_backend_frame, jax_baseline_backend
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns

def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing='ij')
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

def test_sand_simulation_10_frames():
    from omegaconf import OmegaConf
    N = 100
    num_grids = 15
    x0 = jnp.ones((N, 3)) * 0.5
    params = make_params(n_particles=N, num_grids=num_grids, dt=3e-4)
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {"type": "surface_collider", "point": [1.0, 1.0, 0.02],
         "normal": [0.0, 0.0, 1.0], "surface": "sticky",
         "start_time": 0.0, "end_time": 1e3},
    ]
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx, x0, params.dt)
    elasticity_fn = get_constitutive(OmegaConf.create(
        {"name": "StVKElasticityJacobi", "E": 2e6, "nu": 0.4}))
    plasticity_fn = get_constitutive(OmegaConf.create({
        "name": "DruckerPragerPlasticityJacobi",
        "E": 2e6,
        "nu": 0.4,
        "friction_angle": 25.0,
        "cohesion": 0.0,
    }))
    state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    frame = build_backend_frame(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
        jax_baseline_backend(), steps_per_frame=5)
    for _ in range(10):
        state = frame(state)
    jax.block_until_ready(state.x)
    assert jnp.mean(state.x[:, 2]) < 0.5
    assert jnp.all(jnp.isfinite(state.x))


def test_outer_frame_jit_runs_multiple_frames():
    from omegaconf import OmegaConf
    N = 200
    num_grids = 16
    steps_per_frame = 4
    x0 = jnp.ones((N, 3), dtype=jnp.float32) * 0.5

    params = make_params(n_particles=N, num_grids=num_grids, dt=3e-4)
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {"type": "surface_collider", "point": [1.0, 1.0, 0.02],
         "normal": [0.0, 0.0, 1.0], "surface": "sticky",
         "start_time": 0.0, "end_time": 1e3},
    ]
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx, x0, params.dt)
    elasticity_fn = get_constitutive(OmegaConf.create(
        {"name": "StVKElasticityJacobi", "E": 2e6, "nu": 0.4}))
    plasticity_fn = get_constitutive(OmegaConf.create({
        "name": "DruckerPragerPlasticityJacobi",
        "E": 2e6,
        "nu": 0.4,
        "friction_angle": 25.0,
        "cohesion": 0.0,
    }))

    state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    jit_frame = build_backend_frame(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
        jax_baseline_backend(), steps_per_frame)
    state = jit_frame(state)
    state = jit_frame(state)
    jax.block_until_ready(state.x)

    assert state.x.shape == (N, 3)
    assert state.v.shape == (N, 3)
    assert state.F.shape == (N, 3, 3)
    assert jnp.all(jnp.isfinite(state.x))
