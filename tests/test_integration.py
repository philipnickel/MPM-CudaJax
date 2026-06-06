import jax.numpy as jnp
import jax
from omegaconf import OmegaConf
from mpm_jax.types import MPMState, MPMParams
from mpm_jax.backends import JaxBackend
from mpm_jax.solver import build_backend_frame
from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.boundary import build_boundary_fns


def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _runtime_params(n_particles, num_grids, dt=3e-4):
    return MPMParams(
        OmegaConf.create(
            {
                "n_particles": n_particles,
                "num_grids": num_grids,
                "dt": dt,
                "gravity": [0.0, 0.0, -9.8],
                "rho": 1000.0,
                "clip_bound": 0.5,
                "damping": 1.0,
                "size": [1.0, 1.0, 1.0],
            }
        )
    )


def test_elastic_simulation_10_frames():
    N = 100
    num_grids = 15
    x0 = jnp.ones((N, 3)) * 0.5
    params = _runtime_params(n_particles=N, num_grids=num_grids, dt=3e-4)
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
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx)
    elasticity_fn = stvk_elasticity_jacobi(E=2e6, nu=0.4)
    state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    frame = build_backend_frame(
        params,
        elasticity_fn,
        pre_fn,
        post_fn,
        JaxBackend(),
        steps_per_frame=5,
    )
    for _ in range(10):
        state = frame(state)
    jax.block_until_ready(state.x)
    assert jnp.mean(state.x[:, 2]) < 0.5
    assert jnp.all(jnp.isfinite(state.x))


def test_outer_frame_jit_runs_multiple_frames():
    N = 200
    num_grids = 16
    steps_per_frame = 4
    x0 = jnp.ones((N, 3), dtype=jnp.float32) * 0.5

    params = _runtime_params(n_particles=N, num_grids=num_grids, dt=3e-4)
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
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx)
    elasticity_fn = stvk_elasticity_jacobi(E=2e6, nu=0.4)

    state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    jit_frame = build_backend_frame(
        params,
        elasticity_fn,
        pre_fn,
        post_fn,
        JaxBackend(),
        steps_per_frame,
    )
    state = jit_frame(state)
    state = jit_frame(state)
    jax.block_until_ready(state.x)

    assert state.x.shape == (N, 3)
    assert state.v.shape == (N, 3)
    assert state.F.shape == (N, 3, 3)
    assert jnp.all(jnp.isfinite(state.x))
