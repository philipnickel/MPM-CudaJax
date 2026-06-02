import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.types import MPMState, make_params
from mpm_jax.solver import MPMSolver
from mpm_jax.stepping.jax_frames import build_jax_frame
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.blocks.grid import build_grid_x


def _make_solver(steps_per_frame=2, n=64, G=16):
    params = make_params(n_particles=n, num_grids=G, dt=3e-4)
    grid_x = build_grid_x(G)
    x = jnp.array([[0.5, 0.5, 0.5]] * n, dtype=jnp.float32)
    pre_fn, post_fn = build_boundary_fns([], grid_x, params.dx, x, params.dt, params.p_mass)
    elasticity = get_constitutive(OmegaConf.create({"name": "CorotatedElasticity"}))
    plasticity = get_constitutive(OmegaConf.create({"name": "IdentityPlasticity"}))
    init = MPMState(x=x, v=jnp.zeros((n, 3)), C=jnp.zeros((n, 3, 3)),
                    F=jnp.broadcast_to(jnp.eye(3), (n, 3, 3)).copy())
    return MPMSolver(params, elasticity_fn=elasticity, plasticity_fn=plasticity,
                     pre_fn=pre_fn, post_fn=post_fn, build_frame=build_jax_frame,
                     steps_per_frame=steps_per_frame, init_state=init)


def test_step_returns_and_mutates_state():
    s = _make_solver()
    x0 = s.state.x
    out = s.step()
    assert out is s.state
    assert s.state.x.shape == x0.shape


def test_solve_equals_n_steps_and_fires_hook():
    s = _make_solver()
    calls = []
    s.solve(3, on_frame=lambda i, st: calls.append(i))
    assert calls == [0, 1, 2]


def test_reset_restores_state():
    s = _make_solver()
    init = s.state
    s.step()
    s.reset_to_initial()
    assert s.state is init


import numpy as np
import pytest


def _warp_available():
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


@pytest.mark.skipif(not _warp_available(), reason="needs Warp + CUDA")
def test_warp_graph_solver_interface():
    from mpm_jax.solver import WarpGraphSolver
    from mpm_jax.stepping.warp_graph_frame import build_warp_graph
    cfg = OmegaConf.create({
        "sim": {"n_particles": 4096, "num_grids": 32, "dt": 3e-4,
                "steps_per_frame": 2, "clip_bound": 0.5, "damping": 1.0,
                "gravity": [0, 0, -9.8], "rho": 1000.0, "size": [0.5, 0.5, 0.5],
                "initial_velocity": [0, 0, 0], "center": [0.5, 0.5, 0.5]},
        "material": {"elasticity": {"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4},
                     "plasticity": {"name": "IdentityPlasticity"}},
    })
    particles = np.random.RandomState(0).uniform(0.3, 0.7, size=(4096, 3)).astype(np.float32)
    solver = build_warp_graph(cfg, particles=particles)
    assert isinstance(solver, WarpGraphSolver)
    solver.step()
    solver.solve(2)
