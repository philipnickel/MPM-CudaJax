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
    assert not jnp.array_equal(s.state.x, x0)


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

