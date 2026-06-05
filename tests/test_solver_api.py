import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.solver import MPMSolver, RuntimeConfig
from mpm_jax.backends import Backend


def _make_solver(steps_per_frame=2, n=64, G=16):
    sim = OmegaConf.create(
        {
            "n_particles": n,
            "num_grids": G,
            "dt": 3e-4,
            "steps_per_frame": steps_per_frame,
            "clip_bound": 0.5,
            "damping": 1.0,
            "gravity": [0.0, 0.0, -9.8],
            "rho": 1000.0,
            "size": [0.5, 0.5, 0.5],
            "initial_velocity": [0.0, 0.0, 0.0],
            "center": [0.5, 0.5, 0.5],
            "boundary_conditions": [],
        }
    )
    material = OmegaConf.create(
        {
            "elasticity": {"name": "StVKElasticityJacobi"},
        }
    )
    return MPMSolver(
        RuntimeConfig(material=material, sim=sim, backend=Backend())
    )


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


def test_stepped_returns_new_solver():
    s = _make_solver()
    out = s.stepped()
    assert isinstance(out, MPMSolver)
    assert out is not s
    # shallow copy shares the static parts; only state is replaced
    assert out.backend is s.backend
    assert out._frame is s._frame
    assert out.state is not s.state
    assert out.state.x.shape == s.state.x.shape
    assert not jnp.array_equal(out.state.x, s.state.x)


def test_solver_exposes_backend_and_constitutive():
    s = _make_solver()
    assert s.backend.name == "jax_baseline"
    assert callable(s.elasticity_fn)
    assert s.state.x is not None and s.state.v is not None
