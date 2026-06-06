from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from mpm_jax.backends import JaxBackend
from mpm_jax.backends.cuda import CudaV4Backend
from mpm_jax.boundary import StickyPlane
from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.solver import MPMSolver, RuntimeConfig


def _boundary():
    return StickyPlane(
        point=(1.0, 1.0, 0.02),
        normal=(0.0, 0.0, 1.0),
        start_time=0.0,
        end_time=1e3,
    )


def _make_solver(steps_per_frame=2, n=64, G=16):
    sim = SimpleNamespace(
        n_particles=n,
        num_grids=G,
        dt=3e-4,
        steps_per_frame=steps_per_frame,
        clip_bound=0.5,
        damping=1.0,
        gravity=[0.0, 0.0, -9.8],
        rho=1000.0,
        size=[0.5, 0.5, 0.5],
        initial_velocity=[0.0, 0.0, 0.0],
        center=[0.5, 0.5, 0.5],
        boundary=_boundary(),
    )
    material = SimpleNamespace(elasticity=stvk_elasticity_jacobi())
    return MPMSolver(
        RuntimeConfig(material=material, sim=sim, backend=JaxBackend())
    )


def test_solver_validates_backend_against_runtime_num_grids(monkeypatch):
    monkeypatch.setattr("mpm_jax.backends.cuda.register_p2g_v4_inline", lambda: True)

    sim = SimpleNamespace(
        n_particles=64,
        num_grids=18,
        dt=3e-4,
        steps_per_frame=1,
        clip_bound=0.5,
        damping=1.0,
        gravity=[0.0, 0.0, -9.8],
        rho=1000.0,
        size=[0.5, 0.5, 0.5],
        initial_velocity=[0.0, 0.0, 0.0],
        center=[0.5, 0.5, 0.5],
        boundary=_boundary(),
    )
    material = SimpleNamespace(elasticity=stvk_elasticity_jacobi())
    backend = CudaV4Backend(num_grids=None, super_cell_width=4)

    with pytest.raises(RuntimeError, match="requires num_grids"):
        MPMSolver(RuntimeConfig(material=material, sim=sim, backend=backend))


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
    assert s.backend.name == "jax"
    assert callable(s.elasticity_fn)
    assert s.state.x is not None and s.state.v is not None
