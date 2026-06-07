from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from mpm_jax.p2g.backends import JaxBackend
from mpm_jax.p2g.backends.cuda import CudaV4Backend
from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.solver import MPMSolver, RuntimeConfig


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
    )
    material = SimpleNamespace(elasticity=stvk_elasticity_jacobi())
    return MPMSolver(RuntimeConfig(material=material, sim=sim, backend=JaxBackend()))


def test_solver_validates_backend_against_runtime_num_grids(monkeypatch):
    monkeypatch.setattr("mpm_jax.p2g.cuda.p2g_cuda.CudaV4P2G.register", lambda self: True)

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


def test_step_matches_frame_when_frame_has_one_substep():
    staged = _make_solver(steps_per_frame=1)
    framed = _make_solver(steps_per_frame=1)

    staged.step()
    framed.state = framed._frame(framed.state)

    for staged_leaf, framed_leaf in zip(staged.state, framed.state, strict=True):
        assert jnp.allclose(staged_leaf, framed_leaf, rtol=1e-5, atol=1e-6)


def test_run_uses_configured_frames_and_can_capture_initial_states():
    s = _make_solver(steps_per_frame=1)
    s.num_frames = 2

    frames, elapsed = s.run(capture_frames=True, progress=False)

    assert len(frames) == 2
    assert frames[0].shape == (64, 3)
    assert elapsed >= 0


def test_run_without_capture_returns_no_frames():
    s = _make_solver(steps_per_frame=1)
    s.num_frames = 2

    frames, elapsed = s.run(capture_frames=False, progress=False)

    assert frames == []
    assert elapsed >= 0


def test_run_can_use_staged_steps():
    s = _make_solver(steps_per_frame=1)
    s.num_frames = 1

    frames, elapsed = s.run(capture_frames=False, progress=False, staged=True)

    assert frames == []
    assert elapsed >= 0


def test_run_can_reuse_explicit_warmup():
    s = _make_solver(steps_per_frame=1)
    s.num_frames = 1

    s.warmup()
    frames, elapsed = s.run(capture_frames=False, progress=False)

    assert frames == []
    assert elapsed >= 0


def test_warmup_can_compile_staged_steps():
    s = _make_solver(steps_per_frame=1)
    init = s.state

    out = s.warmup(staged=True)

    assert out is init
    assert s.state is init


def test_solver_exposes_backend_and_constitutive():
    s = _make_solver()
    assert s.backend.name == "jax"
    assert callable(s.elasticity_fn)
    assert s.state.x is not None and s.state.v is not None
