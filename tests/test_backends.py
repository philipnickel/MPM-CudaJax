import hydra
from omegaconf import OmegaConf

from mpm_jax.p2g.backends import (
    CudaV1Backend,
    CudaV2Backend,
    CudaV3Backend,
    CudaV4Backend,
    JaxBackend,
)
from mpm_jax.solver import MPMSolver


def test_jax_backend_constructs_without_gpu():
    backend = JaxBackend(num_grids=16)
    assert backend.name == "jax"
    assert callable(backend.prepare) and callable(backend.scatter)


def test_cuda_backend_constructors_register_expected_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV1P2G.register",
        lambda self: calls.append("cuda_v1"),
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV2P2G.register",
        lambda self: calls.append("cuda_v2"),
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV3P2G.register",
        lambda self: calls.append("cuda_v3"),
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV4P2G.register",
        lambda self: calls.append("cuda_v4"),
    )

    CudaV1Backend(num_grids=16)
    CudaV2Backend(num_grids=16)
    CudaV3Backend(num_grids=16)
    CudaV4Backend(num_grids=16)

    assert calls == ["cuda_v1", "cuda_v2", "cuda_v3", "cuda_v4"]


def test_cuda_progression_backends_do_not_require_grid_divisors(monkeypatch):
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV1P2G.register", lambda self: True
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV2P2G.register", lambda self: True
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV3P2G.register", lambda self: True
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV4P2G.register", lambda self: True
    )

    for backend in (
        CudaV1Backend(num_grids=18),
        CudaV2Backend(num_grids=18),
        CudaV3Backend(num_grids=18),
        CudaV4Backend(num_grids=18),
    ):
        assert backend.grid_divisor() is None


def test_hydra_instantiates_runtime_config_and_solver():
    cfg = OmegaConf.create(
        {
            "backend": {
                "_target_": "mpm_jax.p2g.backends.jax.JaxBackend",
                "num_grids": 16,
            },
            "sim": {
                "n_particles": 64,
                "num_grids": 16,
                "dt": 3e-4,
                "steps_per_frame": 1,
                "clip_bound": 0.5,
                "damping": 1.0,
                "gravity": [0, 0, -9.8],
                "rho": 1000.0,
                "size": [0.5, 0.5, 0.5],
                "initial_velocity": [0, 0, 0],
                "center": [0.5, 0.5, 0.5],
            },
            "material": {
                "elasticity": {
                    "_target_": "mpm_jax.constitutive.stvk_elasticity_jacobi",
                    "E": 2e6,
                    "nu": 0.4,
                },
            },
        }
    )
    cfg.solver = {
        "_target_": "mpm_jax.solver.RuntimeConfig",
        "_recursive_": False,
        "material": cfg.material,
        "sim": cfg.sim,
        "backend": cfg.backend,
    }
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    assert isinstance(solver, MPMSolver)
    assert solver.backend.name == "jax"
    solver.run(capture_frames=False, progress=False)
