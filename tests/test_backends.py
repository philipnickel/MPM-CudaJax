import hydra
import pytest
from omegaconf import OmegaConf

from mpm_jax.backends import (
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
    assert callable(backend.p2g) and callable(backend.g2p)


def test_cuda_backend_constructors_register_expected_kind(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "mpm_jax.backends.cuda.register_p2g_inline",
        lambda: calls.append("cuda_v1"),
    )
    monkeypatch.setattr(
        "mpm_jax.backends.cuda.register_p2g_v2_inline",
        lambda: calls.append("cuda_v2"),
    )
    monkeypatch.setattr(
        "mpm_jax.backends.cuda.register_p2g_v3_inline",
        lambda: calls.append("cuda_v3"),
    )
    monkeypatch.setattr(
        "mpm_jax.backends.cuda.register_p2g_v4_inline",
        lambda: calls.append("cuda_v4"),
    )

    CudaV1Backend(num_grids=16)
    CudaV2Backend(num_grids=16)
    CudaV3Backend(num_grids=16)
    CudaV4Backend(num_grids=16)

    assert calls == ["cuda_v1", "cuda_v2", "cuda_v3", "cuda_v4"]


def test_supercell_backend_validates_num_grids():
    with pytest.raises(RuntimeError, match="requires num_grids"):
        CudaV4Backend(num_grids=18, super_cell_width=4)


def test_hydra_instantiates_runtime_config_and_solver():
    cfg = OmegaConf.create(
        {
            "backend": {
                "_target_": "mpm_jax.backends.jax.JaxBackend",
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
                "boundary": {
                    "_target_": "mpm_jax.boundary.StickyPlane",
                    "point": [1.0, 1.0, 0.02],
                    "normal": [0.0, 0.0, 1.0],
                    "start_time": 0.0,
                    "end_time": 1e3,
                },
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
    solver.step()
