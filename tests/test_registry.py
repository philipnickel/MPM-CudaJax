from mpm_jax.registry import KERNEL_NAMES, build_backend


def test_kernel_names():
    assert set(KERNEL_NAMES) == {
        "jax_baseline",
        "cutile_v6_atomic_tile",
        "cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline",
    }


def test_build_backend_jax_baseline_no_gpu():
    # jax_baseline has no availability/super-cell requirement, so it builds on CPU.
    from mpm_jax.backends import Backend
    backend = build_backend("jax_baseline", 16)
    assert isinstance(backend, Backend)
    assert backend.name == "jax_baseline"
    assert callable(backend.p2g) and callable(backend.g2p)


def test_build_backend_rejects_unknown():
    import pytest
    with pytest.raises(KeyError):
        build_backend("not_a_kernel", 16)


def test_build_solver_dispatches_jax_baseline_kernel():
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import MPMSolver
    cfg = OmegaConf.create({
        "kernel": {"name": "jax_baseline"},
        "sim": {"n_particles": 64, "num_grids": 16, "dt": 3e-4, "steps_per_frame": 2,
                "clip_bound": 0.5, "damping": 1.0, "gravity": [0, 0, -9.8], "rho": 1000.0,
                "size": [0.5, 0.5, 0.5], "initial_velocity": [0, 0, 0],
                "center": [0.5, 0.5, 0.5], "boundary_conditions": []},
        "material": {
            "elasticity": {"name": "StVKElasticityJacobi", "E": 2e6, "nu": 0.4},
            "plasticity": {
                "name": "DruckerPragerPlasticityJacobi",
                "E": 2e6,
                "nu": 0.4,
                "friction_angle": 25.0,
                "cohesion": 0.0,
            },
        },
    })
    solver = build_solver(cfg)
    assert isinstance(solver, MPMSolver)
    solver.step()


def test_build_solver_rejects_unknown_kernel():
    import pytest
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    cfg = OmegaConf.create({"kernel": {"name": "unknown_kernel"}, "sim": {}, "material": {}})
    with pytest.raises(KeyError):
        build_solver(cfg)


def test_build_solver_dispatches_sand_jacobi_on_jax_baseline():
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import MPMSolver
    cfg = OmegaConf.create({
        "kernel": {"name": "jax_baseline"},
        "sim": {"n_particles": 64, "num_grids": 16, "dt": 3e-4, "steps_per_frame": 1,
                "clip_bound": 0.5, "damping": 1.0, "gravity": [0, 0, -9.8], "rho": 1000.0,
                "size": [0.5, 0.5, 0.5], "initial_velocity": [0, 0, 0],
                "center": [0.5, 0.5, 0.5], "boundary_conditions": []},
        "material": {
            "elasticity": {"name": "StVKElasticityJacobi", "E": 2e6, "nu": 0.4},
            "plasticity": {
                "name": "DruckerPragerPlasticityJacobi",
                "E": 2e6,
                "nu": 0.4,
                "friction_angle": 25.0,
                "cohesion": 0.0,
            },
        },
    })
    solver = build_solver(cfg)
    assert isinstance(solver, MPMSolver)
    solver.step()
