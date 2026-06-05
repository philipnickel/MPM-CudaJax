from mpm_jax.backends import KERNEL_NAMES, build_backend


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


def test_hydra_instantiates_runtime_config_and_solver():
    import hydra
    from omegaconf import OmegaConf
    from mpm_jax.solver import MPMSolver

    cfg = OmegaConf.create({
        "backend": {
            "_target_": "mpm_jax.backends.Backend",
            "num_grids": 16,
        },
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
    cfg.solver = {
        "_target_": "mpm_jax.solver.RuntimeConfig",
        "material": cfg.material,
        "sim": cfg.sim,
        "backend": cfg.backend,
    }
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    assert isinstance(solver, MPMSolver)
    assert solver.backend.name == "jax_baseline"
    solver.step()
