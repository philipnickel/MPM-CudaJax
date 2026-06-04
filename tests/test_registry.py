from mpm_jax.registry import KERNELS, KernelSpec


def test_every_kernel_has_a_spec():
    expected = {
        "jax_v1_5", "jax_v2", "jax_v3", "jax_cuda_g2p",
        "cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline",
        "warp_v3_supercell_tile",
    }
    assert set(KERNELS) == expected
    for spec in KERNELS.values():
        assert isinstance(spec, KernelSpec)
        assert spec.solver_cls is not None
        assert callable(spec.backend_factory)


def test_build_solver_dispatches_jax_baseline_kernel():
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import MPMSolver
    cfg = OmegaConf.create({
        "kernel": {"name": "jax_v1_5"},
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
        "kernel": {"name": "jax_v1_5"},
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
