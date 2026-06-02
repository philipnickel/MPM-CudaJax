from mpm_jax.registry import KERNELS, REMOVED_KERNELS, KernelSpec


def test_every_kernel_has_a_spec():
    expected = {
        "jax", "jax_v1_5",
        "cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline",
        "warp_v1_inline", "warp_v3_supercell_tile",
        "warp_bonus_graph", "warp_bonus_v2_graph",
    }
    assert set(KERNELS) == expected
    for spec in KERNELS.values():
        assert isinstance(spec, KernelSpec)
        assert spec.solver_cls is not None
        assert callable(spec.build_frame)


def test_removed_kernels_listed():
    for name in ("cuda_v1", "cuda_v2", "cuda_v4", "cuda_fused",
                 "cuda_v2_fori_inline", "cuda_v3_fori_inline", "cuda_v6_inline"):
        assert name in REMOVED_KERNELS


def test_build_solver_dispatches_jax_kernel():
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import MPMSolver
    cfg = OmegaConf.create({
        "kernel": {"name": "jax"},
        "sim": {"n_particles": 64, "num_grids": 16, "dt": 3e-4, "steps_per_frame": 2,
                "clip_bound": 0.5, "damping": 1.0, "gravity": [0, 0, -9.8], "rho": 1000.0,
                "size": [0.5, 0.5, 0.5], "initial_velocity": [0, 0, 0],
                "center": [0.5, 0.5, 0.5], "boundary_conditions": []},
        "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                     "plasticity": {"name": "IdentityPlasticity"}},
    })
    solver = build_solver(cfg)
    assert isinstance(solver, MPMSolver)
    solver.step()


def test_build_solver_rejects_removed_kernel():
    import pytest
    from omegaconf import OmegaConf
    from mpm_jax.registry import build_solver
    cfg = OmegaConf.create({"kernel": {"name": "cuda_v6_inline"}, "sim": {}, "material": {}})
    with pytest.raises(ValueError, match="cuda_v3_inline"):
        build_solver(cfg)
