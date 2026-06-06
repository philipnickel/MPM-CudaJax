from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import profile_nsight
from mpm_jax.solver import MPMSolver


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_config(config_name="config", overrides=None):
    overrides = [
        "backend=jax",
        "sim.n_particles=8",
        "sim.num_grids=16",
        "sim.steps_per_frame=1",
        "sim.num_frames=1",
        *(overrides or []),
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides)
    if "nsight" not in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg.nsight = OmegaConf.create(
            {
                "phase": "p2g",
                "write_json": False,
                "plot": {"enabled": False},
                "sweep": None,
                "configs": None,
                "analyze": {},
            }
        )
    return cfg


def test_backend_choices_come_from_registered_hydra_configs():
    assert set(profile_nsight._backend_choices()) == {
        "jax",
        "cuda_v1",
        "cuda_v2",
        "cuda_v3",
        "cuda_v4",
        "cutile_v1",
        "cutile_v2",
    }


def test_nsight_profile_config_composes():
    cfg = _compose_config("nsight_profile")

    assert cfg.nsight.phase == "p2g"
    assert cfg.backend._target_ == "mpm_jax.backends.jax.JaxBackend"
    assert profile_nsight._nsight_configs(cfg) == [("jax", 8, 16, 1)]


def test_p2g_stage_reuses_solver_state_without_extra_particle_prepass():
    cfg = _compose_config()

    jit_stage, state, backend_name = profile_nsight._build_p2g_stage(cfg)
    grid_mv, grid_m = jit_stage(state)

    assert backend_name == "jax"
    assert grid_mv.shape == (16**3, 3)
    assert grid_m.shape == (16**3,)


def test_sweep_backend_choices_are_hydra_choices():
    cfg = _compose_config()
    cfg.nsight.sweep = {
        "kernels": ["jax", " cuda_v3"],
        "n_particles": [8, 12],
        "num_grids": [16],
        "steps_per_frame": [1],
    }

    assert profile_nsight._sweep_backend_choices(cfg) == ["jax", "cuda_v3"]
    assert profile_nsight._nsight_configs(cfg) == [
        ("jax", 8, 16, 1),
        ("jax", 12, 16, 1),
        ("cuda_v3", 8, 16, 1),
        ("cuda_v3", 12, 16, 1),
    ]


def test_runtime_backend_names_are_rejected_as_profile_choices():
    cfg = _compose_config()
    cfg.nsight.sweep = {"kernels": ["cuda_v3_inline"]}

    with pytest.raises(RuntimeError, match="Hydra backend choices"):
        profile_nsight._sweep_backend_choices(cfg)


def test_nsight_configs_are_exact_backend_variants():
    cfg = _compose_config()
    cfg.nsight.configs = [
        {
            "backend": "cuda_v2",
            "sim": {"n_particles": 12, "num_grids": 40, "steps_per_frame": 2},
        },
        {"sim": {"n_particles": 10}},
    ]

    assert profile_nsight._nsight_configs(cfg) == [
        ("cuda_v2", 12, 40, 2),
        ("jax", 10, 16, 1),
    ]


def test_merge_variant_cfg_loads_registered_backend_config_into_solver_reference():
    cfg = _compose_config()

    variant = profile_nsight._merge_variant_cfg(
        cfg,
        backend_choice="cuda_v3",
        n_particles=12,
        num_grids=40,
        steps_per_frame=2,
    )

    assert profile_nsight._backend_choice_from_cfg(variant) == "cuda_v3"
    assert variant.backend._target_ == "mpm_jax.backends.cuda.CudaV3Backend"
    resolved_solver_backend = OmegaConf.to_container(
        variant.solver.backend, resolve=True
    )
    assert resolved_solver_backend["_target_"] == "mpm_jax.backends.cuda.CudaV3Backend"
    assert resolved_solver_backend["num_grids"] == 40


def test_merge_variant_cfg_matches_solver_instantiation_path():
    cfg = _compose_config()
    variant = profile_nsight._merge_variant_cfg(
        cfg,
        backend_choice="jax",
        n_particles=8,
        num_grids=16,
        steps_per_frame=1,
    )

    solver = MPMSolver(hydra.utils.instantiate(variant.solver))

    assert solver.backend.name == "jax"
    assert callable(solver.elasticity_fn)
