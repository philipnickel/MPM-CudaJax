from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import profile_nsight
import mpm_jax.p2g.backends as backend_configs
from mpm_jax.profiling import build_profile_target
from mpm_jax.solver import MPMSolver


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


class _NullAnnotation:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


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
                "target": "p2g",
                "write_json": False,
                "plot": {"enabled": False},
                "sweep": None,
                "configs": None,
                "analyze": {},
            }
        )
    return cfg


def test_backend_choices_come_from_registered_hydra_configs():
    assert set(backend_configs.backend_choices()) == {
        "jax",
        "cuda_v1",
        "cuda_v2",
        "cuda_v3",
        "cuda_v4",
        "cutile_v1",
        "cutile_v3",
    }


def test_nsight_profile_config_composes():
    cfg = _compose_config("nsight_profile")

    assert cfg.nsight.target == "p2g"
    assert cfg.backend._target_ == "mpm_jax.p2g.backends.jax.JaxBackend"
    assert profile_nsight._nsight_configs(cfg) == [("jax", 8, 16, 1)]


def test_profile_target_reuses_solver_state_without_extra_particle_prepass():
    cfg = _compose_config()

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    target = build_profile_target(solver, "p2g")
    grid_mv, grid_m = target.run()

    assert target.backend_name == "jax"
    assert grid_mv.shape == (16**3, 3)
    assert grid_m.shape == (16**3,)


def test_profile_runner_accepts_scatter_target():
    cfg = _compose_config()
    cfg.nsight.target = "scatter"

    runner = profile_nsight._profile_runner(
        cfg, type("Nsight", (), {"annotate": _NullAnnotation})()
    )

    runner()


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
    cfg.nsight.sweep = {"kernels": ["CudaV3Backend"]}

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
    assert variant.backend._target_ == "mpm_jax.p2g.backends.cuda.CudaV3Backend"
    resolved_solver_backend = OmegaConf.to_container(
        variant.solver.backend, resolve=True
    )
    assert resolved_solver_backend["_target_"] == "mpm_jax.p2g.backends.cuda.CudaV3Backend"
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
