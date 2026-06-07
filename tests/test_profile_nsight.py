from pathlib import Path

import hydra
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

    assert cfg.nsight.target == "scatter"
    assert cfg.backend._target_ == "mpm_jax.p2g.backends.jax.JaxBackend"
    assert profile_nsight._profile_config(cfg, "jax") == ("jax", 8, 16, 1)


def test_profile_config_reads_sim_axes():
    cfg = _compose_config(overrides=["sim.n_particles=12", "sim.num_grids=40"])
    assert profile_nsight._profile_config(cfg, "cuda_v3") == ("cuda_v3", 12, 40, 1)


def test_backend_choice_from_cfg_infers_from_backend_target():
    cfg = _compose_config(overrides=["backend=cuda_v3"])
    # Outside a Hydra run there is no runtime choice, so it falls back to the
    # backend config _target_.
    assert profile_nsight._backend_choice_from_cfg(cfg) == "cuda_v3"


def test_analyze_kwargs_carry_single_config_and_callables(tmp_path):
    cfg = _compose_config("nsight_profile")
    profile_config = profile_nsight._profile_config(cfg, "jax")

    kwargs = profile_nsight._nsight_analyze_kwargs(cfg, tmp_path, profile_config)

    assert kwargs["configs"] == [("jax", 8, 16, 1)]
    assert callable(kwargs["derive_metric"])
    assert kwargs["replay_mode"] == "kernel"
    assert "gpu__time_duration.sum" in kwargs["metrics"]


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
