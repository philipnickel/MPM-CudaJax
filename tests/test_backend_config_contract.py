from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import mpm_jax.backends as backends
from mpm_jax.solver import MPMSolver


CONF_DIR = Path("conf").resolve()
EXPECTED_BACKENDS = {
    "jax": "mpm_jax.backends.jax.JaxBackend",
    "cuda_v1": "mpm_jax.backends.cuda.CudaV1Backend",
    "cuda_v2": "mpm_jax.backends.cuda.CudaV2Backend",
    "cuda_v3": "mpm_jax.backends.cuda.CudaV3Backend",
    "cuda_v4": "mpm_jax.backends.cuda.CudaV4Backend",
    "cutile": "mpm_jax.backends.cutile.CutileBackend",
}


def _registered_backend_config(choice):
    return backends.backend_config(choice)


def test_backend_config_choices_are_registered_backend_names():
    assert set(backends.backend_choices()) == set(EXPECTED_BACKENDS)


def test_backend_config_choices_point_at_expected_targets():
    for choice, target in EXPECTED_BACKENDS.items():
        cfg = _registered_backend_config(choice)
        assert cfg._target_ == target


def test_each_backend_config_instantiates_expected_backend_name(monkeypatch):
    monkeypatch.setattr("mpm_jax.backends.cuda.register_p2g_inline", lambda: True)
    monkeypatch.setattr("mpm_jax.backends.cuda.register_p2g_v2_inline", lambda: True)
    monkeypatch.setattr("mpm_jax.backends.cuda.register_p2g_v3_inline", lambda: True)
    monkeypatch.setattr("mpm_jax.backends.cuda.register_p2g_v4_inline", lambda: True)
    monkeypatch.setattr("mpm_jax.backends.cutile.load_cutile_kernels", lambda: None)
    monkeypatch.setattr("mpm_jax.backends.cutile.arena_super_cell", lambda: 2)

    for choice in EXPECTED_BACKENDS:
        cfg = _registered_backend_config(choice)
        cfg.num_grids = 16
        backend = hydra.utils.instantiate(cfg)
        assert backend.name == choice


def test_sweep_all_backend_choices_exist_and_are_trimmed():
    cfg = OmegaConf.load(CONF_DIR / "sweep_all.yaml")
    backend_axis = cfg.hydra.sweeper.params.backend
    choices = backend_axis.split(",")
    valid_choices = set(backends.backend_choices())

    assert choices
    for choice in choices:
        assert choice == choice.strip()
        assert choice in valid_choices


def test_hydra_composed_default_config_instantiates_small_jax_solver():
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "backend=jax",
                "sim=default",
                "sim.n_particles=32",
                "sim.num_grids=16",
                "sim.steps_per_frame=1",
                "sim.num_frames=1",
            ],
        )

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    assert solver.backend.name == "jax"
    assert callable(solver.elasticity_fn)
