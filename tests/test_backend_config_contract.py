from pathlib import Path
from types import SimpleNamespace

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import mpm_jax.p2g.backends as backends
from mpm_jax.solver import MPMSolver


CONF_DIR = Path("conf").resolve()
EXPECTED_BACKENDS = {
    "jax": "mpm_jax.p2g.backends.jax.JaxBackend",
    "cuda_v1": "mpm_jax.p2g.backends.cuda.CudaV1Backend",
    "cuda_v2": "mpm_jax.p2g.backends.cuda.CudaV2Backend",
    "cuda_v3": "mpm_jax.p2g.backends.cuda.CudaV3Backend",
    "cuda_v4": "mpm_jax.p2g.backends.cuda.CudaV4Backend",
    "cutile_v1": "mpm_jax.p2g.backends.cutile.CutileV1Backend",
    "cutile_v3": "mpm_jax.p2g.backends.cutile.CutileV3Backend",
}


def test_backend_config_choices_are_registered_backend_names():
    assert set(backends.backend_choices()) == set(EXPECTED_BACKENDS)


def test_backend_config_choices_point_at_expected_targets():
    for choice, target in EXPECTED_BACKENDS.items():
        cfg = backends.backend_config(choice)
        assert cfg._target_ == target


def test_each_backend_config_instantiates_expected_backend_name(monkeypatch):
    monkeypatch.setattr("mpm_jax.p2g.cuda.p2g_cuda.CudaV1P2G.register", lambda self: True)
    monkeypatch.setattr("mpm_jax.p2g.cuda.p2g_cuda.CudaV2P2G.register", lambda self: True)
    monkeypatch.setattr("mpm_jax.p2g.cuda.p2g_cuda.CudaV3P2G.register", lambda self: True)
    monkeypatch.setattr("mpm_jax.p2g.cuda.p2g_cuda.CudaV4P2G.register", lambda self: True)
    monkeypatch.setattr(
        "mpm_jax.p2g.backends.cutile._cutile_module",
        lambda: SimpleNamespace(
            cutile_p2g_v1=lambda *args, **kwargs: None,
            cutile_p2g_v3=lambda *args, **kwargs: None,
        ),
    )

    for choice in EXPECTED_BACKENDS:
        cfg = backends.backend_config(choice)
        cfg.num_grids = 16
        backend = hydra.utils.instantiate(cfg)
        assert backend.name == choice


def _compose_sweep(choice: str = "particle_count"):
    # Mirror Hydra's defaults list: group choice first, then `_self_` (sweep.yaml) on top.
    # The group choices use `# @package _global_`, so their bodies merge at the root.
    group_cfg = OmegaConf.load(CONF_DIR / "sweep" / f"{choice}.yaml")
    sweep_cfg = OmegaConf.load(CONF_DIR / "sweep.yaml")
    return OmegaConf.merge(group_cfg, sweep_cfg)


def _scale_choices(cfg) -> list[str]:
    return cfg.hydra.sweeper.params.scale.split(",")


def _backend_choices(cfg) -> list[str]:
    return cfg.hydra.sweeper.params.backend.split(",")


def test_scaling_sweeps_use_backend_choices_and_valid_scale_configs():
    valid_choices = set(backends.backend_choices())
    for choice in ["particle_count", "weak_scaling"]:
        cfg = _compose_sweep(choice)
        backend_choices = _backend_choices(cfg)
        scale_choices = _scale_choices(cfg)

        assert backend_choices
        assert scale_choices
        for backend_choice in backend_choices:
            assert backend_choice == backend_choice.strip()
            assert backend_choice in valid_choices
        for scale_choice in scale_choices:
            scale_cfg = OmegaConf.load(CONF_DIR / "scale" / f"{scale_choice}.yaml")
            assert scale_cfg.sim.n_particles > 0
            assert scale_cfg.sim.num_grids % 4 == 0


_PARTICLE_COUNT_SEQUENCE = [
    100_000, 200_000, 400_000, 800_000, 1_000_000,
    2_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000,
    100_000_000, 125_000_000,
]


def test_particle_count_sweep_uses_canonical_n_subset_at_fixed_g96():
    cfg = _compose_sweep("particle_count")
    particle_choices = []
    grid_choices = []
    for choice in _scale_choices(cfg):
        scale_cfg = OmegaConf.load(CONF_DIR / "scale" / f"{choice}.yaml")
        particle_choices.append(int(scale_cfg.sim.n_particles))
        grid_choices.append(int(scale_cfg.sim.num_grids))

    assert particle_choices, "particle_count sweep must select at least one scale"
    assert set(particle_choices).issubset(_PARTICLE_COUNT_SEQUENCE)
    assert particle_choices == sorted(particle_choices)
    assert set(grid_choices) == {96}


def test_weak_scaling_sweep_keeps_active_ppc_near_target():
    cfg = _compose_sweep("weak_scaling")
    grid_choices = []
    particle_choices = []
    active_ppc_choices = []
    for choice in _scale_choices(cfg):
        scale_cfg = OmegaConf.load(CONF_DIR / "scale" / f"{choice}.yaml")
        n_particles = int(scale_cfg.sim.n_particles)
        num_grids = int(scale_cfg.sim.num_grids)
        active_cells = 0.8**3 * num_grids**3
        grid_choices.append(num_grids)
        particle_choices.append(n_particles)
        active_ppc_choices.append(n_particles / active_cells)

    assert particle_choices, "weak_scaling sweep must select at least one scale"
    assert set(particle_choices).issubset(_PARTICLE_COUNT_SEQUENCE)
    assert particle_choices == sorted(particle_choices)
    assert all(g > 0 and g % 4 == 0 for g in grid_choices)
    # G is rounded to multiples of 4 to satisfy super-cell-tiled backends, so PPC
    # isn't exactly the target — but should stay within ~10% of 8.5.
    target_ppc = 10000000 / (0.8**3 * 132**3)
    for active_ppc in active_ppc_choices:
        assert abs(active_ppc - target_ppc) / target_ppc < 0.10


def test_sweep_axis_tags_match_plot_sweeps_specs():
    expected = {
        "particle_count": "sweep_particle_count",
        "weak_scaling": "sweep_weak_scaling",
        "particle_density": "sweep_particle_density",
    }
    for choice, expected_tag in expected.items():
        cfg = _compose_sweep(choice)
        assert cfg.tag == expected_tag


def test_particle_density_sweep_uses_fixed_n_and_varying_grid():
    cfg = _compose_sweep("particle_density")
    particle_choices = []
    grid_choices = []
    for choice in _scale_choices(cfg):
        scale_cfg = OmegaConf.load(CONF_DIR / "scale" / f"{choice}.yaml")
        particle_choices.append(int(scale_cfg.sim.n_particles))
        grid_choices.append(int(scale_cfg.sim.num_grids))

    assert set(particle_choices) == {20_000_000}
    assert grid_choices == sorted(grid_choices)
    assert all(g % 4 == 0 for g in grid_choices)
    assert grid_choices[0] >= 64 and grid_choices[-1] <= 196


def test_default_hydra_output_dirs_are_aggregation_friendly():
    cfg = OmegaConf.load(CONF_DIR / "config.yaml")
    raw = OmegaConf.to_container(cfg, resolve=False)
    hydra_cfg = raw["hydra"]

    assert hydra_cfg["run"]["dir"].startswith("outputs/runs/${gpu_kind:}/")
    assert hydra_cfg["sweep"]["dir"].startswith("outputs/sweeps/${gpu_kind:}/runs/")
    assert "${hydra.job.override_dirname}" in hydra_cfg["sweep"]["subdir"]


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
