from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import profile_nsight
import mpm_jax.p2g.backends as backend_configs


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
    assert "derive_metric" not in kwargs
    assert kwargs["replay_mode"] == "kernel"
    assert kwargs["combine_kernel_metrics"] is None
    assert kwargs["output_csv"] is False
    assert "gpu__time_duration.sum" in kwargs["metrics"]


def test_results_dataframe_adds_config_metadata(tmp_path):
    cfg = _compose_config("nsight_profile")

    class Results:
        def to_dataframe(self):
            import pandas as pd

            return pd.DataFrame(
                {
                    "Metric": ["gpu__time_duration.sum"],
                    "AvgValue": [1000.0],
                }
            )

    df = profile_nsight._results_dataframe(Results(), cfg, tmp_path, "jax", "p2g")

    assert df.loc[0, "backend"] == "jax"
    assert df.loc[0, "target"] == "p2g"
    assert df.loc[0, "sim.n_particles"] == 8
    assert "hydra_config" in df


def test_write_dataframe_appends_parquet(tmp_path):
    import pandas as pd

    first = pd.DataFrame({"Metric": ["gpu__time_duration.sum"], "AvgValue": [1.0]})
    second = pd.DataFrame({"Metric": ["gpu__time_duration.sum"], "AvgValue": [2.0]})
    path = tmp_path / "results.parquet"

    profile_nsight._write_dataframe(first, path)
    profile_nsight._write_dataframe(second, path, append=True)

    assert pd.read_parquet(path)["AvgValue"].tolist() == [1.0, 2.0]


def test_p2g_runner_profiles_prepared_scatter():
    cfg = _compose_config()

    runner = profile_nsight._p2g_runner(cfg)
    runner.ensure_ready()
    grid_mv, grid_m = runner()

    assert grid_mv.shape == (16**3, 3)
    assert grid_m.shape == (16**3,)
