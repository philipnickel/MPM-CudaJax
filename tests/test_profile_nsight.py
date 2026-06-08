from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import profile_nsight
from postprocessing import nsight_plots
from postprocessing.callbacks import NsightPlotCallback
from postprocessing.nsight_plots import render_nsight_figures
import mpm_jax.p2g.backends as backend_configs


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_config(config_name="config", overrides=None):
    overrides = [
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


def _compose_config_with_hydra(config_name="config", overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(
            config_name=config_name,
            overrides=overrides or [],
            return_hydra_config=True,
        )


def test_backend_choices_come_from_registered_hydra_configs():
    assert set(backend_configs.backend_choices()) == {
        "jax",
        "cuda_v1",
        "cuda_v2",
        "cuda_v3",
        "CuTile",
    }


def test_nsight_profile_config_composes():
    cfg = _compose_config("nsight_profile", overrides=["backend=cuda_v1"])

    assert cfg.backend._target_ == "mpm_jax.p2g.backends.cuda.CudaV1Backend"
    assert profile_nsight._profile_config(cfg, "cuda_v1") == ("cuda_v1", 8, 16, 1)


def test_nsight_profile_rejects_jax_backend():
    cfg = _compose_config("nsight_profile", overrides=["backend=jax"])

    try:
        profile_nsight._profile_config(cfg, "jax")
    except RuntimeError as exc:
        assert "custom CUDA/cuTile P2G scatter kernels only" in str(exc)
        assert "XProf" in str(exc)
    else:
        raise AssertionError("backend=jax should not be valid for Nsight analysis")


def test_profile_config_reads_sim_axes():
    cfg = _compose_config(
        overrides=["backend=cuda_v3", "sim.n_particles=12", "sim.num_grids=40"]
    )
    assert profile_nsight._profile_config(cfg, "cuda_v3") == ("cuda_v3", 12, 40, 1)


def test_backend_choice_from_cfg_infers_from_backend_target():
    cfg = _compose_config(overrides=["backend=cuda_v3"])
    # Outside a Hydra run there is no runtime choice, so it falls back to the
    # backend config _target_.
    assert profile_nsight._backend_choice_from_cfg(cfg) == "cuda_v3"


def test_analyze_kwargs_carry_single_config_and_callables(tmp_path):
    cfg = _compose_config("nsight_profile", overrides=["backend=cuda_v1"])
    profile_config = profile_nsight._profile_config(cfg, "cuda_v1")

    kwargs = profile_nsight._nsight_analyze_kwargs(cfg, tmp_path, profile_config)

    assert kwargs["configs"] == [("cuda_v1", 8, 16, 1)]
    assert "derive_metric" not in kwargs
    assert kwargs["replay_mode"] == "kernel"
    assert kwargs["combine_kernel_metrics"] is None
    assert kwargs["output_csv"] is False
    assert "gpu__time_duration.sum" in kwargs["metrics"]


def test_nsight_metric_presets_compose():
    full = _compose_config("nsight_profile", overrides=["backend=cuda_v1"])
    timing = _compose_config(
        "nsight_profile", overrides=["backend=cuda_v1", "nsight_metrics=timing"]
    )
    roofline = _compose_config(
        "nsight_profile", overrides=["backend=cuda_v1", "nsight_metrics=roofline"]
    )

    assert "gpu__time_duration.sum" in full.nsight.analyze.metrics
    assert list(timing.nsight.analyze.metrics) == ["gpu__time_duration.sum"]
    assert "lts__t_bytes.sum" in roofline.nsight.analyze.metrics
    assert "sm__sass_thread_inst_executed_op_ffma_pred_on.sum" in roofline.nsight.analyze.metrics
    assert "launch__registers_per_thread" not in roofline.nsight.analyze.metrics


def test_nsight_sweep_configs_exclude_jax_and_use_direct_axes():
    valid_backends = set(backend_configs.backend_choices()) - {"jax"}
    expectations = {
        "single_point": {},
        "particle_count": {
            "sweep_axis": "sim.n_particles",
            "fixed": ("sim.num_grids", 128),
            "tag": "nsight_particle_count",
        },
        "density": {
            "sweep_axis": "sim.num_grids",
            "fixed": ("sim.n_particles", 20_000_000),
            "tag": "nsight_density",
        },
        "weak": {
            "sweep_axis": "sim.n_particles",
            "fixed": ("sim.num_grids", "${ppc_grid:${sim.n_particles}}"),
            "tag": "nsight_weak",
        },
        "sm_scaling": {
            "fixed": ("sim.num_grids", 128),
            "tag": "nsight_sm_scaling",
        },
    }
    for sweep, expected in expectations.items():
        cfg = _compose_config_with_hydra(
            "nsight_profile", overrides=[f"nsight_sweep={sweep}"]
        )
        params = cfg.hydra.sweeper.params
        backend_choices = params.backend.split(",")
        assert backend_choices
        assert "jax" not in backend_choices
        assert set(backend_choices).issubset(valid_backends)
        assert "scale" not in params
        if expected:
            if "sweep_axis" in expected:
                sweep_values = params[expected["sweep_axis"]].split(",")
                assert sweep_values
                assert [int(v) for v in sweep_values] == sorted(
                    int(v) for v in sweep_values
                )
            key, value = expected["fixed"]
            if isinstance(value, str):
                raw_cfg = OmegaConf.to_container(cfg, resolve=False)
                current = raw_cfg
                for part in key.split("."):
                    current = current[part]
                assert current == value
            else:
                assert OmegaConf.select(cfg, key) == value
            assert cfg.tag == expected["tag"]
            if sweep == "sm_scaling":
                hydra_cfg = OmegaConf.to_container(cfg, resolve=False)["hydra"]
                assert hydra_cfg["sweep"]["dir"] == (
                    "outputs/nsight/${gpu_kind:}/sm_scaling"
                )
                assert cfg.hydra.callbacks.nsight_plot.scale_axis == (
                    "mps_thread_percent"
                )


def test_nsight_plot_callback_requires_aggregate_parquet(tmp_path):
    callback = NsightPlotCallback()
    cfg = OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}})

    try:
        callback.on_multirun_end(cfg)
    except FileNotFoundError as exc:
        assert "results.parquet" in str(exc)
    else:
        raise AssertionError("NsightPlotCallback must require results.parquet")


def test_render_nsight_figures_from_parquet(tmp_path):
    import pandas as pd

    rows = []
    for backend, boost in [("cuda_v1", 1.0), ("cuda_v2", 1.4)]:
        for n_particles, mult in [(1000, 1.0), (2000, 1.3)]:
            rows.append(
                {
                    "backend": backend,
                    "target": "p2g",
                    "n_particles": n_particles,
                    "num_grids": 16,
                    "gflops_per_s": 500.0 * boost * mult,
                    "peak_compute_gflops": 19000.0,
                    "peak_l1_gbps": 120000.0,
                    "peak_l2_gbps": 8000.0,
                    "peak_hbm_gbps": 1600.0,
                    "ai_l1_flop_per_byte": 2.0 * mult,
                    "ai_l2_flop_per_byte": 1.0 * mult,
                    "ai_hbm_flop_per_byte": 0.25 * mult,
                }
            )
    path = tmp_path / "results.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    written = render_nsight_figures(
        path,
        tmp_path / "figures",
        plots=["roofline_scaling"],
        style={"theme": "whitegrid", "context": "paper"},
    )

    assert [p.name for p in written] == ["roofline_scaling.png"]
    assert written[0].exists()


def test_render_nsight_mps_roofline_trajectory_from_parquet(tmp_path):
    import pandas as pd

    rows = []
    for backend, boost in [("cuda_v1", 1.0), ("cuda_v2", 1.4)]:
        for mps_thread_percent, mult in [(25, 0.7), (100, 1.0)]:
            rows.append(
                {
                    "backend": backend,
                    "target": "p2g",
                    "mps_thread_percent": mps_thread_percent,
                    "n_particles": 1000,
                    "num_grids": 16,
                    "gflops_per_s": 500.0 * boost * mult,
                    "peak_compute_gflops": 19000.0,
                    "peak_l1_gbps": 120000.0,
                    "peak_l2_gbps": 8000.0,
                    "peak_hbm_gbps": 1600.0,
                    "ai_l1_flop_per_byte": 2.0 * mult,
                    "ai_l2_flop_per_byte": 1.0 * mult,
                    "ai_hbm_flop_per_byte": 0.25 * mult,
                }
            )
    path = tmp_path / "results.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    written = render_nsight_figures(
        path,
        tmp_path / "figures",
        plots=["roofline_scaling"],
        style={"theme": "whitegrid", "context": "paper"},
    )

    assert nsight_plots._resolve_scale_axis(nsight_plots.load_dataframe(path)) == (
        "mps_thread_percent",
        "CUDA MPS active-thread %",
    )
    assert [p.name for p in written] == ["roofline_scaling.png"]
    assert written[0].exists()


def test_render_roofline_from_nersc_2020_metrics(tmp_path):
    import pandas as pd

    metric_values = {
        "sm__cycles_elapsed.avg": 1_000_000.0,
        "sm__cycles_elapsed.avg.per_second": 1_000_000_000.0,
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": 1_000_000.0,
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": 1_000_000.0,
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": 1_000_000.0,
        "l1tex__t_bytes.sum": 4_000_000.0,
        "lts__t_bytes.sum": 2_000_000.0,
        "dram__bytes.sum": 1_000_000.0,
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained": 10.0,
        "l1tex__t_bytes.sum.peak_sustained": 8.0,
        "l1tex__cycles_elapsed.avg.per_second": 1_000_000_000.0,
        "lts__t_bytes.sum.peak_sustained": 4.0,
        "lts__cycles_elapsed.avg.per_second": 1_000_000_000.0,
        "dram__bytes.sum.peak_sustained": 2.0,
        "dram__cycles_elapsed.avg.per_second": 1_000_000_000.0,
    }
    rows = []
    for backend in ["cuda_v1", "cuda_v2"]:
        for metric, value in metric_values.items():
            rows.append(
                {
                    "backend": backend,
                    "target": "p2g",
                    "n_particles": 1000,
                    "num_grids": 16,
                    "Annotation": f"{backend}_p2g",
                    "Metric": metric,
                    "AvgValue": value,
                    "Kernel": "p2g",
                }
            )
    path = tmp_path / "results.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    written = render_nsight_figures(
        path,
        tmp_path / "figures",
        plots=["roofline"],
        style={"theme": "darkgrid", "context": "paper"},
    )

    assert [p.name for p in written] == ["roofline.png"]
    assert written[0].exists()


def test_render_nsight_figures_skips_missing_metric_plots(tmp_path):
    import pandas as pd

    path = tmp_path / "results.parquet"
    pd.DataFrame(
        {
            "backend": ["cuda_v1"],
            "target": ["p2g"],
            "n_particles": [8],
            "num_grids": [16],
            "Metric": ["gpu__time_duration.sum"],
            "AvgValue": [1000.0],
        }
    ).to_parquet(path, index=False)

    written = render_nsight_figures(path, tmp_path / "figures")

    assert written == []
    assert not list((tmp_path / "figures").glob("*.png"))


def test_nsight_gpu_label_falls_back_to_output_path():
    import pandas as pd

    path = Path("outputs/nsight/nvidia_test_gpu/sweeps/2026-06-08/results.parquet")

    assert nsight_plots._gpu_label(pd.DataFrame({"backend": ["cuda_v1"]}), path) == (
        "nvidia_test_gpu"
    )


def test_results_dataframe_adds_config_metadata(tmp_path, monkeypatch):
    cfg = _compose_config("nsight_profile", overrides=["mps_thread_percent=50"])
    monkeypatch.setattr(profile_nsight, "_current_gpu_type", lambda: "NVIDIA Test GPU")

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
    assert df.loc[0, "gpu_type"] == "NVIDIA Test GPU"
    assert df.loc[0, "gpu_kind"] == "nvidia_test_gpu"
    assert df.loc[0, "mps_thread_percent"] == 50
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
