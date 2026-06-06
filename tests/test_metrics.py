import json

from mpm_jax.metrics import RunConfig, RunMetrics, metrics_dataframe


def test_run_metrics_writes_flat_results_record(tmp_path):
    run_config = RunConfig(
        kernel="jax",
        material_elasticity="jelly",
        n_particles=32,
        num_grids=16,
        num_frames=2,
        steps_per_frame=3,
        render_enabled=False,
    )
    metrics = RunMetrics(config=run_config, elapsed_s=0.012, render_path=None)

    path = tmp_path / "results.json"
    metrics.write_json(path)

    record = json.loads(path.read_text())
    assert record == {
        "kernel": "jax",
        "material_elasticity": "jelly",
        "n_particles": 32,
        "num_grids": 16,
        "num_frames": 2,
        "steps_per_frame": 3,
        "render_enabled": False,
        "total_steps": 6,
        "elapsed_s": 0.012,
        "ms_per_step": 2.0,
        "steps_per_sec": 500.0,
        "render_path": None,
    }


def test_metrics_dataframe_loads_results_json_files(tmp_path):
    first = RunMetrics(
        config=RunConfig("jax", "jelly", 32, 16, 1, 2, False),
        elapsed_s=0.01,
    )
    second = RunMetrics(
        config=RunConfig("cuda_v3", "jelly", 64, 16, 1, 2, False),
        elapsed_s=0.005,
    )
    first.write_json(tmp_path / "a.json")
    second.write_json(tmp_path / "b.json")

    df = metrics_dataframe([tmp_path / "a.json", tmp_path / "b.json"])

    assert list(df["kernel"]) == ["jax", "cuda_v3"]
    assert list(df["ms_per_step"]) == [5.0, 2.5]
