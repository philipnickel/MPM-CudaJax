from types import SimpleNamespace

from mpm_jax.solver import MPMSolver


def test_solver_metrics_returns_flat_results_record():
    solver = object.__new__(MPMSolver)
    solver.backend = SimpleNamespace(name="jax")
    solver.params = SimpleNamespace(n_particles=32, num_grids=16)
    solver.steps_per_frame = 3

    record = solver.metrics(
        elapsed_s=0.012,
        num_frames=2,
        render_enabled=False,
        material_elasticity="jelly",
        render_path=None,
    )

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
