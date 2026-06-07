from types import SimpleNamespace

from mpm_jax.solver import MPMSolver


def test_solver_metrics_returns_flat_results_record():
    solver = object.__new__(MPMSolver)
    solver.backend = SimpleNamespace(name="jax")
    solver.params = SimpleNamespace(
        n_particles=32,
        num_grids=16,
        dx=1 / 16,
        vol=0.125 / 32,
    )
    solver.num_frames = 2
    solver.steps_per_frame = 3
    solver.material_elasticity = "stvk_elasticity_jacobi"
    solver.gpu_type = "NVIDIA H100 80GB HBM3"

    record = solver.metrics(elapsed_s=0.012)

    assert record == {
        "kernel": "jax",
        "material_elasticity": "stvk_elasticity_jacobi",
        "n_particles": 32,
        "num_grids": 16,
        "dx": 0.0625,
        "grid_cells": 4096,
        "material_volume": 0.125,
        "active_grid_cells": 512.0,
        "particles_per_grid_cell": 0.0078125,
        "particles_per_active_cell": 0.0625,
        "num_frames": 2,
        "steps_per_frame": 3,
        "total_steps": 6,
        "elapsed_s": 0.012,
        "ms_per_frame": 6.0,
        "ms_per_step": 2.0,
        "steps_per_sec": 500.0,
        "particles_per_sec": 16000.0,
        "gpu_type": "NVIDIA H100 80GB HBM3",
    }
