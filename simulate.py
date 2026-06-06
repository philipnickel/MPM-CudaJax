import json
import os
import time

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

import mpm_jax.backends  # noqa: F401 - registers Hydra backend config choices
from mpm_jax.rendering import render_warp_opengl
from mpm_jax.solver import MPMSolver


# ---------------------------------------------------------------------------
# Unified run path
# ---------------------------------------------------------------------------


def _run_solver(solver, cfg: DictConfig):
    """Drive an MPMSolver: warmup, then benchmark or frame-capture loop."""
    sim = cfg.sim
    capture_frames = not cfg.get("benchmark", False)

    solver.step()
    jax.block_until_ready(solver.state.x)
    solver.reset_to_initial()

    frames = []

    t0 = time.perf_counter()
    for _ in tqdm(range(sim.num_frames), desc="simulate"):
        if capture_frames:
            frames.append(np.array(solver.state.x))
        solver.step()
        if capture_frames:
            jax.block_until_ready(solver.state.x)
    if not capture_frames:
        jax.block_until_ready(solver.state.x)
    elapsed = time.perf_counter() - t0

    total_steps = sim.num_frames * solver.steps_per_frame
    avg_frame_ms = elapsed / sim.num_frames * 1000
    summary = {
        "timestep": {
            "mean_ms": avg_frame_ms,
            "std_ms": 0.0,
            "total_ms": elapsed * 1000,
            "count": sim.num_frames,
        }
    }
    return frames, elapsed, total_steps, summary


def run(cfg: DictConfig):
    """Instantiate the runtime config, build the solver, and drive it.

    Returns (frames, elapsed, total_steps, summary).
    """
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    return _run_solver(solver, cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)

    frames, elapsed, total_steps, summary = run(cfg)
    kernel_name = HydraConfig.get().runtime.choices.get("backend", "jax")

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    backend_label = "solver-loop"
    print(
        f"\n{backend_label} ({kernel_name}): {total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)"
    )

    total_ms = sum(s["total_ms"] for s in summary.values())
    print(f"\nWall-clock timing (per frame, {cfg.sim.steps_per_frame} substeps each):")
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]["total_ms"]):
        pct = stats["total_ms"] / total_ms * 100 if total_ms > 0 else 0
        print(
            f"  {stage:15s}: {stats['mean_ms']:8.3f} ms/frame ({pct:5.1f}%  std={stats['std_ms']:.3f}  n={stats['count']})"
        )

    # Render GIF (skip in benchmark mode)
    export_path = None
    if not cfg.get("benchmark", False) and frames:
        render_cfg = cfg.get("render", {})
        fps = int(render_cfg.get("fps", 30))
        radius = float(render_cfg.get("point_radius", 0.008))
        export_path = os.path.join(run_dir, "render.gif")
        print(f"\nRendering with Warp OpenGL to {export_path}...")
        render_warp_opengl(
            frames,
            export_path,
            color=cfg.material.color,
            radius=radius,
            fps=fps,
            width=int(render_cfg.get("width", 960)),
            height=int(render_cfg.get("height", 720)),
        )
    elif cfg.get("benchmark", False):
        print("\nBenchmark mode: skipping GIF rendering.")

    # Dump a small results.json into the Hydra run dir so multirun callbacks
    # and post-hoc aggregation can pick up the per-run numbers. One file per
    # run, fixed shape.
    mat_name = HydraConfig.get().runtime.choices.get("material", "unknown")
    results = {
        "kernel": kernel_name,
        "material_elasticity": mat_name,
        "n_particles": int(cfg.sim.n_particles),
        "num_grids": int(cfg.sim.num_grids),
        "num_frames": int(cfg.sim.num_frames),
        "steps_per_frame": int(cfg.sim.steps_per_frame),
        "total_steps": int(total_steps),
        "elapsed_s": float(elapsed),
        "ms_per_step": float(ms_per_step),
        "steps_per_sec": float(steps_per_sec),
        "render_path": export_path,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
