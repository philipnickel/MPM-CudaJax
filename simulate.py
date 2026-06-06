import os
import time
from dataclasses import replace

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

import mpm_jax.backends  # noqa: F401 - registers Hydra backend config choices
from mpm_jax.metrics import RunConfig, RunMetrics
from mpm_jax.rendering import render_warp_opengl
from mpm_jax.solver import MPMSolver


# ---------------------------------------------------------------------------
# Unified run path
# ---------------------------------------------------------------------------


def _run_solver(solver, run_config: RunConfig):
    """Drive an MPMSolver, capturing frames only when rendering is enabled."""
    solver.step()
    jax.block_until_ready(solver.state.x)
    solver.reset_to_initial()

    frames = []

    t0 = time.perf_counter()
    for _ in tqdm(range(run_config.num_frames), desc="simulate"):
        if run_config.render_enabled:
            frames.append(np.array(solver.state.x))
        solver.step()
        if run_config.render_enabled:
            jax.block_until_ready(solver.state.x)
    if not run_config.render_enabled:
        jax.block_until_ready(solver.state.x)
    elapsed = time.perf_counter() - t0

    return frames, elapsed


def run(cfg: DictConfig, run_config: RunConfig):
    """Instantiate the runtime config, build the solver, and drive it.

    Returns (frames, elapsed_s).
    """
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    return _run_solver(solver, run_config)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    choices = HydraConfig.get().runtime.choices
    run_config = RunConfig.from_hydra(cfg, choices)

    frames, elapsed = run(cfg, run_config)
    metrics = RunMetrics(config=run_config, elapsed_s=elapsed)

    # Print timing summary
    backend_label = "solver-loop"
    print(
        f"\n{backend_label} ({run_config.kernel}): {metrics.total_steps} steps in {metrics.elapsed_s:.2f}s "
        f"({metrics.steps_per_sec:.1f} steps/s, {metrics.ms_per_step:.2f} ms/step)"
    )

    print(
        f"\nWall-clock timing: {metrics.ms_per_frame:.3f} ms/frame "
        f"({run_config.steps_per_frame} substeps each, n={run_config.num_frames})"
    )

    export_path = None
    if run_config.render_enabled and frames:
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
        metrics = replace(metrics, render_path=export_path)
    elif not run_config.render_enabled:
        print("\nRendering disabled.")

    metrics.write_json(os.path.join(run_dir, "results.json"))


if __name__ == "__main__":
    main()
