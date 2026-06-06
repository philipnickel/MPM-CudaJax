import json
import logging
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified run path
# ---------------------------------------------------------------------------


def _run_solver(solver, *, render_enabled):
    """Drive an MPMSolver, capturing frames only when rendering is enabled."""
    solver.step()
    jax.block_until_ready(solver.state.x)
    solver.reset_to_initial()

    frames = []

    t0 = time.perf_counter()
    for _ in tqdm(range(solver.num_frames), desc="simulate"):
        if render_enabled:
            frames.append(np.array(solver.state.x))
        solver.step()
        if render_enabled:
            jax.block_until_ready(solver.state.x)
    if not render_enabled:
        jax.block_until_ready(solver.state.x)
    elapsed = time.perf_counter() - t0

    return frames, elapsed


def run(cfg: DictConfig):
    """Instantiate the runtime config, build the solver, and drive it.

    Returns (solver, frames, elapsed_s).
    """
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    frames, elapsed = _run_solver(
        solver,
        render_enabled=bool(cfg.render.get("enabled", True)),
    )
    return solver, frames, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    render_enabled = bool(cfg.render.get("enabled", True))

    solver, frames, elapsed = run(cfg)
    metrics = solver.metrics(elapsed)
    metrics["render_enabled"] = render_enabled

    backend_label = "solver-loop"
    logger.info(
        "%s (%s): %d steps in %.2fs (%.1f steps/s, %.2f ms/step)",
        backend_label,
        metrics["kernel"],
        metrics["total_steps"],
        elapsed,
        metrics["steps_per_sec"],
        metrics["ms_per_step"],
    )

    logger.info(
        "Wall-clock timing: %.3f ms/frame (%d substeps each, n=%d, %.3e particles/s)",
        metrics["ms_per_frame"],
        solver.steps_per_frame,
        solver.num_frames,
        metrics["particles_per_sec"],
    )

    metrics["render_path"] = None
    if render_enabled and frames:
        render_cfg = cfg.get("render", {})
        fps = int(render_cfg.get("fps", 30))
        radius = float(render_cfg.get("point_radius", 0.008))
        export_path = os.path.join(run_dir, "render.gif")
        logger.info("Rendering with Warp OpenGL to %s", export_path)
        render_warp_opengl(
            frames,
            export_path,
            color=cfg.material.color,
            radius=radius,
            fps=fps,
            width=int(render_cfg.get("width", 960)),
            height=int(render_cfg.get("height", 720)),
        )
        metrics["render_path"] = export_path
    elif not render_enabled:
        logger.info("Rendering disabled.")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
