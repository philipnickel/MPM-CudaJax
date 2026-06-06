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

    # Print timing summary
    backend_label = "solver-loop"
    print(
        f"\n{backend_label} ({metrics['kernel']}): {metrics['total_steps']} steps in {elapsed:.2f}s "
        f"({metrics['steps_per_sec']:.1f} steps/s, {metrics['ms_per_step']:.2f} ms/step)"
    )

    print(
        f"\nWall-clock timing: {metrics['ms_per_frame']:.3f} ms/frame "
        f"({solver.steps_per_frame} substeps each, n={solver.num_frames}, "
        f"{metrics['particles_per_sec']:.3e} particles/s)"
    )

    metrics["render_path"] = None
    if render_enabled and frames:
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
        metrics["render_path"] = export_path
    elif not render_enabled:
        print("\nRendering disabled.")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
