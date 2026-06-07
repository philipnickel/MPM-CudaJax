import json
import logging
import os

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mpm_jax.p2g.backends  # noqa: F401 - registers Hydra backend config choices
from mpm_jax.rendering import render_warp_opengl
from mpm_jax.solver import MPMSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    render_enabled = bool(cfg.render.get("enabled", True))

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))

    profile_cfg = cfg.get("profile", {})
    profile_enabled = bool(profile_cfg.get("enabled", False))

    # Warmup (JIT compilation) always runs outside any trace.
    solver.warmup(int(profile_cfg.get("warmup_frames", 1)))

    if profile_enabled:
        # Traces live in a shared top-level dir (one run per label) so
        # `xprof --logdir traces` lists them side by side. The label defaults to
        # the backend name; override `profile.label` to keep a focused capture
        # (e.g. a single-substep run) out of the backend-comparison runs.
        traces_root = os.path.join(hydra.utils.get_original_cwd(), "traces")
        run_label = profile_cfg.get("label") or solver.backend.name
        trace_dir = os.path.join(traces_root, run_label)
        os.makedirs(trace_dir, exist_ok=True)
        if profile_cfg.get("dump_hlo", False):
            hlo_path = os.path.join(trace_dir, "optimized_hlo.txt")
            hlo = solver._frame.lower(solver._init_state).compile().as_text()
            with open(hlo_path, "w") as f:
                f.write(hlo)
            logger.info("Wrote optimized HLO (post-fusion) to %s", hlo_path)
        options = jax.profiler.ProfileOptions()
        opts_cfg = OmegaConf.to_container(profile_cfg.get("options", {}), resolve=True)
        for key, value in (opts_cfg or {}).items():
            setattr(options, key, value)
        logger.info("Profiling enabled; writing XProf trace to %s", trace_dir)
        with jax.profiler.trace(trace_dir, profiler_options=options):
            frames, elapsed = solver.run(capture_frames=render_enabled)
        logger.info("View/compare: pixi run xprof --logdir %s", traces_root)
    else:
        frames, elapsed = solver.run(capture_frames=render_enabled)

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
        metrics["n_particles"],
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
