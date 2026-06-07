# Trace collection disables CUDA graphs so kernels are visible to the profiler
# (graphs hide them inside cuGraphLaunch -> empty GPU timeline). XLA reads
# XLA_FLAGS once, at backend init during the jax imports below, so the flag must
# be flipped here -- before importing jax, hence too early for the Hydra config
# (we sniff argv for the trace configs / an explicit profile.enabled=true).
# ruff: noqa: E402
import os
import re
import sys


def _disable_command_buffers_for_trace(argv):
    if any("disable_command_buffers=false" in a.lower() for a in argv):
        return
    cfg_name = "config"
    for i, a in enumerate(argv):
        if a in ("-cn", "--config-name") and i + 1 < len(argv):
            cfg_name = argv[i + 1]
        elif a.startswith(("--config-name=", "-cn=")):
            cfg_name = a.split("=", 1)[1]
    if cfg_name not in ("trace", "trace_substep") and not any(
        "profile.enabled=true" in a.lower() for a in argv
    ):
        return
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer" in flags:
        flags = re.sub(
            r"--xla_gpu_enable_command_buffer=\S*",
            "--xla_gpu_enable_command_buffer=",
            flags,
        )
    else:
        flags = f"{flags} --xla_gpu_enable_command_buffer=".strip()
    os.environ["XLA_FLAGS"] = flags


_disable_command_buffers_for_trace(sys.argv)

import json
import logging

import hydra
import jax
import nvtx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mpm_jax.p2g.backends  # noqa: F401 - registers Hydra backend config choices
from mpm_jax.profiling import NVTX_DOMAIN
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

    profile_cfg = cfg.get("profile", {})
    profile_enabled = bool(profile_cfg.get("enabled", False))
    step_mode = str(profile_cfg.get("step_mode", "frame"))
    if step_mode not in {"frame", "staged"}:
        raise ValueError("profile.step_mode must be 'frame' or 'staged'.")
    staged = step_mode == "staged"

    if profile_enabled:
        logger.info(
            "Trace collection: XLA_FLAGS=%r", os.environ.get("XLA_FLAGS", "")
        )

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))

    # Warmup (JIT compilation) always runs outside any trace.
    solver.warmup(int(profile_cfg.get("warmup_frames", 1)), staged=staged)

    def run_measured_solve():
        solve_range = f"{solver.backend.name}_solve"
        with nvtx.annotate(solve_range, domain=NVTX_DOMAIN):
            return solver.run(
                capture_frames=render_enabled,
                staged=staged,
                profile_stages=staged,
            )

    if profile_enabled:
        # Traces live in a shared top-level dir (one run per label) so
        # `xprof --logdir traces` lists them side by side. The label defaults to
        # the backend name; override `profile.label` to keep a focused capture
        # (e.g. a single-substep run) out of the backend-comparison runs.
        traces_root = os.path.join(hydra.utils.get_original_cwd(), "traces")
        run_label = profile_cfg.get("label") or solver.backend.name
        trace_dir = os.path.join(traces_root, run_label)
        os.makedirs(trace_dir, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        opts_cfg = OmegaConf.to_container(profile_cfg.get("options", {}), resolve=True)
        for key, value in (opts_cfg or {}).items():
            setattr(options, key, value)
        logger.info("Profiling enabled; writing XProf trace to %s", trace_dir)
        with jax.profiler.trace(trace_dir, profiler_options=options):
            frames, elapsed = run_measured_solve()
        logger.info("View/compare: pixi run xprof --logdir %s", traces_root)
    else:
        frames, elapsed = run_measured_solve()

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
