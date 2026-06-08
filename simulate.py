import csv
import json
import logging
import os
import re
import sys
from pathlib import Path

# Disable command buffers before JAX init; sniff argv pre-Hydra.
# ruff: noqa: E402
def _config_name_from_argv(argv):
    config_name = "config"
    for i, arg in enumerate(argv):
        if arg in {"-cn", "--config-name"} and i + 1 < len(argv):
            config_name = argv[i + 1]
        elif arg.startswith(("-cn=", "--config-name=")):
            config_name = arg.split("=", 1)[1]
    return config_name


def _disable_command_buffers_for_trace(argv):
    lowered = [str(arg).lower() for arg in argv]
    if any("profile.disable_command_buffers=false" in arg for arg in lowered):
        return False

    config_name = _config_name_from_argv(argv)
    profile_enabled = any("profile.enabled=true" in arg for arg in lowered)
    if not (config_name.startswith("trace") or profile_enabled):
        return False

    flags = os.environ.get("XLA_FLAGS", "")
    replacement = "--xla_gpu_enable_command_buffer="
    if "--xla_gpu_enable_command_buffer" in flags:
        flags = re.sub(r"--xla_gpu_enable_command_buffer=\S*", replacement, flags)
    else:
        flags = f"{flags} {replacement}".strip()
    os.environ["XLA_FLAGS"] = flags
    return True


_disable_command_buffers_for_trace(sys.argv)

import hydra
import jax
import nvtx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mpm_jax.p2g.backends  # noqa: F401 - registers Hydra backend config choices
import mpm_jax.resolvers  # noqa: F401 - registers OmegaConf resolvers (ppc_grid, gpu_kind)
import postprocessing  # noqa: F401 - registers Hydra `plot` config group via hydra-zen
from mpm_jax.rendering import render_warp_opengl
from mpm_jax.resolvers import slugify
from mpm_jax.solver import MPMSolver

logger = logging.getLogger(__name__)
NVTX_DOMAIN = "mpm_cudajax"


def _analysis_csv_path(metrics):
    job_num = metrics.get("hydra_job_num")
    if job_num is None:
        return None
    original_cwd = Path(hydra.utils.get_original_cwd())
    sweep_dir = Path(metrics["hydra_sweep_dir"])
    if not sweep_dir.is_absolute():
        sweep_dir = original_cwd / sweep_dir
    try:
        runs_index = sweep_dir.parts.index("runs")
        sweep_root = Path(*sweep_dir.parts[:runs_index])
    except ValueError:
        sweep_root = sweep_dir
    return sweep_root / "results.csv"


def _write_csv_row(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def _write_metrics(run_dir, metrics):
    results_path = Path(run_dir) / "results.json"
    metrics["results_json_path"] = str(results_path)

    analysis_csv_path = _analysis_csv_path(metrics)
    metrics["analysis_csv_path"] = (
        str(analysis_csv_path) if analysis_csv_path else None
    )

    results_path.write_text(json.dumps(metrics, indent=2))
    (Path(run_dir) / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    _write_csv_row(Path(run_dir) / "metrics.csv", metrics)

    if analysis_csv_path is not None:
        _write_csv_row(analysis_csv_path, metrics)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    render_enabled = bool(cfg.render.get("enabled", True))

    profile_cfg = cfg.get("profile", {})
    profile_enabled = bool(profile_cfg.get("enabled", False))

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))

    # Warmup outside the trace.
    solver.warmup(int(profile_cfg.get("warmup_frames", 1)))

    def run_measured_solve():
        solve_range = f"{solver.backend.name}_solve"
        with nvtx.annotate(solve_range, domain=NVTX_DOMAIN):
            return solver.run(capture_frames=render_enabled)

    if profile_enabled:
        # Shared trace root for side-by-side comparison.
        traces_root = os.path.join(hydra.utils.get_original_cwd(), "traces")
        run_label = profile_cfg.get("label") or solver.backend.name
        trace_dir = os.path.join(traces_root, run_label)
        os.makedirs(trace_dir, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        opts_cfg = OmegaConf.to_container(profile_cfg.get("options", {}), resolve=True)
        for key, value in (opts_cfg or {}).items():
            setattr(options, key, value)
        logger.info("Trace collection XLA_FLAGS=%r", os.environ.get("XLA_FLAGS", ""))
        logger.info("Profiling enabled; writing XProf trace to %s", trace_dir)
        with jax.profiler.trace(trace_dir, profiler_options=options):
            frames, elapsed = run_measured_solve()
        logger.info("View/compare: pixi run xprof --logdir %s", traces_root)
    else:
        frames, elapsed = run_measured_solve()

    metrics = solver.metrics(elapsed)
    metrics["tag"] = cfg.get("tag")
    metrics["render_enabled"] = render_enabled
    metrics["gpu_kind"] = slugify(metrics["gpu_type"])
    metrics["mps_thread_percent"] = cfg.get("mps_thread_percent")
    hydra_cfg = HydraConfig.get()
    metrics["output_dir"] = run_dir
    metrics["hydra_job_num"] = OmegaConf.select(hydra_cfg, "job.num")
    metrics["hydra_sweep_dir"] = OmegaConf.select(hydra_cfg, "sweep.dir")
    metrics["hydra_override_dirname"] = OmegaConf.select(
        hydra_cfg, "job.override_dirname"
    )
    task_overrides = OmegaConf.select(hydra_cfg, "overrides.task") or []
    metrics["hydra_task_overrides"] = ",".join(str(item) for item in task_overrides)

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

    _write_metrics(run_dir, metrics)


if __name__ == "__main__":
    main()
