import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# CUDA reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE when the driver attaches, which
# happens during `import jax` under JAX_PLATFORMS=cuda. Pre-parse the Hydra CLI
# (mps_thread_percent=N, +mps_thread_percent=N, ++mps_thread_percent=N) and set
# the env var before the heavy CUDA imports below.
for _arg in sys.argv[1:]:
    if "mps_thread_percent=" in _arg:
        _val = _arg.split("=", 1)[1].strip()
        if _val and _val.lower() not in ("null", "none"):
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = _val
        break

import hydra  # noqa: E402 - kept below MPS preamble (must run before jax)
import jax  # noqa: E402
import nvtx  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import mpm_jax.p2g.backends  # noqa: F401, E402 - registers Hydra backend config choices
import mpm_jax.resolvers  # noqa: F401, E402 - registers OmegaConf resolvers (e.g. ppc_grid)
import postprocessing  # noqa: F401, E402 - registers Hydra `plot` config group via hydra-zen
from mpm_jax.profiling import NVTX_DOMAIN  # noqa: E402
from mpm_jax.rendering import render_warp_opengl  # noqa: E402
from mpm_jax.solver import MPMSolver  # noqa: E402

logger = logging.getLogger(__name__)


def _slugify(value):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def _current_gpu_kind():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return _slugify(result.stdout.splitlines()[0])
    except Exception:
        return "unknown"


OmegaConf.register_new_resolver("gpu_kind", _current_gpu_kind, replace=True)


def _analysis_csv_path(metrics):  # this should not be needed.
    job_num = metrics.get("hydra_job_num")
    if job_num is None:
        return None
    original_cwd = Path(hydra.utils.get_original_cwd())
    sweep_dir = Path(metrics["hydra_sweep_dir"])  # doesn't belong in metrics...
    if not sweep_dir.is_absolute():
        sweep_dir = original_cwd / sweep_dir
    try:
        runs_index = sweep_dir.parts.index("runs")
        sweep_root = Path(*sweep_dir.parts[:runs_index])
    except ValueError:
        sweep_root = sweep_dir
    return sweep_root / "results.csv"


def _write_csv_row(
    path, metrics
):  # this should not be needed. metrics should be a dataclass that can be serialized directly and saved..
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)


def _write_metrics(run_dir, metrics):
    results_path = Path(run_dir) / "results.json"
    metrics["results_json_path"] = str(results_path)  # not needed

    analysis_csv_path = _analysis_csv_path(metrics)
    metrics["analysis_csv_path"] = (
        str(analysis_csv_path) if analysis_csv_path else None
    )  # not needed

    results_path.write_text(json.dumps(metrics, indent=2))
    (Path(run_dir) / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    _write_csv_row(Path(run_dir) / "metrics.csv", metrics)

    if analysis_csv_path is not None:
        _write_csv_row(analysis_csv_path, metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # MPS thread-percentage clamp is set by the preamble at the top of this
    # file (must happen before `import jax`). Echo the value here for the log.
    mps_pct = cfg.get("mps_thread_percent")
    if mps_pct is not None:
        logger.info("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=%d", int(mps_pct))

    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    render_enabled = bool(cfg.render.get("enabled", True))

    profile_cfg = cfg.get("profile", {})
    profile_enabled = bool(profile_cfg.get("enabled", False))

    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))

    # Warmup (JIT compilation) always runs outside any trace.
    solver.warmup(int(profile_cfg.get("warmup_frames", 1)))

    def run_measured_solve():
        solve_range = f"{solver.backend.name}_solve"
        with nvtx.annotate(solve_range, domain=NVTX_DOMAIN):
            return solver.run(capture_frames=render_enabled)

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
    metrics["tag"] = cfg.get("tag")
    metrics["mps_thread_percent"] = cfg.get("mps_thread_percent")
    metrics["render_enabled"] = render_enabled
    metrics["gpu_kind"] = _slugify(metrics["gpu_type"])
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
        metrics["render_path"] = export_path  # not needed
    elif not render_enabled:
        logger.info("Rendering disabled.")

    _write_metrics(run_dir, metrics)


if __name__ == "__main__":
    main()
