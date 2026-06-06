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


def _run_solver(solver, cfg: DictConfig, trace_dir=None, profile_opts=None):
    """Drive an MPMSolver: warmup, then benchmark or frame-capture loop.

    When ``trace_dir`` is set, the JAX profiler is started *after* warmup and
    stopped after the steady-state loop, so the one-time JIT compile +
    autotuning (and the Python import-tracer flood) stay out of the trace.
    """
    sim = cfg.sim
    kernel_name = solver.backend.name
    bench = cfg.get("benchmark", False)

    with jax.profiler.TraceAnnotation("warmup", kernel=kernel_name):
        solver.step()
        jax.block_until_ready(solver.state.x)
        solver.reset_to_initial()

    frames = []

    # Start the profiler AFTER warmup so only steady-state work is captured —
    # the one-time JIT compile + autotuning (and its import-tracer flood) stay
    # out of the window.
    tracing = trace_dir is not None
    if tracing:
        jax.profiler.start_trace(trace_dir, profiler_options=profile_opts)
        print(f"JAX profiler started (steady-state only) -> {trace_dir}")

    try:
        if bench:
            with jax.profiler.TraceAnnotation("benchmark", kernel=kernel_name):
                t0 = time.perf_counter()
                for frame in tqdm(range(sim.num_frames), desc="simulate"):
                    with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                        solver.step()
                jax.block_until_ready(solver.state.x)
                elapsed = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with jax.profiler.TraceAnnotation("render_loop", kernel=kernel_name):
                for frame in tqdm(range(sim.num_frames), desc="simulate"):
                    with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                        # capture current state BEFORE advancing (frame 0 == initial config)
                        frames.append(np.array(solver.state.x))
                        solver.step()
                        jax.block_until_ready(solver.state.x)
            elapsed = time.perf_counter() - t0
    finally:
        if tracing:
            jax.profiler.stop_trace()
            print(f"JAX trace saved (steady-state only) -> {trace_dir}")

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


def run(cfg: DictConfig, trace_dir=None, profile_opts=None):
    """Instantiate the runtime config, build the solver, and drive it.

    Returns (frames, elapsed, total_steps, summary).
    """
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    return _run_solver(solver, cfg, trace_dir=trace_dir, profile_opts=profile_opts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_profile_options(profile_cfg):
    """Translate the `conf/profile/jax.yaml` block into a ProfileOptions.

    python_tracer_level defaults to 0: the Python import/call tracer otherwise
    adds ~1e6 events and buries the device timeline. host_tracer_level=2 keeps
    the TraceMe annotations. The `gpu:` sub-block maps to CUPTI
    advanced_configuration knobs (NVTX, CUDA-graph tracing, event caps, PM
    sampling).
    """
    opts = jax.profiler.ProfileOptions()
    opts.python_tracer_level = int(profile_cfg.get("python_tracer_level", 0))
    opts.host_tracer_level = int(profile_cfg.get("host_tracer_level", 2))
    gpu = profile_cfg.get("gpu", {}) or {}
    adv = opts.advanced_configuration
    if gpu.get("nvtx"):
        adv["gpu_enable_nvtx_tracking"] = True
    if gpu.get("cuda_graph_trace"):
        adv["gpu_enable_cupti_activity_graph_trace"] = True
    if gpu.get("graph_node_mapping"):
        adv["gpu_dump_graph_node_mapping"] = True
    if gpu.get("max_activity_events"):
        adv["gpu_max_activity_api_events"] = int(gpu["max_activity_events"])
    if gpu.get("max_callback_events"):
        adv["gpu_max_callback_api_events"] = int(gpu["max_callback_events"])
    if gpu.get("pm_sample_counters"):
        adv["gpu_pm_sample_counters"] = str(gpu["pm_sample_counters"])
    # Reassign in case advanced_configuration returns a fresh container.
    opts.advanced_configuration = adv
    return opts


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    profile_name = cfg.get("profile", {}).get("name", "none")

    if profile_name not in ("none", "jax"):
        raise RuntimeError(
            f"Unsupported profile={profile_name!r}. Only profile=none, "
            "and profile=jax are supported."
        )

    # JAX profiler: resolve the trace dir + options here, but let the solver
    # driver start/stop the trace around the steady-state loop only, so warmup /
    # JIT compile / autotuning is excluded from the window.
    jax_trace_dir = None
    profile_opts = None
    if profile_name == "jax":
        # Hydra >=1.2 doesn't chdir, so use the output dir from HydraConfig.
        run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        jax_trace_dir = os.path.join(run_dir, "jax_trace")
        profile_opts = _build_profile_options(cfg.get("profile", {}))

    # Run simulation. When profiling, the driver wraps only the steady-state loop.
    frames, elapsed, total_steps, summary = run(
        cfg, trace_dir=jax_trace_dir, profile_opts=profile_opts
    )
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
        orig_cwd = hydra.utils.get_original_cwd()
        output_dir = os.path.join(orig_cwd, cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        render_cfg = cfg.get("render", {})
        fps = int(render_cfg.get("fps", 30))
        radius = float(render_cfg.get("point_radius", 0.008))
        export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.gif")
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
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
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
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
