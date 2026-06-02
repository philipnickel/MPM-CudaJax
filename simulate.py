import pyvista as pv
import os
import time
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def visualize_frames(frames, export_path, center=[0.5, 0.5, 0.5],
                     size=[2.0, 2.0, 2.0], c='blue', s=20, fps=30):
    try:
        # Need to start xvfb for pyvista offscreen rendering to work without a display
        # But we can also set the VTK render window to offscreen before plotting
        pv.start_xvfb()
    except Exception:
        pass

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(export_path)

    # Initialize point cloud
    points = frames[0]
    cloud = pv.PolyData(points)
    plotter.add_mesh(cloud, color=c, point_size=s, render_points_as_spheres=True)

    # Add bounding box
    bounds = [
        center[0] - size[0]/2, center[0] + size[0]/2,
        center[1] - size[1]/2, center[1] + size[1]/2,
        center[2] - size[2]/2, center[2] + size[2]/2
    ]
    box = pv.Box(bounds)
    plotter.add_mesh(box, style='wireframe', color='black')

    plotter.camera_position = 'iso'
    plotter.show(auto_close=False)

    for i in range(len(frames)):
        cloud.points = frames[i]
        plotter.add_text(f"Frame {i}", position="upper_left", name="time_label")
        plotter.write_frame()

    plotter.close()


# ---------------------------------------------------------------------------
# Unified run path
# ---------------------------------------------------------------------------

def _maybe_enable_cuda_graphs(cfg: DictConfig):
    """Toggle XLA command-buffer capture (= CUDA Graphs) when requested.

    Must be called BEFORE the first `import jax`, otherwise XLA has already
    parsed XLA_FLAGS and the new value is ignored. Routed from main() before
    any jax import, and gated by the _MPM_INSIDE_PROFILER ordering in main().

    The cuda_v3_inline pipeline (Morton sort + warp-shuffle inline scatter +
    fused G2P) gains a CUDA-Graph fast path when kernel.cuda_graph=true: we
    ask XLA to wrap FUSION, CUSTOM_CALL (our FFI scatter / fused G2P) and
    WHILE (the lax.scan substep loop) into command buffers, which the GPU
    runtime executes as a single replayed graph per substep.
    """
    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    if kernel_name != 'cuda_v3_inline':
        return
    if not cfg.get('kernel', {}).get('cuda_graph', False):
        return
    extra = "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE"
    cur = os.environ.get("XLA_FLAGS", "")
    if extra not in cur:
        os.environ["XLA_FLAGS"] = (cur + " " + extra).strip()
        print(f"cuda_v3_inline: enabling XLA CUDA Graph capture via XLA_FLAGS={os.environ['XLA_FLAGS']}")


def _run_jax_solver(solver, cfg: DictConfig):
    """Drive an MPMSolver (JAX backend): warmup, then benchmark or GIF loop."""
    import warp as wp
    import jax
    import jax.numpy as jnp

    sim = cfg.sim
    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    bench = cfg.get('benchmark', False)

    def _warmup_metrics(s):
        """Compile the per-frame metric reads so the first timed frame doesn't
        eat a one-shot trace+compile on jnp.mean / jnp.abs.max."""
        _ = float(s.x[:, 2].mean())
        _ = float(jnp.abs(s.v).max())

    with jax.profiler.TraceAnnotation("warmup", kernel=kernel_name):
        solver.step()
        jax.block_until_ready(solver.state.x)
        _warmup_metrics(solver.state)
        solver.reset_to_initial()

    frames = []
    frame_metrics = []

    if bench:
        with jax.profiler.TraceAnnotation("benchmark", kernel=kernel_name):
            t0 = time.perf_counter()
            for frame in tqdm(range(sim.num_frames), desc='JAX'):
                with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                    solver.step()
            jax.block_until_ready(solver.state.x)
            elapsed = time.perf_counter() - t0
    else:
        # Initialize Warp HashGrid for bookkeeping proof-of-concept.
        # HashGrid builds an acceleration structure around JAX positions.
        wp.init()
        grid = wp.HashGrid(dim_x=sim.num_grids, dim_y=sim.num_grids, dim_z=sim.num_grids)
        dx = float(solver.params.dx)

        def on_frame(frame, st):
            # Bookkeeping with Warp: copy jnp array into a wp array.
            # Zero-copy via DLPack since both are on the GPU; fall back to CPU.
            try:
                wp_x = wp.from_dlpack(st.x)
                grid.build(wp_x, radius=dx)
                frames.append(wp_x.numpy())
            except Exception:
                frames.append(np.array(st.x))
            frame_metrics.append({
                'x_mean_z': float(st.x[:, 2].mean()),
                'v_max': float(jnp.abs(st.v).max()),
                'frame_ms': 0.0,
                'timestep_ms': 0.0,
            })

        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("render_loop", kernel=kernel_name):
            for frame in tqdm(range(sim.num_frames), desc='JAX'):
                with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                    t_frame = time.perf_counter()
                    solver.step()
                    jax.block_until_ready(solver.state.x)
                    frame_ms = (time.perf_counter() - t_frame) * 1000
                    on_frame(frame, solver.state)
                    frame_metrics[-1]['frame_ms'] = frame_ms
                    frame_metrics[-1]['timestep_ms'] = frame_ms
        elapsed = time.perf_counter() - t0

    total_steps = sim.num_frames * solver.steps_per_frame
    avg_frame_ms = elapsed / sim.num_frames * 1000
    summary = {
        'timestep': {
            'mean_ms': avg_frame_ms,
            'std_ms': 0.0,
            'total_ms': elapsed * 1000,
            'count': sim.num_frames,
        }
    }
    return frames, elapsed, total_steps, summary, frame_metrics


def _run_warp_graph_solver(solver, cfg: DictConfig):
    """Drive a WarpGraphSolver (pure-Warp graph backend) via its engine."""
    import warp as wp

    sim = cfg.sim
    engine = solver._engine
    profile_name = cfg.get('profile', {}).get('name', 'none')
    profile_warp = profile_name == 'warp'

    frames = []
    frame_metrics = []
    total_steps = int(sim.num_frames) * int(solver.steps_per_frame)

    if profile_warp:
        result = engine.run_frames_with_graph_timing(int(sim.num_frames))
        elapsed = result.elapsed_s
        summary = {
            stage: {
                'mean_ms': result.phase_ms_per_frame[stage],
                'std_ms': 0.0,
                'total_ms': result.phase_total_ms[stage],
                'count': sim.num_frames,
            }
            for stage in result.phase_total_ms
        }
        print("\nWarp graph event timing (inside captured graph):")
        for stage, ms_per_step in sorted(result.phase_ms_per_step.items(), key=lambda x: -x[1]):
            print(f"  {stage:15s}: {ms_per_step:8.3f} ms/step")
        return frames, elapsed, total_steps, summary, frame_metrics

    if cfg.get('benchmark', False):
        result = engine.run_frames(int(sim.num_frames))
        elapsed = result.elapsed_s
    else:
        t0 = time.perf_counter()
        for frame in tqdm(range(sim.num_frames), desc='Warp'):
            t_frame = time.perf_counter()
            engine.launch_frame()
            wp.synchronize_device(engine.device)
            frame_ms = (time.perf_counter() - t_frame) * 1000
            x_np = engine.x.numpy()
            v_np = engine.v.numpy()
            frames.append(x_np)
            frame_metrics.append({
                'x_mean_z': float(x_np[:, 2].mean()),
                'v_max': float(np.abs(v_np).max()),
                'frame_ms': frame_ms,
                'timestep_ms': frame_ms,
            })
        elapsed = time.perf_counter() - t0

    avg_frame_ms = elapsed / sim.num_frames * 1000
    summary = {
        'timestep': {
            'mean_ms': avg_frame_ms,
            'std_ms': 0.0,
            'total_ms': elapsed * 1000,
            'count': sim.num_frames,
        }
    }
    return frames, elapsed, total_steps, summary, frame_metrics


def run(cfg: DictConfig):
    """Construct the solver via the registry and drive it to results.

    Returns (frames, elapsed, total_steps, summary, frame_metrics).
    """
    from mpm_jax.registry import build_solver
    from mpm_jax.solver import WarpGraphSolver

    solver = build_solver(cfg)
    if isinstance(solver, WarpGraphSolver):
        return _run_warp_graph_solver(solver, cfg)
    return _run_jax_solver(solver, cfg)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    profile_name = cfg.get('profile', {}).get('name', 'none')

    if profile_name not in ('none', 'jax', 'warp'):
        raise RuntimeError(
            f"Unsupported profile={profile_name!r}. Only profile=none, "
            "profile=jax, and profile=warp are supported."
        )

    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    is_warp_bonus = kernel_name in {'warp_bonus_graph', 'warp_bonus_v2_graph'}

    # CUDA Graphs toggle must happen before any `import jax` in this process
    # — including the profile=jax branch a few lines down and the registry
    # import inside run(cfg) (which pulls in jax.numpy).
    _maybe_enable_cuda_graphs(cfg)

    if is_warp_bonus and profile_name == 'jax':
        raise RuntimeError(f"kernel={kernel_name} is pure Warp and does not emit a JAX trace.")
    if profile_name == 'warp' and not is_warp_bonus:
        raise RuntimeError(f"profile=warp is only supported for pure Warp kernels, got kernel={kernel_name}.")

    # JAX profiler (in-process, writes TensorBoard trace)
    jax_trace_dir = None
    if profile_name == 'jax':
        import jax
        from hydra.core.hydra_config import HydraConfig
        # Hydra >=1.2 doesn't chdir, so use the output dir from HydraConfig.
        run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        jax_trace_dir = os.path.join(run_dir, "jax_trace")
        jax.profiler.start_trace(jax_trace_dir)
        print(f"JAX profiler started -> {jax_trace_dir}")

    # Run simulation (construction routed through build_solver).
    frames, elapsed, total_steps, summary, _frame_metrics = run(cfg)

    # Stop JAX profiler
    if profile_name == 'jax':
        import jax
        jax.profiler.stop_trace()
        print(f"JAX trace saved to {jax_trace_dir}")

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    backend_label = "warp" if is_warp_bonus else "jax"
    print(f"\n{backend_label} ({kernel_name}): {total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)")

    total_ms = sum(s['total_ms'] for s in summary.values())
    print(f"\nWall-clock timing (per frame, {cfg.sim.steps_per_frame} substeps each):")
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        pct = stats['total_ms'] / total_ms * 100 if total_ms > 0 else 0
        print(f"  {stage:15s}: {stats['mean_ms']:8.3f} ms/frame ({pct:5.1f}%  std={stats['std_ms']:.3f}  n={stats['count']})")

    # Render GIF (skip in benchmark mode)
    export_path = None
    if not cfg.get('benchmark', False) and frames:
        orig_cwd = hydra.utils.get_original_cwd()
        output_dir = os.path.join(orig_cwd, cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.gif")
        print(f"\nRendering to {export_path}...")
        visualize_frames(frames, export_path, size=[1, 1, 1], c=cfg.material.color)
    elif cfg.get('benchmark', False):
        print("\nBenchmark mode: skipping GIF rendering.")

    # Dump a small results.json into the Hydra run dir so multirun callbacks
    # and post-hoc aggregation can pick up the per-run numbers. One file per
    # run, fixed shape.
    import json
    from hydra.core.hydra_config import HydraConfig
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    mat_name = cfg.get('material', {}).get('elasticity', {}).get('name', None) \
        or cfg.get('material', {}).get('name', 'unknown')
    results = {
        'kernel': kernel_name,
        'material_elasticity': mat_name,
        'n_particles': int(cfg.sim.n_particles),
        'num_grids': int(cfg.sim.num_grids),
        'num_frames': int(cfg.sim.num_frames),
        'steps_per_frame': int(cfg.sim.steps_per_frame),
        'total_steps': int(total_steps),
        'elapsed_s': float(elapsed),
        'ms_per_step': float(ms_per_step),
        'steps_per_sec': float(steps_per_sec),
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
