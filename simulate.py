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


def _color_to_rgb(color):
    colors = {
        "blue": (0.25, 0.45, 1.0),
        "orange": (1.0, 0.55, 0.15),
        "white": (0.92, 0.94, 1.0),
    }
    if isinstance(color, str):
        return colors.get(color, (0.8, 0.8, 0.8))
    return tuple(color)


def visualize_frames_warp_usd(frames, export_path, color='white', radius=0.008, fps=30):
    import warp.render

    renderer = warp.render.UsdRenderer(export_path, up_axis='Z', fps=fps, scaling=1.0)
    renderer.render_ground(size=1.0)
    rgb = _color_to_rgb(color)
    for i, points in enumerate(frames):
        renderer.begin_frame(i / fps)
        renderer.render_points(
            name="particles",
            points=np.asarray(points, dtype=np.float32),
            radius=radius,
            colors=rgb,
            as_spheres=True,
        )
        renderer.end_frame()
    renderer.save()


def visualize_frames_warp_opengl(frames, export_path, color='white', radius=0.008,
                                 fps=30, width=960, height=720, write_usd_path=None):
    import imageio.v2 as imageio
    import pyglet

    pyglet.options['headless'] = True

    import warp as wp
    import warp.render

    renderer = wp.render.OpenGLRenderer(
        title="MPM-CudaJax",
        headless=True,
        screen_width=width,
        screen_height=height,
        near_plane=0.01,
        far_plane=10.0,
        camera_fov=32.0,
        camera_pos=(2.05, 2.05, 1.55),
        camera_front=(-0.62, -0.62, -0.46),
        camera_up=(0.0, 0.0, 1.0),
        background_color=(0.94, 0.95, 0.96),
        draw_grid=False,
        draw_sky=False,
        draw_axis=False,
        show_info=False,
        fps=fps,
        vsync=False,
    )

    rgb = _color_to_rgb(color)
    pixels = wp.zeros((height, width, 3), dtype=wp.float32)
    images = []

    try:
        for i, points in enumerate(frames):
            renderer.begin_frame(i / fps)
            renderer.render_plane(
                "floor",
                pos=(0.5, 0.5, 0.02),
                rot=(0.70710678, 0.0, 0.0, 0.70710678),
                width=0.65,
                length=0.65,
                color=(0.82, 0.84, 0.86),
            )
            renderer.render_points(
                name="particles",
                points=np.asarray(points, dtype=np.float32),
                radius=radius,
                colors=rgb,
                as_spheres=True,
            )
            renderer.end_frame()
            renderer.get_pixels(pixels, split_up_tiles=False, mode='rgb')
            img = np.clip(pixels.numpy() * 255.0, 0, 255).astype(np.uint8)
            images.append(img)
    finally:
        renderer.clear()

    imageio.mimsave(export_path, images, fps=fps)
    if write_usd_path:
        visualize_frames_warp_usd(frames, write_usd_path, color=color, radius=radius, fps=fps)


def _ensure_ovrtx_render_product(stage_path, width=960, height=540):
    from pxr import Gf, Usd, UsdGeom, UsdRender

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {stage_path}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    camera_path = "/Render/ViewCamera"
    product_path = "/Render/Camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(30.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    eye = Gf.Vec3d(2.4, 2.2, 1.7)
    target = Gf.Vec3d(0.5, 0.5, 0.28)
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    view = Gf.Matrix4d().SetLookAt(eye, target, up)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())

    stage.DefinePrim("/Render")
    camera_product = UsdRender.Product.Define(stage, product_path)
    camera_product.CreateResolutionAttr(Gf.Vec2i(width, height))
    camera_product.GetCameraRel().SetTargets([camera_path])
    camera_product.GetOrderedVarsRel().SetTargets(["/Render/Camera/LdrColor"])

    ldr_color = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    ldr_color.CreateSourceNameAttr("LdrColor")

    root = stage.GetPrimAtPath("/root")
    if root.IsValid():
        stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    return product_path


def visualize_frames_ovrtx_mp4(frames, export_path, color='white', radius=0.008,
                               fps=30, width=960, height=540, write_usd_path=None):
    import imageio.v2 as imageio

    usd_path = write_usd_path or os.path.splitext(export_path)[0] + ".ovrtx.usd"
    visualize_frames_warp_usd(frames, usd_path, color=color, radius=radius, fps=fps)
    product_path = _ensure_ovrtx_render_product(usd_path, width=width, height=height)

    import ovrtx

    print("Creating OVRTX renderer. First run may compile shaders...")
    renderer = ovrtx.Renderer()
    renderer.open_usd(str(usd_path))

    with imageio.get_writer(export_path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for frame in tqdm(range(len(frames)), desc='OVRTX render'):
            renderer.update_from_usd_time(frame / fps)
            products = renderer.step(
                render_products={product_path},
                delta_time=1.0 / fps,
            )
            product = products[product_path]
            if not product.frames:
                raise RuntimeError("OVRTX returned no frames.")
            var = product.frames[0].render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            writer.append_data(np.asarray(pixels[..., :3]))


# ---------------------------------------------------------------------------
# Unified run path
# ---------------------------------------------------------------------------

def _run_jax_solver(solver, cfg: DictConfig, trace_dir=None, profile_opts=None):
    """Drive an MPMSolver (JAX backend): warmup, then benchmark or GIF loop.

    When ``trace_dir`` is set, the JAX profiler is started *after* warmup and
    stopped after the steady-state loop, so the one-time JIT compile +
    autotuning (and the Python import-tracer flood) stay out of the trace.
    """
    import jax
    import jax.numpy as jnp

    sim = cfg.sim
    kernel_name = cfg.get('kernel', {}).get('name', 'jax_baseline')
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
                for frame in tqdm(range(sim.num_frames), desc='JAX'):
                    with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                        solver.step()
                jax.block_until_ready(solver.state.x)
                elapsed = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with jax.profiler.TraceAnnotation("render_loop", kernel=kernel_name):
                for frame in tqdm(range(sim.num_frames), desc='JAX'):
                    with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                        # capture current state BEFORE advancing (frame 0 == initial config)
                        frames.append(np.array(solver.state.x))
                        t_frame = time.perf_counter()
                        solver.step()
                        jax.block_until_ready(solver.state.x)
                        frame_ms = (time.perf_counter() - t_frame) * 1000
                        frame_metrics.append({
                            'x_mean_z': float(solver.state.x[:, 2].mean()),
                            'v_max': float(jnp.abs(solver.state.v).max()),
                            'frame_ms': frame_ms,
                            'timestep_ms': frame_ms,
                        })
            elapsed = time.perf_counter() - t0
    finally:
        if tracing:
            jax.profiler.stop_trace()
            print(f"JAX trace saved (steady-state only) -> {trace_dir}")

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


def run(cfg: DictConfig, trace_dir=None, profile_opts=None):
    """Construct the solver via the registry and drive it to results.

    Returns (frames, elapsed, total_steps, summary, frame_metrics).
    """
    from mpm_jax.registry import build_solver

    solver = build_solver(cfg)
    return _run_jax_solver(solver, cfg, trace_dir=trace_dir, profile_opts=profile_opts)



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
    import jax
    opts = jax.profiler.ProfileOptions()
    opts.python_tracer_level = int(profile_cfg.get('python_tracer_level', 0))
    opts.host_tracer_level = int(profile_cfg.get('host_tracer_level', 2))
    gpu = profile_cfg.get('gpu', {}) or {}
    adv = opts.advanced_configuration
    if gpu.get('nvtx'):
        adv['gpu_enable_nvtx_tracking'] = True
    if gpu.get('cuda_graph_trace'):
        adv['gpu_enable_cupti_activity_graph_trace'] = True
    if gpu.get('graph_node_mapping'):
        adv['gpu_dump_graph_node_mapping'] = True
    if gpu.get('max_activity_events'):
        adv['gpu_max_activity_api_events'] = int(gpu['max_activity_events'])
    if gpu.get('max_callback_events'):
        adv['gpu_max_callback_api_events'] = int(gpu['max_callback_events'])
    if gpu.get('pm_sample_counters'):
        adv['gpu_pm_sample_counters'] = str(gpu['pm_sample_counters'])
    # Reassign in case advanced_configuration returns a fresh container.
    opts.advanced_configuration = adv
    return opts


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    profile_name = cfg.get('profile', {}).get('name', 'none')
    kernel_name = cfg.get('kernel', {}).get('name', 'jax_baseline')

    if profile_name not in ('none', 'jax'):
        raise RuntimeError(
            f"Unsupported profile={profile_name!r}. Only profile=none, "
            "and profile=jax are supported."
        )

    # JAX profiler: resolve the trace dir + options here, but let the solver
    # driver start/stop the trace around the steady-state loop only, so warmup /
    # JIT compile / autotuning is excluded from the window.
    jax_trace_dir = None
    profile_opts = None
    if profile_name == 'jax':
        from hydra.core.hydra_config import HydraConfig
        # Hydra >=1.2 doesn't chdir, so use the output dir from HydraConfig.
        run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        jax_trace_dir = os.path.join(run_dir, "jax_trace")
        profile_opts = _build_profile_options(cfg.get('profile', {}))

    # Run simulation (construction routed through build_solver). When profiling,
    # the driver wraps only the steady-state loop.
    frames, elapsed, total_steps, summary, _frame_metrics = run(
        cfg, trace_dir=jax_trace_dir, profile_opts=profile_opts
    )

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    backend_label = "jax-loop"
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
        render_cfg = cfg.get('render', {})
        render_backend = render_cfg.get('backend', 'pyvista_gif')
        fps = int(render_cfg.get('fps', 30))
        radius = float(render_cfg.get('point_radius', 0.008))
        if render_backend == 'warp_opengl':
            export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.gif")
            usd_path = None
            if bool(render_cfg.get('write_usd', False)):
                usd_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.usd")
            print(f"\nRendering with Warp OpenGL to {export_path}...")
            visualize_frames_warp_opengl(
                frames, export_path, color=cfg.material.color, radius=radius,
                fps=fps, width=int(render_cfg.get('width', 960)),
                height=int(render_cfg.get('height', 720)), write_usd_path=usd_path,
            )
        elif render_backend == 'warp_usd':
            export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.usd")
            print(f"\nRendering with Warp USD to {export_path}...")
            visualize_frames_warp_usd(
                frames, export_path, color=cfg.material.color, radius=radius, fps=fps)
        elif render_backend == 'ovrtx_mp4':
            export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.mp4")
            usd_path = None
            if bool(render_cfg.get('write_usd', False)):
                usd_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.ovrtx.usd")
            print(f"\nRendering with OVRTX to {export_path}...")
            visualize_frames_ovrtx_mp4(
                frames, export_path, color=cfg.material.color, radius=radius,
                fps=fps, width=int(render_cfg.get('width', 960)),
                height=int(render_cfg.get('height', 540)), write_usd_path=usd_path,
            )
        elif render_backend == 'pyvista_gif':
            export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.gif")
            print(f"\nRendering to {export_path}...")
            visualize_frames(frames, export_path, size=[1, 1, 1], c=cfg.material.color, fps=fps)
        else:
            raise RuntimeError(f"Unsupported render.backend={render_backend!r}.")
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
