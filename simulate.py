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
        camera_fov=35.0,
        camera_pos=(1.65, 1.65, 1.25),
        camera_front=(-0.62, -0.62, -0.48),
        camera_up=(0.0, 0.0, 1.0),
        background_color=(0.78, 0.82, 0.86),
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
            renderer.render_ground(size=1.0)
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
    kernel_name = cfg.get('kernel', {}).get('name', 'jax_v1_5')
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
    kernel_name = cfg.get('kernel', {}).get('name', 'jax_v1_5')
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

        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("render_loop", kernel=kernel_name):
            for frame in tqdm(range(sim.num_frames), desc='JAX'):
                with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                    # capture current state BEFORE advancing (frame 0 == initial config)
                    try:
                        wp_x = wp.from_dlpack(solver.state.x)
                        grid.build(wp_x, radius=float(solver.params.dx))
                        frames.append(wp_x.numpy())
                    except Exception:
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

    kernel_name = cfg.get('kernel', {}).get('name', 'jax_v1_5')
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
