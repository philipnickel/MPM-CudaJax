"""Produce report snapshots of the MLS-MPM sand drop.

Runs the *standard benchmark physics* (``sim=benchmark``: 8M particles,
num_grids=124, CFL-safe dt=5e-5, sticky floor) with the fast Morton-sorted
CUDA P2G (``backend=cuda_v3_inline``) and captures still PNGs at five
physically-meaningful moments of the fall -> impact -> settle arc:

    initial -> impact-onset -> impact-peak -> impact-spread -> settled

Unlike ``simulate.py`` (which keeps every frame and writes a GIF), this script
keeps only a subsampled point cloud per frame for rendering, reads cheap scalar
metrics (min_z / mean_z / v_max) to *locate* the impact, then renders the chosen
five frames with a shared camera so the sequence is directly comparable.

Usage (gpu env):

    pixi run -e gpu python make_report_snapshots.py                 # full 8M run
    pixi run -e gpu python make_report_snapshots.py --frames 30 --no-render   # timing probe
    pixi run -e gpu python make_report_snapshots.py --grid 40 --particles 262144 --frames 700  # quick stable validation
    pixi run -e gpu python make_report_snapshots.py --select 0,256,300,420,640 # manual frames
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Keep pyvista headless/offscreen (EGL) regardless of import order.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

REPO = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Config / solver construction (reuses the project registry + benchmark preset)
# --------------------------------------------------------------------------- #
def build_cfg(args):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "sim=benchmark",
                "backend=cuda_v3_inline",
                "material=sand_jacobi",
            ],
        )
    # Override the run length / scale on the resolved benchmark preset.
    cfg.sim.num_frames = int(args.frames)
    cfg.sim.steps_per_frame = int(args.steps_per_frame)
    if args.grid:
        cfg.sim.num_grids = int(args.grid)
    if args.dt and args.dt > 0:
        cfg.sim.dt = float(args.dt)
    if args.size and args.size > 0:
        cfg.sim.size = [float(args.size)] * 3
    if args.center_z is not None and args.center_z >= 0:
        cfg.sim.center = [0.5, 0.5, float(args.center_z)]
    # 8-ppc particle budget from the block's cell footprint (keeps resolution).
    if args.ppc and args.ppc > 0 and args.size and args.size > 0:
        cells = round(float(args.size) * int(cfg.sim.num_grids))
        cfg.sim.n_particles = int(cells ** 3 * int(args.ppc))
    if args.particles:
        cfg.sim.n_particles = int(args.particles)
    return cfg


# --------------------------------------------------------------------------- #
# Simulation pass: step the arc, record metrics + a subsampled cloud per frame
# --------------------------------------------------------------------------- #
def run_arc(cfg, args, target_frames=None):
    import hydra
    import jax
    import jax.numpy as jnp
    from mpm_jax.solver import MPMSolver

    n = int(cfg.sim.n_particles)
    dt = float(cfg.sim.dt)
    spf = int(cfg.sim.steps_per_frame)
    frame_dt = dt * spf

    ctr, sz = list(cfg.sim.center), list(cfg.sim.size)
    margin = (1.0 - float(sz[0])) / 2.0  # lateral gap to the (wall-free) grid edge
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    print(f"Building solver: backend={solver.backend.name} n_particles={n:,} "
          f"num_grids={cfg.sim.num_grids} dx={1.0/float(cfg.sim.num_grids):.5f} dt={dt:g} "
          f"steps/frame={spf} frames={cfg.sim.num_frames}")
    print(f"  block center={ctr} size={sz} -> lateral margin to edge ~{margin:.3f} "
          f"({margin * float(cfg.sim.num_grids):.0f} cells)")

    # Deterministic spatial subsample used for *every* snapshot (positions are
    # gathered on-device then copied, so per-frame host transfer stays small).
    pts = min(int(args.points), n)
    rng = np.random.RandomState(0)
    stash_idx = jnp.asarray(np.sort(rng.choice(n, pts, replace=False)))

    def metrics(state):
        z = state.x[:, 2]
        return (float(z.min()), float(z.mean()),
                float(jnp.abs(state.v).max()))

    def stash(state):
        return np.asarray(state.x[stash_idx], dtype=np.float32)

    # Warmup: one compiled frame so the timed loop excludes JIT/autotune.
    t0 = time.perf_counter()
    solver.step()
    jax.block_until_ready(solver.state.x)
    solver.reset_to_initial()
    print(f"Warmup/compile: {time.perf_counter() - t0:.1f}s")

    num_frames = int(cfg.sim.num_frames)
    # clouds[k] = subsampled positions after k steps (k=0 -> initial). Metrics are
    # recorded every frame (cheap scalars), but clouds are stashed only at frames
    # we will render — `target_frames` keeps host RAM bounded on long runs.
    stash_set = None if target_frames is None else {int(t) for t in target_frames}
    clouds = {}
    rows = []            # (frame, t, min_z, mean_z, v_max)

    def record(k):
        mn, me, vx = metrics(solver.state)
        if stash_set is None or k in stash_set:
            clouds[k] = stash(solver.state)
        rows.append((k, k * frame_dt, mn, me, vx))

    record(0)  # initial config, before any step
    loop_t0 = time.perf_counter()
    last_report = loop_t0
    for f in range(num_frames):
        solver.step()
        jax.block_until_ready(solver.state.x)
        record(f + 1)
        now = time.perf_counter()
        if now - last_report > 5.0 or f == num_frames - 1:
            done = f + 1
            rate = (now - loop_t0) / done
            eta = rate * (num_frames - done)
            print(f"  frame {done}/{num_frames}  "
                  f"{rate * 1000 / spf:.2f} ms/step  "
                  f"min_z={rows[-1][2]:.4f} v_max={rows[-1][4]:.3f}  "
                  f"ETA {eta:.0f}s")
            last_report = now

    elapsed = time.perf_counter() - loop_t0
    total_steps = num_frames * spf
    print(f"\nArc done: {total_steps} steps in {elapsed:.1f}s "
          f"({elapsed / total_steps * 1000:.2f} ms/step)")

    metrics_arr = np.array([(r[2], r[3], r[4]) for r in rows], dtype=np.float64)
    times = np.array([r[1] for r in rows], dtype=np.float64)
    return clouds, metrics_arr, times, frame_dt


# --------------------------------------------------------------------------- #
# Frame selection: locate impact from the metric timeline
# --------------------------------------------------------------------------- #
ROLES = ["initial", "impact-onset", "impact-splat", "impact-spread", "post-impact"]


def select_frames(metrics_arr, times, cfg, args):
    """Return [(frame_index, role_label), ...] of length 5.

    Anchored to *floor contact* (min_z reaching the collider) rather than peak
    velocity: v_max climbs through the whole collapse and can spike on a few
    stray particles late, so it is a poor impact marker. min_z is monotone and
    clean. The three impact stages span a fixed physical window after onset, so
    the choice transfers across grid resolutions (the fall is gravity-driven and
    geometry-fixed, so timing is ~scale-invariant).
    """
    min_z = metrics_arr[:, 0]
    v_max = metrics_arr[:, 2]
    nframes = len(min_z) - 1

    if args.select:
        idxs = [int(s) for s in args.select.split(",")]
        return [(min(max(i, 0), nframes), ROLES[k] if k < len(ROLES) else f"frame-{i}")
                for k, i in enumerate(idxs)]

    # Floor height from the sticky surface collider; "touch" = within ~one cell.
    floor_z = 0.02
    for bc in cfg.sim.boundary_conditions:
        if bc.get("type") == "surface_collider":
            floor_z = float(bc["point"][2])
            break
    dx = 1.0 / float(cfg.sim.num_grids)
    touch_thr = floor_z + max(0.006, 0.8 * dx)

    below = np.where(min_z <= touch_thr)[0]
    onset = int(below[0]) if len(below) else int(np.argmin(min_z))

    # Impact window: a fixed physical duration after onset (frames).
    frame_dt = (times[1] - times[0]) if len(times) > 1 else 1.0
    win = int(round(args.impact_window / frame_dt)) if frame_dt > 0 else (nframes - onset)
    end = min(onset + win, nframes)

    # Flag late instability inside the window (a few particles can blow up in the
    # long tail); the timeline CSV + --select let you dodge it.
    seg = v_max[onset:end + 1]
    if len(seg) and float(seg.max()) > args.v_cap:
        bad = onset + int(np.argmax(seg > args.v_cap))
        print(f"  [warn] v_max exceeds {args.v_cap} at frame {bad} "
              f"(t={times[bad] * 1000:.0f} ms) inside the impact window — possible "
              f"late instability; inspect {args.tag}_timeline.csv or pass --select.")

    span = max(end - onset, 1)
    chosen = [
        (0, ROLES[0]),
        (onset, ROLES[1]),
        (onset + int(round(0.40 * span)), ROLES[2]),
        (onset + int(round(0.72 * span)), ROLES[3]),
        (end, ROLES[4]),
    ]
    # Clamp + force strictly-increasing distinct indices.
    seen, out = set(), []
    for i, label in chosen:
        i = min(max(i, 0), nframes)
        while i in seen and i < nframes:
            i += 1
        seen.add(i)
        out.append((i, label))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
# Shared camera so the 5 stills line up. (eye, focal-point, view-up); the floor
# region is [0,1]^2, material settles near z<0.4, so aim slightly low.
CAMERA = [(2.55, 2.55, 1.65), (0.5, 0.5, 0.20), (0.0, 0.0, 1.0)]
ZOOM = 1.05
CMAP = "copper"
# Color-by-height range, fixed across frames for comparability. The low bound is
# pushed negative so floor-level particles map to warm brown (not pure black).
CLIM = (-0.35, 0.85)
FLOOR_COLOR = (0.82, 0.83, 0.85)

# Warp OpenGL camera (eye position + normalized look direction), shared across the
# 5 stills. Aimed at the domain center; slightly zoomed vs simulate.py's default.
OPENGL_CAMERA_POS = (1.85, 1.85, 1.40)
OPENGL_CAMERA_FRONT = (-0.61, -0.61, -0.50)
OPENGL_COLORS = {"orange": (1.0, 0.55, 0.15), "blue": (0.25, 0.45, 1.0),
                 "white": (0.92, 0.94, 1.0), "tan": (0.85, 0.62, 0.38)}


def render_snapshot(points, out_path, *, label, t, point_size):
    import pyvista as pv

    points = np.ascontiguousarray(points, dtype=np.float32)
    if not np.all(np.isfinite(points)):
        n_bad = int((~np.isfinite(points)).any(axis=1).sum())
        print(f"  [warn] {label}: {n_bad} non-finite points clamped")
        points = np.nan_to_num(points, nan=0.0, posinf=1.0, neginf=0.0)

    pdata = pv.PolyData(points)
    pdata["height"] = points[:, 2]

    p = pv.Plotter(off_screen=True, window_size=[1500, 1100])
    p.set_background("white")

    # Floor (sticky surface) + unit-cube wireframe for a constant frame of reference.
    floor = pv.Plane(center=(0.5, 0.5, 0.02), direction=(0, 0, 1), i_size=1.0, j_size=1.0)
    p.add_mesh(floor, color=FLOOR_COLOR, ambient=0.3, diffuse=0.6, specular=0.0)
    cube = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
    p.add_mesh(cube, style="wireframe", color=(0.7, 0.7, 0.72), line_width=1)

    p.add_mesh(
        pdata, scalars="height", cmap=CMAP, clim=CLIM,
        render_points_as_spheres=True, point_size=point_size,
        show_scalar_bar=False, ambient=0.25, diffuse=0.8, specular=0.15,
    )

    p.camera_position = CAMERA
    p.camera.zoom(ZOOM)
    p.add_text(f"{label}\nt = {t * 1000:.1f} ms", position="upper_left",
               font_size=15, color="black")
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    p.screenshot(out_path)
    p.close()


def _enrich_usd_realism(usd_path, sand_rgb=(0.80, 0.64, 0.42)):
    """Make a warp-generated USD look more photoreal: bind matte PBR materials to
    the sand + floor and tune the sky dome / warm angled sun. Pure pxr — must run
    before `import ovrtx`."""
    from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf
    stage = Usd.Stage.Open(usd_path)

    # Soft sky dome (ambient fill) + a warm, slightly-soft directional sun (key).
    # Kept modest so the floor doesn't blow out to white.
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/dome_light"))
    if dome:
        dome.GetIntensityAttr().Set(1.1)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.82, 0.86, 0.95))
    sun_prim = stage.GetPrimAtPath("/distant_light")
    sun = UsdLux.DistantLight(sun_prim)
    if sun:
        sun.GetIntensityAttr().Set(2.6)
        sun.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.94, 0.82))
        sun.CreateAngleAttr().Set(1.6)                       # soft shadow edges
        xf = UsdGeom.Xformable(sun_prim)
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-48.0, 12.0, 28.0))  # upper-front key

    def surface(path, color, rough, spec):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(spec)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    sand = surface("/root/mat_sand", sand_rgb, 0.95, 0.15)
    floor = surface("/root/mat_floor", (0.40, 0.38, 0.35), 0.92, 0.08)
    for path, mat in (("/root/particles/sphere", sand), ("/root/particles", sand),
                      ("/root/ground", floor)):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    # warp's ground is only size=1.0 (a small floating plane). Enlarge it so it
    # fills the frame and reaches the horizon instead of floating in the void.
    ground = stage.GetPrimAtPath("/root/ground")
    if ground and ground.IsValid():
        UsdGeom.Xformable(ground).AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 1.0))
    stage.GetRootLayer().Save()


def _set_rtx_background(usd_path, product_path, sky_rgb=(0.62, 0.70, 0.85)):
    """Force a solid sky-colour OVRTX background (default renders black). Authors
    the OmniRtxSettingsCommon background attrs on the render product + a settings
    prim — best-effort; harmless if OVRTX ignores them. Pure pxr."""
    from pxr import Usd, UsdRender, Sdf, Gf
    stage = Usd.Stage.Open(usd_path)

    def author(prim):
        prim.CreateAttribute("omni:rtx:background:source:type",
                             Sdf.ValueTypeNames.Token).Set("color")
        prim.CreateAttribute("omni:rtx:background:source:color",
                             Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*sky_rgb))

    prod = stage.GetPrimAtPath(product_path)
    if prod and prod.IsValid():
        author(prod)
    settings = UsdRender.Settings.Define(stage, "/Render/rtxSettings")
    author(settings.GetPrim())
    settings.CreateProductsRel().SetTargets([Sdf.Path(product_path)])
    stage.GetRootLayer().Save()


def render_selection_ovrtx(clouds, labels, times, args):
    """Ray-trace the selected clouds to per-frame PNGs via the project's OVRTX path.

    Writes a warp USD (points + ground + lights), optionally enriches it with PBR
    materials + tuned lighting (``--realism``), then ray-traces each frame. Renders
    at ``supersample x`` resolution and downscales (LANCZOS) to suppress the
    missing-denoiser grain on no-RT-core GPUs. Run in the `rendering` env.
    """
    import imageio.v2 as imageio
    from PIL import Image

    import simulate as S  # tested USD + render-product helpers

    ss = max(1, int(args.supersample))
    w, h = int(args.width), int(args.height)
    rw, rh = w * ss, h * ss

    frames = [np.ascontiguousarray(c, dtype=np.float32) for c in clouds]
    usd_path = os.path.join(args.out, f"{args.tag}_ovrtx.usd")
    # fps=1 => USD time-sample i lands exactly on integer time i (clean per-frame seek).
    S.visualize_frames_warp_usd(frames, usd_path, color=args.color, radius=args.radius, fps=1)
    if args.realism:
        _enrich_usd_realism(usd_path)
    product_path = S._ensure_ovrtx_render_product(usd_path, width=rw, height=rh)
    if args.realism:
        _set_rtx_background(usd_path, product_path)

    # Import ovrtx ONLY after every pxr/USD stage op above: importing it earlier
    # registers USD schema plugins that collide with usd-core's UsdVol (the
    # 'ParticleField' alias) and break Usd.Stage.CreateNew. (Mirrors simulate.py.)
    import ovrtx
    print(f"Creating OVRTX renderer (first run compiles shaders); render {rw}x{rh} -> {w}x{h}...")
    renderer = ovrtx.Renderer()
    renderer.open_usd(str(usd_path))

    written = []
    for n, (label, t) in enumerate(zip(labels, times), 1):
        renderer.update_from_usd_time(float(n - 1))  # fps=1 -> time == frame index
        products = renderer.step(render_products={product_path}, delta_time=1.0)
        product = products[product_path]
        if not product.frames:
            raise RuntimeError("OVRTX returned no frames.")
        var = product.frames[0].render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
        img = np.asarray(np.from_dlpack(var)[..., :3])
        if ss > 1:
            img = np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))
        out = os.path.join(args.out, f"{args.tag}_{n}_{label}.png")
        imageio.imwrite(out, img)
        print(f"  [{n}/{len(clouds)}] {out}  ({os.path.getsize(out) // 1024} KB)  "
              f"[{label}, t={float(t) * 1000:.1f} ms]")
        written.append(out)
    print("\nDone. OVRTX snapshots:")
    for ww in written:
        print(f"  {ww}")


def render_selection_opengl(clouds, labels, times, args):
    """Rasterize the selected clouds to per-frame PNGs via Warp's headless OpenGL
    renderer (shaded spheres + floor plane). No USD/pxr dependency. Run in gpu env.
    """
    import imageio.v2 as imageio
    import pyglet

    pyglet.options["headless"] = True
    import warp as wp
    import warp.render

    rgb = OPENGL_COLORS.get(args.color, OPENGL_COLORS["orange"])
    w, h = int(args.width), int(args.height)
    renderer = wp.render.OpenGLRenderer(
        title="MPM-CudaJax", headless=True, screen_width=w, screen_height=h,
        near_plane=0.01, far_plane=10.0, camera_fov=32.0,
        camera_pos=OPENGL_CAMERA_POS, camera_front=OPENGL_CAMERA_FRONT,
        camera_up=(0.0, 0.0, 1.0), background_color=(0.96, 0.97, 0.98),
        draw_grid=False, draw_sky=False, draw_axis=False, show_info=False, vsync=False,
    )
    pixels = wp.zeros((h, w, 3), dtype=wp.float32)
    written = []
    try:
        for n, (cloud, label, t) in enumerate(zip(clouds, labels, times), 1):
            pts = np.ascontiguousarray(cloud, dtype=np.float32)
            renderer.begin_frame(float(n))
            renderer.render_plane("floor", pos=(0.5, 0.5, 0.02),
                                  rot=(0.70710678, 0.0, 0.0, 0.70710678),
                                  width=0.55, length=0.55, color=(0.82, 0.84, 0.86))
            renderer.render_points(name="particles", points=pts,
                                   radius=float(args.radius), colors=rgb, as_spheres=True)
            renderer.end_frame()
            renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb")
            img = np.clip(pixels.numpy() * 255.0, 0, 255).astype(np.uint8)
            out = os.path.join(args.out, f"{args.tag}_{n}_{label}.png")
            imageio.imwrite(out, img)
            print(f"  [{n}/{len(clouds)}] {out}  ({os.path.getsize(out) // 1024} KB)  "
                  f"[{label}, t={float(t) * 1000:.1f} ms]")
            written.append(out)
    finally:
        renderer.clear()
    print("\nDone. OpenGL snapshots:")
    for ww in written:
        print(f"  {ww}")


def render_selection(clouds, labels, times, args):
    """Render a stack of (pts,3) clouds to numbered PNGs via the chosen backend."""
    print(f"\nRendering {len(clouds)} snapshots ({args.renderer}) -> {args.out}")
    if args.renderer == "ovrtx":
        return render_selection_ovrtx(clouds, labels, times, args)
    if args.renderer == "opengl":
        return render_selection_opengl(clouds, labels, times, args)
    written = []
    for n, (cloud, label, t) in enumerate(zip(clouds, labels, times), 1):
        out = os.path.join(args.out, f"{args.tag}_{n}_{label}.png")
        render_snapshot(cloud, out, label=label, t=float(t), point_size=args.point_size)
        print(f"  [{n}/{len(clouds)}] {out}  ({os.path.getsize(out) // 1024} KB)")
        written.append(out)
    print("\nDone. Snapshots:")
    for w in written:
        print(f"  {w}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=700,
                    help="number of recorded frames (each = steps-per-frame substeps)")
    ap.add_argument("--steps-per-frame", type=int, default=10)
    ap.add_argument("--grid", type=int, default=0, help="override num_grids (0=benchmark default 124)")
    ap.add_argument("--particles", type=int, default=0, help="override n_particles (0=benchmark default 8M)")
    ap.add_argument("--dt", type=float, default=0.0, help="override dt (0=benchmark default 5e-5)")
    ap.add_argument("--size", type=float, default=0.0,
                    help="block cube side in world units (0=benchmark default 0.8); smaller = roomier domain")
    ap.add_argument("--center-z", type=float, default=-1.0,
                    help="block center height (negative=benchmark default 0.5)")
    ap.add_argument("--ppc", type=int, default=0,
                    help="particles per cell; with --size, sets n_particles = (size*grid)^3 * ppc")
    ap.add_argument("--points", type=int, default=900_000,
                    help="subsample size per snapshot (stash + render)")
    ap.add_argument("--renderer", choices=["pyvista", "opengl", "ovrtx"], default="pyvista",
                    help="pyvista height-colored points, Warp OpenGL shaded spheres, or OVRTX ray-traced")
    ap.add_argument("--point-size", type=float, default=3.0, help="pyvista point size")
    ap.add_argument("--radius", type=float, default=0.0045, help="ovrtx sphere radius (world units)")
    ap.add_argument("--width", type=int, default=1920, help="ovrtx image width")
    ap.add_argument("--height", type=int, default=1080, help="ovrtx image height")
    ap.add_argument("--color", type=str, default="orange", help="ovrtx particle color")
    ap.add_argument("--supersample", type=int, default=1,
                    help="ovrtx: render at NxN resolution and downscale (denoise-free grain fix)")
    ap.add_argument("--no-realism", dest="realism", action="store_false",
                    help="ovrtx: skip PBR-material + lighting enrichment (plain warp USD)")
    ap.set_defaults(realism=True)
    ap.add_argument("--impact-window", type=float, default=0.215,
                    help="seconds after floor-contact onset that the 3 impact stages span")
    ap.add_argument("--v-cap", type=float, default=6.0,
                    help="warn if v_max exceeds this inside the impact window (late instability)")
    ap.add_argument("--n-snapshots", type=int, default=0,
                    help="if >0, render N frames evenly spaced in time over the whole run "
                         "(labelled by time) instead of the 5-stage auto-selection")
    ap.add_argument("--select", type=str, default="",
                    help="manual comma-separated frame indices, e.g. 0,256,300,420,640")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "report_snapshots"))
    ap.add_argument("--tag", type=str, default="sand_cuda_v3")
    ap.add_argument("--no-render", action="store_true", help="run + select only; skip PNGs")
    ap.add_argument("--from-npz", type=str, default="",
                    help="re-render PNGs from a saved *_clouds.npz (skips the simulation)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Fast path: re-render from saved clouds (tune camera/point-size/color without
    # re-running the multi-minute sim).
    if args.from_npz:
        d = np.load(args.from_npz, allow_pickle=False)
        clouds_s, labels_s, times_s = d["clouds"], [str(x) for x in d["labels"]], d["times"]
        render_selection(clouds_s, labels_s, times_s, args)
        return

    cfg = build_cfg(args)
    # N-evenly-spaced mode: pre-compute the target frames so the sim stashes only
    # those clouds (host RAM stays bounded on long runs).
    nframes = int(cfg.sim.num_frames)
    target_frames = None
    if args.n_snapshots and args.n_snapshots > 0:
        target_frames = sorted({int(round(v))
                                for v in np.linspace(0, nframes, int(args.n_snapshots))})
    clouds, metrics_arr, times, frame_dt = run_arc(cfg, args, target_frames=target_frames)

    # Metric timeline -> CSV for the report appendix.
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"{args.tag}_timeline.csv")
    with open(csv_path, "w") as fh:
        fh.write("frame,t_s,min_z,mean_z,v_max\n")
        for k in range(len(times)):
            fh.write(f"{k},{times[k]:.6f},{metrics_arr[k,0]:.6f},"
                     f"{metrics_arr[k,1]:.6f},{metrics_arr[k,2]:.6f}\n")
    print(f"Wrote timeline: {csv_path}")

    if target_frames is not None:
        chosen = [(f, f"t{times[f] * 1000:04.0f}ms") for f in target_frames]
    else:
        chosen = select_frames(metrics_arr, times, cfg, args)
    print(f"\nSelected {len(chosen)} frames:")
    for i, label in chosen:
        print(f"  frame {i:4d}  t={times[i]*1000:7.1f} ms  "
              f"min_z={metrics_arr[i,0]:.4f}  v_max={metrics_arr[i,2]:.3f}  [{label}]")

    # Persist the selected clouds so rendering can be re-tuned without re-sim.
    sel_clouds = np.stack([clouds[i] for i, _ in chosen])
    sel_labels = [label for _, label in chosen]
    sel_times = np.array([times[i] for i, _ in chosen])
    npz_path = os.path.join(args.out, f"{args.tag}_clouds.npz")
    np.savez_compressed(npz_path, clouds=sel_clouds,
                        labels=np.array(sel_labels), times=sel_times)
    print(f"Saved clouds: {npz_path}  (re-render with --from-npz {npz_path})")

    if args.no_render:
        print("\n--no-render: skipping PNG output.")
        return

    render_selection(sel_clouds, sel_labels, sel_times, args)


if __name__ == "__main__":
    main()
