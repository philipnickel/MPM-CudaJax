"""Rendering helpers used by the Hydra simulation entrypoint."""

from importlib import import_module

import numpy as np
from tqdm import tqdm

_COLORS = {
    "blue": (0.25, 0.45, 1.0),
    "orange": (1.0, 0.55, 0.15),
    "white": (0.92, 0.94, 1.0),
}

_CAMERA = {
    "camera_pos": (2.05, 2.05, 1.55),
    "camera_front": (-0.62, -0.62, -0.46),
    "camera_up": (0.0, 0.0, 1.0),
}

_FLOOR = {
    "pos": (0.5, 0.5, 0.02),
    "rot": (0.70710678, 0.0, 0.0, 0.70710678),
    "width": 0.65,
    "length": 0.65,
    "color": (0.82, 0.84, 0.86),
}


def _color_to_rgb(color):
    if isinstance(color, str):
        return _COLORS.get(color, (0.8, 0.8, 0.8))
    return tuple(color)


def _load_warp_renderer():
    try:
        import imageio.v2 as imageio
        import pyglet
        import warp as wp
    except ImportError as exc:  # pragma: no cover - depends on optional renderer deps
        raise RuntimeError(
            "Warp OpenGL rendering requires imageio, pyglet, and warp-lang."
        ) from exc

    pyglet.options["headless"] = True
    import_module("warp.render")
    return imageio, wp


def _opengl_renderer(wp, *, width, height, fps):
    return wp.render.OpenGLRenderer(
        title="MPM-CudaJax",
        headless=True,
        screen_width=width,
        screen_height=height,
        near_plane=0.01,
        far_plane=10.0,
        camera_fov=32.0,
        background_color=(0.94, 0.95, 0.96),
        draw_grid=False,
        draw_sky=False,
        draw_axis=False,
        show_info=False,
        fps=fps,
        vsync=False,
        **_CAMERA,
    )


def render_warp_opengl(
    frames,
    export_path,
    *,
    color="white",
    radius=0.008,
    fps=30,
    width=960,
    height=720,
):
    """Render particle frames to a GIF using Warp's headless OpenGL renderer."""
    imageio, wp = _load_warp_renderer()
    renderer = _opengl_renderer(wp, width=width, height=height, fps=fps)
    pixels = wp.zeros((height, width, 3), dtype=wp.float32)
    rgb = _color_to_rgb(color)
    images = []

    try:
        for frame, points in enumerate(tqdm(frames, desc="render")):
            renderer.begin_frame(frame / fps)
            renderer.render_plane("floor", **_FLOOR)
            renderer.render_points(
                name="particles",
                points=np.asarray(points, dtype=np.float32),
                radius=radius,
                colors=rgb,
                as_spheres=True,
            )
            renderer.end_frame()
            renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb")
            images.append(np.clip(pixels.numpy() * 255.0, 0, 255).astype(np.uint8))
    finally:
        renderer.clear()

    imageio.mimsave(export_path, images, fps=fps)
