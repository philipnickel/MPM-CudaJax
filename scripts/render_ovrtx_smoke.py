from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdRender

import ovrtx


def _ensure_camera(stage_path: Path, *, width: int = 1280, height: int = 720) -> str:
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
    camera_to_world = view.GetInverse()
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(camera_to_world)

    stage.DefinePrim("/Render")
    camera_product = UsdRender.Product.Define(stage, product_path)
    camera_product.CreateResolutionAttr(Gf.Vec2i(width, height))
    camera_product.GetCameraRel().SetTargets([camera_path])
    camera_product.GetOrderedVarsRel().SetTargets(["/Render/Camera/LdrColor"])

    ldr_color = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
    ldr_color.CreateSourceNameAttr("LdrColor")

    stage.SetDefaultPrim(stage.GetPrimAtPath("/root"))
    stage.GetRootLayer().Save()
    return product_path


def render_png(
    input_usd: Path,
    output_png: Path,
    *,
    frame: int = 0,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    render_usd = output_png.with_suffix(".ovrtx.usd")
    shutil.copy2(input_usd, render_usd)
    product_path = _ensure_camera(render_usd, width=width, height=height)

    print("Creating OVRTX renderer. First run may compile shaders...", file=sys.stderr)
    renderer = ovrtx.Renderer()
    print(f"Opening {render_usd}...", file=sys.stderr)
    renderer.open_usd(str(render_usd))

    renderer.update_from_usd_time(frame / fps)
    products = renderer.step(render_products={product_path}, delta_time=1.0 / fps)

    if not products:
        raise RuntimeError("OVRTX did not return render products.")

    for _product_name, product in products.items():
        for rendered_frame in product.frames:
            var = rendered_frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU)
            pixels = np.from_dlpack(var)
            Image.fromarray(pixels).save(output_png)
            print(f"Saved {output_png}", file=sys.stderr)
            return
    raise RuntimeError("OVRTX returned no frames.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_usd", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render_png(
        args.input_usd,
        args.output_png,
        frame=args.frame,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
