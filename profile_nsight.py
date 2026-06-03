"""Hydra-driven Nsight Python profiler for current JAX-loop MPM backends."""

from __future__ import annotations

import itertools
import json
import os
import shlex
import sys
import sysconfig
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

_UNSUPPORTED_ANALYZE_CONFIG_KEYS = {"configs"}
_SCRIPT_NSIGHT_KEYS = {"phase", "write_json", "plot", "sweep", "configs", "analyze"}
_P2G_KERNELS = {
    "jax_v1_5",
    "cuda_v1_inline",
    "cuda_v2_inline",
    "cuda_v3_inline",
    "cuda_v4_inline",
    "warp_v3_supercell_tile",
}
_SPEED_OF_LIGHT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]


def _require_nsight():
    try:
        import nsight
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nsight-python is not installed. Run `pixi install -e gpu`, or "
            "`pixi run -e gpu python -m pip install nsight-python` for a "
            "one-off local install."
        ) from exc
    return nsight


def _maybe_enable_cuda_graphs(cfg: DictConfig):
    from simulate import _maybe_enable_cuda_graphs as enable_cuda_graphs

    enable_cuda_graphs(cfg)


def _build_p2g_stage(cfg: DictConfig):
    _maybe_enable_cuda_graphs(cfg)

    import jax
    import jax.numpy as jnp

    from mpm_jax.blocks.grid import build_grid_x
    from mpm_jax.blocks.init import get_particles
    from mpm_jax.boundary import build_boundary_fns
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.registry import KERNELS
    from mpm_jax.types import MPMState, make_params

    kernel_name = str(cfg.kernel.name)
    if kernel_name not in KERNELS:
        raise RuntimeError(f"Unsupported P2G kernel={kernel_name!r}.")

    sim = cfg.sim
    mat = cfg.material
    n = int(sim.n_particles)
    params = make_params(
        n_particles=n,
        num_grids=int(sim.num_grids),
        dt=float(sim.dt),
        gravity=list(sim.gravity),
        rho=float(sim.rho),
        clip_bound=float(sim.clip_bound),
        damping=float(sim.damping),
        center=list(sim.center),
        size=list(sim.size),
    )
    particles = jnp.array(
        get_particles(n, center=list(sim.center), size=list(sim.size)),
        dtype=jnp.float32,
    )
    grid_x = build_grid_x(params.num_grids)
    pre_fn, _ = build_boundary_fns(
        list(sim.boundary_conditions), grid_x, params.dx, particles, params.dt, params.p_mass)
    elasticity_fn = get_constitutive(mat.elasticity)
    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity), dtype=jnp.float32), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    spec = KERNELS[kernel_name]
    frame_opts = dict(spec.defaults)
    for key in ("loop_kind", "cuda_graph", "graph_mode"):
        if key in cfg.kernel:
            frame_opts[key] = cfg.kernel[key]
    backend = spec.backend_factory(num_grids=params.num_grids, **frame_opts)

    @jax.jit
    def jit_p2g_stage(state):
        x, v = pre_fn(state.x, state.v, 0.0)
        state = state._replace(x=x, v=v)
        stress = elasticity_fn(state.F)
        prepared = backend.prepare(params, state, stress)
        return backend.p2g(params, prepared)

    warmup = jit_p2g_stage(state)
    jax.block_until_ready(warmup)
    return jit_p2g_stage, state


def _p2g_runner(cfg: DictConfig, nsight):
    import jax

    jit_p2g_stage, state = _build_p2g_stage(cfg)
    annotation_name = f"{cfg.kernel.name}_p2g"

    def run_p2g_once():
        with nsight.annotate(annotation_name):
            out = jit_p2g_stage(state)
            jax.block_until_ready(out)

    return run_p2g_once


def _variant_value(variant: Mapping, path: str, default):
    cursor = variant
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _sweep_values(mapping: Mapping, key: str, default):
    value = mapping.get(key, default)
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _merge_variant_cfg(
    base_cfg: DictConfig,
    *,
    kernel_name: str,
    n_particles: int,
    num_grids: int,
    steps_per_frame: int,
):
    variant_cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(base_cfg, resolve=True)))
    variant_cfg.kernel.name = str(kernel_name)
    variant_cfg.sim.n_particles = int(n_particles)
    variant_cfg.sim.num_grids = int(num_grids)
    variant_cfg.sim.steps_per_frame = int(steps_per_frame)
    return variant_cfg


def _sweep_kernel_names(cfg: DictConfig):
    base_kernel = cfg.get("kernel", {}).get("name", "jax_v1_5")
    sweep = cfg.nsight.get("sweep", None)
    if sweep is not None:
        sweep_dict = OmegaConf.to_container(sweep, resolve=True)
        if not isinstance(sweep_dict, Mapping):
            raise RuntimeError("nsight.sweep must be a mapping of parameter lists.")
        return [str(value) for value in _sweep_values(sweep_dict, "kernels", [base_kernel])]

    configs = cfg.nsight.get("configs", None)
    if configs is not None:
        kernels = []
        for variant in OmegaConf.to_container(configs, resolve=True):
            if not isinstance(variant, Mapping):
                raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
            kernel_name = str(_variant_value(variant, "kernel.name", base_kernel))
            if kernel_name not in kernels:
                kernels.append(kernel_name)
        return kernels or [base_kernel]

    return [str(base_kernel)]


def _nsight_configs(cfg: DictConfig):
    base_n = int(cfg.sim.n_particles)
    base_g = int(cfg.sim.num_grids)
    base_steps = int(cfg.sim.steps_per_frame)

    sweep = cfg.nsight.get("sweep", None)
    if sweep is not None:
        sweep_dict = OmegaConf.to_container(sweep, resolve=True)
        if not isinstance(sweep_dict, Mapping):
            raise RuntimeError("nsight.sweep must be a mapping of parameter lists.")
        n_particles = [int(value) for value in _sweep_values(sweep_dict, "n_particles", [base_n])]
        num_grids = [int(value) for value in _sweep_values(sweep_dict, "num_grids", [base_g])]
        steps_per_frame = [
            int(value) for value in _sweep_values(sweep_dict, "steps_per_frame", [base_steps])
        ]
        return list(itertools.product(n_particles, num_grids, steps_per_frame))

    configs = cfg.nsight.get("configs", None)
    if configs is None:
        return None
    if not isinstance(configs, ListConfig | list):
        raise RuntimeError("nsight.configs must be a list of Hydra override mappings.")
    nsight_configs = []
    for variant in OmegaConf.to_container(configs, resolve=True):
        if not isinstance(variant, Mapping):
            raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
        n_particles = int(_variant_value(variant, "sim.n_particles", base_n))
        num_grids = int(_variant_value(variant, "sim.num_grids", base_g))
        steps_per_frame = int(_variant_value(variant, "sim.steps_per_frame", base_steps))
        nsight_configs.append((n_particles, num_grids, steps_per_frame))
    return nsight_configs


def _value_for_metric(metric_values, metrics: list[str], metric: str):
    if metric not in metrics:
        raise RuntimeError(
            f"Configured derive_metric requires metric {metric!r}. "
            f"Configured metrics: {metrics}"
        )
    return float(metric_values[metrics.index(metric)])


def _n_particles_from_config(config_values):
    if not config_values:
        raise RuntimeError("Expected config values to include n_particles.")
    if isinstance(config_values[0], str):
        if len(config_values) < 2:
            raise RuntimeError("Expected legacy config values to include kernel_name and n_particles.")
        return int(config_values[1])
    return int(config_values[0])


def _p2g_throughput_metric(metrics: list[str]):
    def derive_p2g_throughput(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics):]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
        }

    return derive_p2g_throughput


def _speed_of_light_metric(metrics: list[str]):
    def derive_speed_of_light(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics):]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        sm_pct = _value_for_metric(
            metric_values,
            metrics,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        compute_memory_pct = _value_for_metric(
            metric_values,
            metrics,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        )
        dram_pct = _value_for_metric(
            metric_values,
            metrics,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
            "sol_sm_pct": sm_pct,
            "sol_compute_memory_pct": compute_memory_pct,
            "sol_dram_pct": dram_pct,
            "sol_max_pct": max(sm_pct, compute_memory_pct, dram_pct),
        }

    return derive_speed_of_light


def _derive_metric(name, metrics: list[str]):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError("nsight.analyze.derive_metric must be null or a supported preset name.")
    if name in {"throughput", "p2g_throughput"}:
        if "gpu__time_duration.sum" not in metrics:
            raise RuntimeError(
                "derive_metric='throughput' requires "
                "nsight.analyze.metrics=[gpu__time_duration.sum, ...]."
            )
        return _p2g_throughput_metric(metrics)
    if name in {"speed_of_light", "sol"}:
        missing = [metric for metric in _SPEED_OF_LIGHT_METRICS if metric not in metrics]
        if missing:
            raise RuntimeError(
                "derive_metric='speed_of_light' requires these nsight.analyze.metrics: "
                + ", ".join(missing)
            )
        return _speed_of_light_metric(metrics)
    raise RuntimeError(
        f"Unsupported nsight.analyze.derive_metric={name!r}; "
        "supported presets: throughput, speed_of_light"
    )


def _combine_kernel_metrics(name):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError("nsight.analyze.combine_kernel_metrics must be null or a preset name.")
    if name in {"sum", "add"}:
        return lambda x, y: x + y
    if name == "max":
        return max
    if name == "min":
        return min
    raise RuntimeError(
        f"Unsupported combine_kernel_metrics={name!r}; supported presets: sum, max, min"
    )


def _nsight_analyze_kwargs(cfg: DictConfig, run_dir: Path, kernel_name: str):
    analyze_cfg = cfg.nsight.get("analyze", {})
    kwargs = OmegaConf.to_container(analyze_cfg, resolve=True)
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise RuntimeError("nsight.analyze must be a mapping of nsight.analyze.kernel options.")

    unsupported = _UNSUPPORTED_ANALYZE_CONFIG_KEYS.intersection(kwargs)
    if unsupported:
        keys = ", ".join(sorted(unsupported))
        raise RuntimeError(
            "The Hydra nsight.analyze block only supports YAML-serializable "
            f"nsight.analyze.kernel options; unsupported keys: {keys}."
        )

    kwargs = dict(kwargs)
    kwargs.setdefault("runs", 1)
    kwargs.setdefault("metrics", ["gpu__time_duration.sum"])
    kwargs["metrics"] = list(kwargs["metrics"])
    kwargs["derive_metric"] = _derive_metric(kwargs.get("derive_metric"), kwargs["metrics"])
    kwargs["combine_kernel_metrics"] = _combine_kernel_metrics(
        kwargs.get("combine_kernel_metrics")
    )
    kwargs.setdefault("output", "progress")
    kwargs.setdefault("output_csv", True)
    kwargs.setdefault("output_prefix", str(run_dir / f"nsight_{kernel_name}_p2g_"))
    kwargs.setdefault("configs", _nsight_configs(cfg))
    return kwargs


def _nsight_plot_kwargs(cfg: DictConfig, run_dir: Path):
    plot_cfg = cfg.nsight.get("plot", {})
    filename = Path(str(plot_cfg.get("filename", "nsight_plot.png")))
    if not filename.is_absolute():
        filename = run_dir / filename

    kwargs = OmegaConf.to_container(plot_cfg, resolve=True)
    kwargs.pop("enabled", None)
    kwargs["filename"] = str(filename)

    if "show_aggregate" not in kwargs:
        if kwargs.pop("show_avg", False):
            kwargs["show_aggregate"] = "avg"
        elif kwargs.pop("show_geomean", False):
            kwargs["show_aggregate"] = "geomean"
    else:
        kwargs.pop("show_avg", None)
        kwargs.pop("show_geomean", None)

    return kwargs


def _write_results(results, run_dir: Path, write_json: bool):
    df = results.to_dataframe()
    print("Nsight Python wrote raw and processed CSV files via output_csv=True.")
    print(df)

    if write_json:
        out_json = run_dir / "nsight_results.json"
        out_json.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2))
        print(f"Wrote {out_json}")


def _run_nsight_profile(profiled_func):
    try:
        return profiled_func()
    except Exception as exc:
        if "ERR_NVGPUCTRPERM" in str(exc):
            raise RuntimeError(
                "Nsight Compute denied access to GPU performance counters "
                "(ERR_NVGPUCTRPERM). Enable NVIDIA performance counter access "
                "for this host/user, then rerun this script. See "
                "https://developer.nvidia.com/ERR_NVGPUCTRPERM"
            ) from exc
        raise


def _prepare_nsight_child_python(run_dir: Path):
    """Run NCU's target Python without site `.pth` hooks."""
    if os.environ.get("NSPY_NCU_PROFILE"):
        return

    original_python = sys.executable
    wrapper = run_dir / "nsight_python_no_site.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(original_python)} -S \"$@\"\n"
    )
    wrapper.chmod(0o755)

    paths = []
    for path in [str(Path(__file__).resolve().parent), *sys.path]:
        if not path:
            continue
        resolved = str(Path(path).resolve())
        if resolved not in paths and Path(resolved).exists():
            paths.append(resolved)

    existing = os.environ.get("PYTHONPATH")
    if existing:
        for path in existing.split(os.pathsep):
            if path and path not in paths:
                paths.append(path)

    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ["NSPY_ORIGINAL_PYTHON"] = original_python
    sys.executable = str(wrapper)


@contextmanager
def _disable_editable_pth_for_nsight():
    """Temporarily hide the scikit-build editable hook from NCU target startup."""
    if os.environ.get("NSPY_NCU_PROFILE"):
        yield
        return

    purelib = Path(sysconfig.get_path("purelib"))
    pth = purelib / "_mpm_cudajax_editable.pth"
    disabled = purelib / f"_mpm_cudajax_editable.pth.nsight-disabled-{os.getpid()}"

    moved = False
    try:
        if pth.exists():
            pth.rename(disabled)
            moved = True
        yield
    finally:
        if moved and disabled.exists():
            disabled.rename(pth)


@hydra.main(version_base=None, config_path="conf", config_name="nsight_profile")
def main(cfg: DictConfig):
    nsight = _require_nsight()
    kernel_name = str(cfg.get("kernel", {}).get("name", "jax_v1_5"))
    phase = str(cfg.nsight.get("phase", "p2g"))
    if phase != "p2g":
        raise RuntimeError("profile_nsight.py now supports only nsight.phase=p2g.")

    configured_kernels = set(_sweep_kernel_names(cfg))
    unsupported = configured_kernels - _P2G_KERNELS
    if unsupported:
        supported = ", ".join(sorted(_P2G_KERNELS))
        raise RuntimeError(
            f"Unsupported P2G kernels: {', '.join(sorted(unsupported))}. "
            f"Supported kernels: {supported}"
        )

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    _prepare_nsight_child_python(run_dir)
    analyze_kwargs = _nsight_analyze_kwargs(cfg, run_dir, kernel_name)
    plot_enabled = bool(cfg.nsight.get("plot", {}).get("enabled", False))
    plot_kwargs = _nsight_plot_kwargs(cfg, run_dir) if plot_enabled else None

    def profiled_variant(n_particles, num_grids, steps_per_frame):
        for variant_kernel in _sweep_kernel_names(cfg):
            profile_cfg = _merge_variant_cfg(
                cfg,
                kernel_name=variant_kernel,
                n_particles=n_particles,
                num_grids=num_grids,
                steps_per_frame=steps_per_frame,
            )
            launcher = _p2g_runner(profile_cfg, nsight)
            launcher()

    profiled_variant = nsight.analyze.kernel(**analyze_kwargs)(profiled_variant)
    if plot_kwargs is not None:
        profiled_variant = nsight.analyze.plot(**plot_kwargs)(profiled_variant)

    print("Nsight profile config:")
    print(OmegaConf.to_yaml(cfg.nsight))
    unexpected = set(cfg.nsight.keys()) - _SCRIPT_NSIGHT_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unknown nsight config keys: {keys}.")
    with _disable_editable_pth_for_nsight():
        results = _run_nsight_profile(profiled_variant)
    _write_results(results, run_dir, write_json=bool(cfg.nsight.get("write_json", True)))
    if plot_kwargs is not None:
        print(f"Wrote {plot_kwargs['filename']}")


if __name__ == "__main__":
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
