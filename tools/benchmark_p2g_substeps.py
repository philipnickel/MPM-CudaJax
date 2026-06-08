"""Benchmark P2G prepare/scatter timing beside full solver timing.

Defaults to the benchmark sim preset (G=128, N=10M) and custom P2G backends.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import hydra
import jax
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mpm_jax.p2g.backends as backend_configs
import mpm_jax.resolvers  # noqa: F401 - registers OmegaConf resolvers (ppc_grid, gpu_kind)
from mpm_jax.p2g.backends.common import P2GBackend
from mpm_jax.solver import MPMSolver


logger = logging.getLogger(__name__)

DEFAULT_BACKENDS = (
    "cuda_v1",
    "cuda_v2",
    "cuda_v3",
    "CuTile",
)
DEFAULT_TIMING = {
    "backends": None,
    "warmup": 1,
    "repeats": 3,
    "full_solver": True,
    "full_solver_warmup_frames": 1,
    "output": "p2g_substep_timings.parquet",
}
TABLE_COLUMNS = [
    "backend",
    "prepare_ms",
    "scatter_ms",
    "prepare_plus_scatter_ms",
    "full_solver_ms_per_step",
    "n_particles",
    "num_grids",
    "steps_per_frame",
]


def _task_overrides():
    hydra_cfg = HydraConfig.get()
    return [str(item) for item in (OmegaConf.select(hydra_cfg, "overrides.task") or [])]


def _strip_override_prefix(item: str):
    return item.lstrip("+")


def _group_override(overrides: Iterable[str], group: str):
    prefix = f"{group}="
    for item in overrides:
        item = _strip_override_prefix(item)
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return None


def _sim_dot_overrides(overrides: Iterable[str]):
    return [
        _strip_override_prefix(item)
        for item in overrides
        if _strip_override_prefix(item).startswith("sim.")
    ]


def _with_default_benchmark_sim(cfg: DictConfig, overrides: list[str]):
    """Use sim=benchmark by default while preserving direct sim.* overrides."""
    if _group_override(overrides, "sim") is not None:
        return cfg

    root = Path(hydra.utils.get_original_cwd())
    benchmark_cfg = OmegaConf.load(root / "conf" / "sim" / "benchmark.yaml")
    benchmark_sim = OmegaConf.select(benchmark_cfg, "sim") or benchmark_cfg
    sim_override_cfg = OmegaConf.from_dotlist(_sim_dot_overrides(overrides))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.sim = OmegaConf.merge(benchmark_sim, sim_override_cfg.get("sim", {}))
    return cfg


def _timing_cfg(cfg: DictConfig, overrides: list[str]):
    timing = dict(DEFAULT_TIMING)
    user_timing = cfg.get("timing", {})
    if user_timing:
        timing.update(OmegaConf.to_container(user_timing, resolve=True))

    if timing["backends"] is None:
        backend_override = _group_override(overrides, "backend")
        timing["backends"] = [backend_override] if backend_override else list(DEFAULT_BACKENDS)

    timing["backends"] = _backend_list(timing["backends"])
    timing["warmup"] = max(1, int(timing["warmup"]))
    timing["repeats"] = max(1, int(timing["repeats"]))
    timing["full_solver"] = bool(timing["full_solver"])
    timing["full_solver_warmup_frames"] = max(1, int(timing["full_solver_warmup_frames"]))
    return timing


def _backend_list(value):
    if isinstance(value, str):
        backends = [item.strip() for item in value.split(",") if item.strip()]
    else:
        backends = [str(item) for item in value]
    supported = set(backend_configs.backend_choices())
    unsupported = [backend for backend in backends if backend not in supported]
    if unsupported:
        raise RuntimeError(
            "Unsupported timing backend(s): "
            + ", ".join(unsupported)
            + ". Choices: "
            + ", ".join(sorted(supported))
        )
    return backends


def _cfg_for_backend(cfg: DictConfig, backend: str):
    backend_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    backend_cfg.backend = backend_configs.backend_config(backend)
    return backend_cfg


def _block_until_ready(value):
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _time_ms(fn, *, repeats: int):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        _block_until_ready(result)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return sum(samples) / len(samples), samples


def _p2g_stage_timings(cfg: DictConfig, *, warmup: int, repeats: int):
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    params = solver.params
    backend = solver.backend
    state = solver.state
    elasticity_fn = solver.elasticity_fn

    @jax.jit
    def stress_fn(state):
        return elasticity_fn(state.F)

    @jax.jit
    def prepare_fn(state, stress):
        return backend.prepare(params, state, stress)

    @jax.jit
    def scatter_fn(prepared):
        return backend.scatter(params, prepared)

    @jax.jit
    def prepare_scatter_fn(state, stress):
        return backend.scatter(params, backend.prepare(params, state, stress))

    stress = _block_until_ready(stress_fn(state))
    passthrough_prepare = type(backend).prepare is P2GBackend.prepare
    if passthrough_prepare:
        prepared = backend.prepare(params, state, stress)
        for _ in range(warmup):
            _block_until_ready(scatter_fn(prepared))
        scatter_ms, scatter_samples = _time_ms(
            lambda: scatter_fn(prepared), repeats=repeats
        )
        return {
            "prepare_ms": 0.0,
            "scatter_ms": scatter_ms,
            "prepare_plus_scatter_ms": scatter_ms,
            "prepare_ms_samples": [0.0 for _ in scatter_samples],
            "scatter_ms_samples": scatter_samples,
            "prepare_plus_scatter_ms_samples": list(scatter_samples),
            "n_particles": params.n_particles,
            "num_grids": params.num_grids,
            "steps_per_frame": solver.steps_per_frame,
            "gpu_type": solver.gpu_type,
        }

    prepared = None
    for _ in range(warmup):
        prepared = _block_until_ready(prepare_fn(state, stress))
        _block_until_ready(scatter_fn(prepared))
        _block_until_ready(prepare_scatter_fn(state, stress))

    if prepared is None:
        prepared = _block_until_ready(prepare_fn(state, stress))

    prepare_ms, prepare_samples = _time_ms(
        lambda: prepare_fn(state, stress), repeats=repeats
    )
    prepared = _block_until_ready(prepare_fn(state, stress))
    scatter_ms, scatter_samples = _time_ms(lambda: scatter_fn(prepared), repeats=repeats)
    prepare_plus_scatter_ms, prepare_plus_scatter_samples = _time_ms(
        lambda: prepare_scatter_fn(state, stress), repeats=repeats
    )

    return {
        "prepare_ms": prepare_ms,
        "scatter_ms": scatter_ms,
        "prepare_plus_scatter_ms": prepare_plus_scatter_ms,
        "prepare_ms_samples": prepare_samples,
        "scatter_ms_samples": scatter_samples,
        "prepare_plus_scatter_ms_samples": prepare_plus_scatter_samples,
        "n_particles": params.n_particles,
        "num_grids": params.num_grids,
        "steps_per_frame": solver.steps_per_frame,
        "gpu_type": solver.gpu_type,
    }


def _full_solver_timing(cfg: DictConfig, *, warmup_frames: int):
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    solver.warmup(warmup_frames)
    _frames, elapsed = solver.run(capture_frames=False, progress=False)
    metrics = solver.metrics(elapsed)
    return {
        "full_solver_ms_per_step": metrics["ms_per_step"],
        "full_solver_elapsed_s": metrics["elapsed_s"],
        "full_solver_total_steps": metrics["total_steps"],
        "full_solver_particles_per_sec": metrics["particles_per_sec"],
    }


def _benchmark_backend(cfg: DictConfig, backend: str, timing: dict):
    backend_cfg = _cfg_for_backend(cfg, backend)
    logger.info(
        "Timing backend=%s n=%d G=%d steps/frame=%d",
        backend,
        int(backend_cfg.sim.n_particles),
        int(backend_cfg.sim.num_grids),
        int(backend_cfg.sim.steps_per_frame),
    )
    row = {"backend": backend}
    row.update(
        _p2g_stage_timings(
            backend_cfg,
            warmup=timing["warmup"],
            repeats=timing["repeats"],
        )
    )
    if timing["full_solver"]:
        row.update(
            _full_solver_timing(
                backend_cfg,
                warmup_frames=timing["full_solver_warmup_frames"],
            )
        )
    else:
        row.update(
            {
                "full_solver_ms_per_step": None,
                "full_solver_elapsed_s": None,
                "full_solver_total_steps": None,
                "full_solver_particles_per_sec": None,
            }
        )
    gc.collect()
    return row


def _write_results(df: pd.DataFrame, run_dir: Path, output: str):
    path = Path(output)
    if not path.is_absolute():
        path = run_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _display_table(df: pd.DataFrame):
    columns = [column for column in TABLE_COLUMNS if column in df.columns]
    table = df[columns].copy()
    logger.info(
        "\n%s",
        table.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    overrides = _task_overrides()
    cfg = _with_default_benchmark_sim(cfg, overrides)
    timing = _timing_cfg(cfg, overrides)
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()

    rows = []
    for backend in timing["backends"]:
        rows.append(_benchmark_backend(cfg, backend, timing))

    df = pd.DataFrame(rows)
    df["hydra_output_dir"] = str(run_dir)
    df["hydra_task_overrides"] = ",".join(overrides)
    df["timing_repeats"] = timing["repeats"]
    df["timing_warmup"] = timing["warmup"]
    df["timing_full_solver_warmup_frames"] = timing["full_solver_warmup_frames"]

    output_path = _write_results(df, run_dir, timing["output"])
    _display_table(df)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
