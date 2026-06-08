"""Hydra-wrapped Nsight Compute collection for custom P2G scatter kernels.

Each Hydra job profiles one warmed backend scatter call with prepared inputs.
Multirun sweeps append processed nsight-python rows to the sweep
``results.parquet``; metric derivation stays in postprocessing.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

import hydra
import jax
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import mpm_jax.p2g.backends as backend_configs
import mpm_jax.resolvers  # noqa: F401 - registers ${ppc_grid:N}
from mpm_jax.solver import MPMSolver

logger = logging.getLogger(__name__)


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - optional profiling package
        return None


nsight = _optional_module("nsight")

# Guard top-level nsight.* config typos.
_SCRIPT_NSIGHT_KEYS = {"analyze"}
# Nsight Python targets single custom scatter annotations; use XProf for JAX/XLA.
_NSIGHT_BACKEND_CHOICES = tuple(
    choice for choice in backend_configs.backend_choices() if choice != "jax"
)
# configs is derived from Hydra; metric derivation stays in postprocessing.
_UNSUPPORTED_ANALYZE_CONFIG_KEYS = {
    "configs",
    "derive_metric",
    "metric_preset",
    "metric_presets",
}


def _slugify(value):
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def _current_gpu_type():
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
        return result.stdout.splitlines()[0].strip() or "unknown"
    except Exception:
        return "unknown"


def _current_gpu_kind():
    return _slugify(_current_gpu_type())


OmegaConf.register_new_resolver("gpu_kind", _current_gpu_kind, replace=True)


def _normalize_backend_choice(value) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Backend choices must be Hydra config-group names.")
    choice = value.strip()
    supported = backend_configs.backend_choices()
    if choice not in supported:
        raise RuntimeError(
            f"Unsupported backend choice {value!r}. "
            "Use Hydra backend choices: " + ", ".join(supported)
        )
    return choice


def _backend_choice_from_backend_cfg(backend_cfg) -> str:
    target = backend_cfg.get("_target_", None)
    for choice in backend_configs.backend_choices():
        if target == backend_configs.backend_config(choice).get("_target_", None):
            return choice
    raise RuntimeError(
        "Could not infer backend choice from cfg.backend. "
        "Use one of: " + ", ".join(backend_configs.backend_choices())
    )


def _backend_choice_from_cfg(cfg: DictConfig):
    try:
        return _normalize_backend_choice(HydraConfig.get().runtime.choices["backend"])
    except (ValueError, KeyError):
        pass
    return _backend_choice_from_backend_cfg(cfg.get("backend", {}))


def _profile_config(cfg: DictConfig, backend_choice: str):
    """Config tuple passed to nsight.analyze."""
    if backend_choice not in _NSIGHT_BACKEND_CHOICES:
        supported = ", ".join(_NSIGHT_BACKEND_CHOICES)
        raise RuntimeError(
            "Nsight Python analysis profiles custom CUDA/cuTile P2G scatter "
            f"kernels only; backend={backend_choice!r} is not supported. "
            "Use JAX/XProf tracing for backend='jax', or choose one of: "
            f"{supported}."
        )
    return (
        backend_choice,
        int(cfg.sim.n_particles),
        int(cfg.sim.num_grids),
        int(cfg.sim.steps_per_frame),
    )


def _require_nsight():
    if nsight is None:
        raise RuntimeError(
            "nsight-python is not installed. Run `pixi install` in the GPU environment."
        )
    return nsight


def _block_until_ready(value):
    """Synchronize all JAX arrays in a pytree and return the original value."""
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


class _P2GScatterRunner:
    """Prepared, warmed scatter callable for the selected backend."""

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._run = None

    def ensure_ready(self):
        if self._run is None:
            solver = MPMSolver(hydra.utils.instantiate(self._cfg.solver))
            params = solver.params
            backend = solver.backend
            state = solver.state
            elasticity_fn = solver.elasticity_fn

            @jax.jit
            def prepare(state):
                stress = elasticity_fn(state.F)
                return backend.prepare(params, state, stress)

            @jax.jit
            def p2g_scatter(prepared):
                return backend.scatter(params, prepared)

            prepared = _block_until_ready(prepare(state))
            _block_until_ready(p2g_scatter(prepared))

            def run():
                return _block_until_ready(p2g_scatter(prepared))

            self._run = run
        return self._run

    def __call__(self):
        return self.ensure_ready()()


def _p2g_runner(cfg: DictConfig):
    return _P2GScatterRunner(cfg)


def _combine_kernel_metrics(name):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError(
            "nsight.analyze.combine_kernel_metrics must be null or a preset name."
        )
    if name in {"sum", "add"}:
        return lambda x, y: x + y
    if name == "max":
        return max
    if name == "min":
        return min
    raise RuntimeError(
        f"Unsupported combine_kernel_metrics={name!r}; supported presets: sum, max, min"
    )


def _dedupe_metrics(metrics):
    deduped = []
    for metric in metrics:
        metric = str(metric)
        if metric not in deduped:
            deduped.append(metric)
    return deduped


def _nsight_analyze_kwargs(cfg: DictConfig, run_dir: Path, profile_config):
    analyze_cfg = cfg.nsight.get("analyze", {})
    kwargs = OmegaConf.to_container(analyze_cfg, resolve=True)
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise RuntimeError(
            "nsight.analyze must be a mapping of nsight.analyze.kernel options."
        )

    unsupported = _UNSUPPORTED_ANALYZE_CONFIG_KEYS.intersection(kwargs)
    if unsupported:
        keys = ", ".join(sorted(unsupported))
        raise RuntimeError(
            "The Hydra nsight.analyze block only supports direct nsight.analyze.kernel "
            f"collection options; unsupported keys: {keys}."
        )

    kwargs = dict(kwargs)
    kwargs.setdefault("runs", 1)
    kwargs.setdefault("replay_mode", "kernel")
    kwargs["metrics"] = _dedupe_metrics(kwargs.get("metrics") or [])
    if not kwargs["metrics"]:
        kwargs["metrics"] = ["gpu__time_duration.sum"]
    kwargs["combine_kernel_metrics"] = _combine_kernel_metrics(
        kwargs.get("combine_kernel_metrics")
    )
    kwargs.setdefault("output", "progress")
    kwargs.setdefault("output_csv", False)
    backend_choice = profile_config[0]
    kwargs.setdefault(
        "output_prefix", str(run_dir / f"nsight_{backend_choice}_p2g_")
    )
    # One profile config per Hydra job; multirun handles sweeps.
    kwargs["configs"] = [profile_config]
    return kwargs


def _config_metadata(cfg: DictConfig, run_dir: Path, backend_choice: str, target_name: str):
    gpu_type = _current_gpu_type()
    sim = OmegaConf.to_container(cfg.sim, resolve=True)
    sim_columns = pd.json_normalize({"sim": sim}, sep=".").iloc[0].to_dict()
    metadata = {
        "backend": backend_choice,
        "target": target_name,
        "gpu_type": gpu_type,
        "gpu_kind": _slugify(gpu_type),
        "mps_thread_percent": cfg.get("mps_thread_percent"),
        "hydra_output_dir": str(run_dir),
        "hydra_config": json.dumps(
            OmegaConf.to_container(cfg, resolve=True), sort_keys=True
        ),
    }
    metadata.update(sim_columns)
    return {
        key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in metadata.items()
    }


def _results_dataframe(results, cfg, run_dir, backend_choice, target_name):
    df = results.to_dataframe().copy()
    for key, value in _config_metadata(cfg, run_dir, backend_choice, target_name).items():
        df[key] = value
    return df


def _write_dataframe(df, path: Path, *, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def _aggregate_dir():
    hydra_cfg = HydraConfig.get()
    if str(OmegaConf.select(hydra_cfg, "mode")) != "RunMode.MULTIRUN":
        return None
    sweep_dir = Path(str(OmegaConf.select(hydra_cfg, "sweep.dir")))
    if not sweep_dir.is_absolute():
        sweep_dir = Path(hydra.utils.get_original_cwd()) / sweep_dir
    return sweep_dir.resolve()


def _write_results(df, run_dir):
    _write_dataframe(df, run_dir / "nsight_metrics.parquet")

    aggregate = _aggregate_dir()
    if aggregate is not None:
        _write_dataframe(df, aggregate / "results.parquet", append=True)
        logger.info("Appended %d Nsight rows to %s/results.parquet", len(df), aggregate)


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


@hydra.main(version_base=None, config_path="conf", config_name="nsight_profile")
def main(cfg: DictConfig):
    nsight = _require_nsight()
    unexpected = set(cfg.nsight.keys()) - _SCRIPT_NSIGHT_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unknown nsight config keys: {keys}.")

    backend_choice = _backend_choice_from_cfg(cfg)
    target_name = "p2g"
    profile_config = _profile_config(cfg, backend_choice)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    analyze_kwargs = _nsight_analyze_kwargs(cfg, run_dir, profile_config)
    launcher = _p2g_runner(cfg)

    def profiled_variant(variant_backend, n_particles, num_grids, steps_per_frame):
        launcher.ensure_ready()
        with nsight.annotate(f"{variant_backend}_p2g"):
            launcher()

    profiled_variant = nsight.analyze.kernel(**analyze_kwargs)(profiled_variant)

    logger.info(
        "Nsight profile: backend=%s target=%s config=%s",
        backend_choice,
        target_name,
        profile_config,
    )
    results = _run_nsight_profile(profiled_variant)
    # NCU child re-execs can return None for non-matching Hydra jobs.
    if results is not None:
        df = _results_dataframe(results, cfg, run_dir, backend_choice, target_name)
        _write_results(df, run_dir)


if __name__ == "__main__":
    main()
