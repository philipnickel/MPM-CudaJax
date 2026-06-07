"""Hydra callbacks for the MPM-CudaJax benchmark sweeps.

``ScalingPlotCallback`` fires on multirun end. It walks the multirun's
``hydra.sweep.dir``, aggregates the per-job ``results.json`` files into a
single combined dataframe, writes it once as ``results.parquet`` at the
sweep root, and then renders the standard scaling plots from it.

Usage (already wired in ``conf/sweep.yaml``):

    hydra:
      callbacks:
        scaling_plot:
          _target_: postprocessing.callbacks.ScalingPlotCallback
          baseline_kernel: jax
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from postprocessing.scaling_plots import load_jobs, render

logger = logging.getLogger(__name__)


class ScalingPlotCallback(Callback):
    """Aggregate per-job results into one parquet + render scaling plots."""

    def __init__(self, baseline_kernel: str = "jax"):
        self.baseline_kernel = baseline_kernel

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            self._render(config)
        except Exception as exc:
            logger.warning(
                "ScalingPlotCallback failed (this never blocks the sweep): %s",
                exc,
                exc_info=True,
            )

    def _render(self, config: DictConfig) -> None:
        sweep_root = Path(config.hydra.sweep.dir).resolve()
        if not sweep_root.exists():
            logger.warning("Sweep root does not exist: %s", sweep_root)
            return

        df = load_jobs(sweep_root)
        parquet_path = sweep_root / "results.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s with %d rows", parquet_path, len(df))
        # sweep_root is already gpu-specific (outputs/sweeps/<gpu_kind>/runs/...).
        render(df, sweep_root, baseline=self.baseline_kernel, gpu_subdirs=False)
