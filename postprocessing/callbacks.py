"""Hydra callbacks that render plots after multiruns."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from postprocessing.scaling_plots import render

logger = logging.getLogger(__name__)


class ScalingPlotCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            render(Path(config.hydra.sweep.dir), plots=config.plots)
        except Exception as exc:
            logger.warning(
                "ScalingPlotCallback failed (sweep itself is not blocked): %s",
                exc,
                exc_info=True,
            )


class NsightPlotCallback(Callback):
    """Render Nsight figures from the serial multirun aggregate parquet."""

    def __init__(
        self,
        results_name: str = "results.parquet",
        out_dir: str = "figures",
        plots: list[str] | None = None,
        scale_axis: str = "auto",
        style: dict[str, Any] | None = None,
    ):
        self.results_name = results_name
        self.out_dir = out_dir
        self.plots = plots
        self.scale_axis = scale_axis
        self.style = style

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        # Use the sweep aggregate, not per-job.
        sweep_root = Path(config.hydra.sweep.dir).resolve()
        results_path = sweep_root / self.results_name
        if not results_path.exists():
            raise FileNotFoundError(
                f"Nsight aggregate parquet not found: {results_path}"
            )

        out_dir = Path(self.out_dir)
        if not out_dir.is_absolute():
            out_dir = sweep_root / out_dir

        from postprocessing.nsight_plots import render_nsight_figures

        written = render_nsight_figures(
            results_path,
            out_dir,
            plots=self.plots,
            scale_axis=self.scale_axis,
            style=self.style,
        )
        logger.info(
            "Rendered %d Nsight figure(s) from %s into %s",
            len(written),
            results_path,
            out_dir,
        )
