"""Hydra callback that renders scaling plots when a multirun finishes.

Wired in ``conf/sweep.yaml`` as ``hydra.callbacks.scaling_plot``. Passes the
multirun's ``hydra.sweep.dir`` plus the composed ``cfg.plots`` block to
:func:`postprocessing.scaling_plots.render`.
"""

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
