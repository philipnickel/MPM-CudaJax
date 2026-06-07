"""Re-render scaling plots for an existing sweep multirun.

Same call the on-multirun-end callback runs; useful after a plotting tweak.
The ``plots`` block is loaded from any one of the sweep's per-job
``.hydra/config.yaml`` files (they all share the same plots config).

    pixi run plot-sweeps -- outputs/sweeps/<gpu>/runs/<date>/<time>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf  # noqa: E402

import mpm_jax.resolvers  # noqa: F401, E402 - registers ${ppc_grid:N}
import postprocessing  # noqa: F401, E402 - registers Hydra plot configs
from postprocessing.scaling_plots import render  # noqa: E402


def _load_plots(sweep_root: Path):
    job_cfgs = sorted(sweep_root.glob("*/.hydra/config.yaml"))
    if not job_cfgs:
        raise FileNotFoundError(f"No job .hydra/config.yaml under {sweep_root}")
    cfg = OmegaConf.load(job_cfgs[0])
    return cfg.plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path,
                        help="A multirun directory (the hydra.sweep.dir).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    render(args.sweep_root, plots=_load_plots(args.sweep_root))


if __name__ == "__main__":
    main()
