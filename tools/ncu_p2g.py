"""Small NVTX-marked target for raw Nsight Compute P2G profiling.

Example:

    pixi run ncu --nvtx --nvtx-push-pop-scope process \
      --nvtx-include "mpm_cudajax@cutile_v3_scatter/" \
      --metrics gpu__time_duration.sum --force-overwrite \
      -o outputs/ncu/cutile_v3_scatter \
      python tools/ncu_p2g.py --target scatter backend=cutile_v3 sim=benchmark
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
from pathlib import Path
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(sysconfig.get_path("purelib"))):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)

import hydra
import nvtx
from hydra import compose, initialize_config_dir

import mpm_jax.p2g.backends  # noqa: F401 - registers Hydra backend config choices
from mpm_jax.profiling import NVTX_DOMAIN, block_until_ready, build_profile_target
from mpm_jax.profiling.p2g import PROFILE_TARGETS
from mpm_jax.solver import MPMSolver


logger = logging.getLogger(__name__)
CONF_DIR = ROOT / "conf"


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=PROFILE_TARGETS, default="scatter")
    parser.add_argument("--domain", default=NVTX_DOMAIN)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=list(args.overrides))
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    target = build_profile_target(solver, args.target)
    logger.info(
        "Running %s target for backend=%s with NVTX %s@%s",
        target.name,
        target.backend_name,
        args.domain,
        target.annotation,
    )
    for _ in range(args.repeats):
        with nvtx.annotate(target.annotation, domain=args.domain):
            block_until_ready(target.run())


if __name__ == "__main__":
    main()
