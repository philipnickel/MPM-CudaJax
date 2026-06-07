"""Plot benchmark sweeps from the parquet files emitted by the multirun callback.

Default discovers every ``results.parquet`` under ``outputs/sweeps/`` and
concatenates them, so cross-sweep / cross-GPU figures fall out of one call:

    pixi run plot-sweeps                              # walk outputs/sweeps
    pixi run plot-sweeps -- <path-to-results.parquet> # one specific sweep
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.scaling_plots import load_parquet, render  # noqa: E402


def _discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("results.parquet"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("outputs/sweeps")])
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/sweeps"))
    parser.add_argument("--baseline", default="jax")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = [p for root in args.paths for p in _discover(root)]
    if not paths:
        raise SystemExit(f"No results.parquet found under: {args.paths}")
    df = load_parquet(*paths)
    render(df, args.figures_dir, baseline=args.baseline)


if __name__ == "__main__":
    main()
