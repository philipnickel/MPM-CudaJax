"""Collect hierarchical roofline data for core CUDA/Warp P2G kernels only.

This wrapper profiles the same CUDA/Warp backend P2G call as
``hiearchical_roofline.py`` but filters out setup launches so the retained
Nsight data corresponds to the actual scatter kernels:

* ``p2g_inline_kernel``
* ``p2g_v2_inline_kernel``
* ``p2g_v3_inline_kernel``
* ``p2g_v4_inline_kernel``
* ``_p2g_supercell_tile_kernel...``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOFLINE_SCRIPT = ROOT / "benchmarking" / "hiearchical_roofline.py"
DEFAULT_KERNELS = ["all_cuda", "warp_v3_supercell_tile"]
SETUP_KERNELS_TO_IGNORE = [
    "zero_kernel",
    "zero_kernel(float *, int)",
    "zero_kernel(float*, int)",
    "loop_broadcast_fusion",
    "loop_broadcast_fusion_1",
    "loop_broadcast_fusion_2",
]


def _count_label(n_particles: int):
    if n_particles % 1_000_000 == 0:
        return f"{n_particles // 1_000_000}m"
    if n_particles % 1_000 == 0:
        return f"{n_particles // 1_000}k"
    return str(n_particles)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=list(DEFAULT_KERNELS),
        help=(
            "CUDA/Warp backends to profile. Defaults to all_cuda "
            "warp_v3_supercell_tile."
        ),
    )
    parser.add_argument("--n-particles", type=int, default=1_000_000)
    parser.add_argument("--num-grids", type=int, default=124)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--title", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-csv", action="store_true")
    parser.add_argument(
        "--ignore-kernel",
        action="append",
        default=[],
        help="Additional exact Nsight Compute kernel name to filter out.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the delegated hiearchical_roofline.py command without running it.",
    )
    return parser.parse_args()


def _build_command(args):
    output = args.output
    if output is None:
        output = (
            ROOT
            / "outputs"
            / "roofline"
            / f"core_hierarchical_{_count_label(args.n_particles)}_cuda_warp_p2g_kernel.png"
        )

    command = [
        sys.executable,
        str(ROOFLINE_SCRIPT),
        "--kernels",
        *args.kernels,
        "--roofline",
        "hierarchical",
        "--n-particles",
        str(args.n_particles),
        "--num-grids",
        str(args.num_grids),
        "--stage",
        "p2g_kernel",
        "--replay-mode",
        "kernel",
        "--runs",
        str(args.runs),
        "--output",
        str(output),
    ]
    if args.title:
        command.extend(["--title", args.title])
    if args.output_csv:
        command.append("--output-csv")

    return command


def main():
    args = _parse_args()
    command = _build_command(args)
    print("Running:")
    print(" ".join(command))
    if args.dry_run:
        return
    env = os.environ.copy()
    env["ROOFLINE_IGNORE_KERNELS_JSON"] = json.dumps(
        [*SETUP_KERNELS_TO_IGNORE, *args.ignore_kernel]
    )
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
