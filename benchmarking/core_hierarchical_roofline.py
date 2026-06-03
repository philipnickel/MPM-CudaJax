"""Collect hierarchical roofline data for CUDA/Warp P2G kernels.

This wrapper delegates to ``hiearchical_roofline.py`` with the ``p2g_kernel``
stage, so Nsight Python profiles each backend P2G call after prepare/stress
setup while including the required grid zeroing work:

* ``p2g_inline_kernel``
* ``p2g_v2_inline_kernel``
* ``p2g_v3_inline_kernel``
* ``p2g_v4_inline_kernel``
* ``_p2g_supercell_tile_kernel...``
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOFLINE_SCRIPT = ROOT / "benchmarking" / "hiearchical_roofline.py"
DEFAULT_KERNELS = ["all_cuda", "warp_v3_supercell_tile"]


def _count_label(n_particles: int):
    if n_particles % 1_000_000 == 0:
        return f"{n_particles // 1_000_000}m"
    if n_particles % 1_000 == 0:
        return f"{n_particles // 1_000}k"
    return str(n_particles)


def _parse_particle_counts(args):
    if args.particle_counts is None:
        return [int(args.n_particles)]
    counts = []
    for item in args.particle_counts:
        for value in str(item).split(","):
            value = value.strip()
            if value:
                counts.append(int(value))
    counts = list(dict.fromkeys(counts))
    if not counts:
        raise RuntimeError("--particle-counts did not contain any valid counts.")
    return counts


def _count_range_label(counts: list[int]):
    if len(counts) == 1:
        return _count_label(counts[0])
    return f"{_count_label(min(counts))}_to_{_count_label(max(counts))}"


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
    parser.add_argument(
        "--particle-counts",
        nargs="+",
        default=None,
        help=(
            "Profile a particle-count trajectory. Accepts space-separated or "
            "comma-separated counts. Defaults to --n-particles."
        ),
    )
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
    particle_counts = _parse_particle_counts(args)
    output = args.output
    if output is None:
        output = (
            ROOT
            / "outputs"
            / "roofline"
            / f"hierarchical_{_count_range_label(particle_counts)}_cuda_warp_p2g_kernel.png"
        )

    command = [
        sys.executable,
        str(ROOFLINE_SCRIPT),
        "--kernels",
        *args.kernels,
        "--roofline",
        "hierarchical",
        "--n-particles",
        str(particle_counts[0]),
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
    if len(particle_counts) > 1:
        command.extend(["--particle-counts", *[str(count) for count in particle_counts]])
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
    if args.ignore_kernel:
        import json  # pylint: disable=import-outside-toplevel

        env["ROOFLINE_IGNORE_KERNELS_JSON"] = json.dumps(args.ignore_kernel)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
