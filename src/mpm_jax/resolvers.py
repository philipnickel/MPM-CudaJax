"""OmegaConf resolvers and GPU-name helpers shared across the entry points.
Importing this module registers the resolvers globally."""

from __future__ import annotations

import re
import subprocess

from omegaconf import OmegaConf


def _ppc_grid(
    n_particles: int,
    target_ppc: float = 9.31,
    region_volume: float = 0.512,
) -> int:
    """Return G (multiple of 4) so N / (region_volume * G^3) ~= target_ppc."""
    g = (n_particles / target_ppc / region_volume) ** (1 / 3)
    return max(4, 4 * round(g / 4))


def slugify(value) -> str:
    """Lowercase-alphanumeric slug (e.g. of a GPU name), or 'unknown' if empty."""
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def current_gpu_type() -> str:
    """Raw GPU product name from ``nvidia-smi``, or 'unknown' if unavailable."""
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


def current_gpu_kind() -> str:
    """Slugified GPU name, for grouping outputs by device."""
    return slugify(current_gpu_type())


OmegaConf.register_new_resolver("ppc_grid", _ppc_grid, replace=True)
OmegaConf.register_new_resolver("gpu_kind", current_gpu_kind, replace=True)
