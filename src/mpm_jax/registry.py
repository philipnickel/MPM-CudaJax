"""Solver/backend construction entry points.

The solver is built straight from the resolved Hydra config by the config-aware
constructor ``MPMSolver.from_cfg(cfg)``. ``build_solver(cfg)`` is a thin alias,
kept so existing imports (`from mpm_jax.registry import build_solver`) and the
Hydra apps keep working.
"""

from mpm_jax.backends import KERNEL_NAMES, build_backend
from mpm_jax.solver import MPMSolver

__all__ = ["build_solver", "build_backend", "KERNEL_NAMES"]


def build_solver(cfg):
    """Build an MPMSolver from a resolved Hydra config (alias for from_cfg)."""
    return MPMSolver.from_cfg(cfg)
