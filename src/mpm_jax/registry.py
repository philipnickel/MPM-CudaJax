"""Kernel/solver construction entry points.

Construction is declarative: `mpm_jax.zen_build` expresses the MPMSolver as a
hydra-zen `builds()` graph and `build_solver(cfg)` / `instantiate(cfg.solver)`
builds it. This module just re-exports the public names so existing imports
(`from mpm_jax.registry import build_solver`) keep working.
"""

from mpm_jax.backends import KERNEL_NAMES, build_backend
from mpm_jax.zen_build import build_solver

__all__ = ["build_solver", "build_backend", "KERNEL_NAMES"]
