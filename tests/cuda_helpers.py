"""Shared GPU/CUDA skip-gates for the kernel test suite."""

from __future__ import annotations

import jax
import pytest


def has_cuda() -> bool:
    """True when JAX has a GPU backend available."""
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def require_kernels(*kernel_types, label="CUDA P2G kernels"):
    """Skip the test unless a GPU backend is present and the kernels build."""
    if not has_cuda():
        pytest.skip(f"{label} require a GPU backend")
    try:
        for kernel_type in kernel_types:
            kernel_type()
    except ImportError as exc:
        pytest.skip(f"{label} native extension not built: {exc}")
