"""Wraps the hand-written CUDA P2G kernel as a callable Python function.

Uses PyCUDA to compile and launch the kernel for standalone benchmarking.
Note: The CUDA kernel uses float64 (double) while the JAX solver defaults
to float32. The benchmark converts JAX arrays to float64 for fair comparison.
"""
import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

import os

_KERNEL_FN = None


def _get_kernel():
    global _KERNEL_FN
    if _KERNEL_FN is not None:
        return _KERNEL_FN
    kernel_path = os.path.join(os.path.dirname(__file__), "p2g_kernel.cu")
    with open(kernel_path) as f:
        source = f.read()
    module = SourceModule(source, options=["-arch=sm_90"])
    _KERNEL_FN = module.get_function("p2g_kernel")
    return _KERNEL_FN


def cuda_p2g(x, v, C, stress, N, G, dt, vol, p_mass, inv_dx, dx):
    """Launch the CUDA P2G kernel. All inputs are numpy float64 arrays.

    Returns (grid_mv, grid_m) as numpy arrays.
    """
    if not HAS_PYCUDA:
        raise RuntimeError("pycuda is required for the CUDA benchmark")

    kernel = _get_kernel()
    grid_mv = np.zeros((G ** 3, 3), dtype=np.float64)
    grid_m = np.zeros((G ** 3,), dtype=np.float64)

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel(
        cuda.In(x), cuda.In(v), cuda.In(C), cuda.In(stress),
        cuda.InOut(grid_mv), cuda.InOut(grid_m),
        np.int32(N), np.int32(G),
        np.float64(dt), np.float64(vol), np.float64(p_mass),
        np.float64(inv_dx), np.float64(dx),
        block=(block_size, 1, 1), grid=(grid_size, 1),
    )
    return grid_mv, grid_m
