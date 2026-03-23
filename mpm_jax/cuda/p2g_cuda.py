"""CUDA P2G scatter kernel, integrated via JAX FFI.

Loads the compiled .so and registers it as a JAX FFI target.
Falls back to the JAX implementation if the .so is not built.

Build the kernel first:
    cd mpm_jax/cuda/kernels && make
"""

import os
import ctypes
import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")
_REGISTERED = {}


def _register(name, lib_name, symbol):
    """Load a .so and register an FFI target. Called once per kernel."""
    if name in _REGISTERED:
        return _REGISTERED[name]

    lib_path = os.path.join(_KERNEL_DIR, lib_name)
    if not os.path.exists(lib_path):
        logger.warning(
            "CUDA kernel not built: %s not found. "
            "Build with: cd mpm_jax/cuda/kernels && make",
            lib_path,
        )
        _REGISTERED[name] = False
        return False

    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        jax.ffi.register_ffi_target(
            name,
            jax.ffi.pycapsule(getattr(lib, symbol)),
            platform="CUDA",
        )
        _REGISTERED[name] = True
        logger.info("Registered CUDA kernel '%s' from %s", name, lib_path)
        return True
    except Exception as e:
        logger.warning("Failed to register CUDA kernel '%s': %s", name, e)
        _REGISTERED[name] = False
        return False


def _register_scatter():
    return _register("p2g_scatter_cuda", "libp2g_scatter.so", "P2GScatter")


def _register_fused():
    return _register("p2g_fused_cuda", "libp2g_fused.so", "P2GFused")


def cuda_p2g_scatter(mv, m, index, num_grids):
    """CUDA P2G scatter via JAX FFI.

    Drop-in replacement for solver.p2g_scatter().

    Args:
        mv:    (N, 27, 3) float32 — momentum contributions
        m:     (N, 27)    float32 — mass contributions
        index: (N, 27)    int32   — flat grid indices

    Returns:
        grid_mv: (G^3, 3) float32 — grid momentum
        grid_m:  (G^3,)   float32 — grid mass
    """
    G3 = num_grids ** 3

    # Ensure index is int32 (JAX may default to int64)
    index = index.astype(jnp.int32)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_scatter_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),  # grid_mv
            jax.ShapeDtypeStruct((G3,), jnp.float32),    # grid_m
        ),
        vmap_method="broadcast_all",
    )(mv, m, index)

    return grid_mv, grid_m


def cuda_p2g_fused(x, v, C, F, num_grids, dt, vol, p_mass, inv_dx, dx,
                   mu_0, lambda_0, theta_c=0.025, theta_s=0.0075, hardening=0.0):
    """Fused CUDA P2G via JAX FFI.

    Replaces the entire P2G pipeline (stress + weights + compute + scatter)
    with a single CUDA kernel. Also returns plasticity-corrected F.

    Args:
        x: (N, 3)    positions
        v: (N, 3)    velocities
        C: (N, 3, 3) APIC matrix
        F: (N, 3, 3) deformation gradient

    Returns:
        grid_mv: (G^3, 3) grid momentum
        grid_m:  (G^3,)   grid mass
        F_out:   (N, 3, 3) corrected deformation gradient
    """
    N = x.shape[0]
    G = num_grids
    G3 = G ** 3

    # Flatten C and F to (N, 9) for the kernel
    C_flat = C.reshape(N, 9)
    F_flat = F.reshape(N, 9)

    grid_mv, grid_m, F_out_flat = jax.ffi.ffi_call(
        "p2g_fused_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
            jax.ShapeDtypeStruct((N, 9), jnp.float32),
        ),
        vmap_method="broadcast_all",
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
        mu_0=np.float32(mu_0),
        lambda_0=np.float32(lambda_0),
        theta_c=np.float32(theta_c),
        theta_s=np.float32(theta_s),
        hardening_coeff=np.float32(hardening),
    )(x, v, C_flat, F_flat)

    return grid_mv, grid_m, F_out_flat.reshape(N, 3, 3)


def is_available(kernel='scatter'):
    """Check if a CUDA kernel is built and can be registered."""
    if kernel == 'scatter':
        return _register_scatter()
    elif kernel == 'fused':
        return _register_fused()
    return False


def make_cuda_p2g(num_grids, kernel='scatter'):
    """Create a CUDA-accelerated p2g function matching the solver interface.

    Args:
        num_grids: grid resolution
        kernel: 'scatter' (v1, just the scatter) or 'fused' (v2, full P2G)

    Returns a function compatible with solver.step(p2g_fn=...).
    Returns None if CUDA is not available.
    """
    if kernel == 'scatter':
        if not is_available('scatter'):
            return None

        from mpm_jax.solver import p2g_compute

        def cuda_p2g_v1(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
            mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
            return cuda_p2g_scatter(mv, m, index, num_grids)

        return cuda_p2g_v1

    elif kernel == 'fused':
        if not is_available('fused'):
            return None

        # The fused kernel has a different interface — it takes raw particle
        # state and does stress+scatter internally. The step() function needs
        # to be aware of this. For now, return None and handle in the driver.
        logger.info("Fused CUDA P2G registered — use cuda_p2g_fused() directly")
        return None  # handled specially in the driver

    return None
