"""CUDA P2G kernels, integrated via JAX FFI.

The native ``mpm_jax.cuda._p2g_ffi`` extension is built by scikit-build-core +
CMake (see CMakeLists.txt). In editable Pixi environments,
``editable.rebuild=true`` lets scikit-build-core incrementally rebuild changed
CUDA or binding sources when the extension module is imported.

Override the CUDA architecture at build time with ``MPM_CUDA_ARCH=sm_86``
(default: ``native``).
"""

import importlib
import logging
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_FFI_MODULE = "mpm_jax.cuda._p2g_ffi"
_REGISTERED: dict[str, bool] = {}
_REGISTER_LOCK = Lock()

_P2G_INLINE_TARGET = "p2g_inline_cuda"
_P2G_V2_INLINE_TARGET = "p2g_v2_inline_cuda"
_P2G_V3_INLINE_TARGET = "p2g_v3_inline_cuda"
_P2G_V4_INLINE_TARGET = "p2g_v4_inline_cuda"


def _register(name: str, capsule_factory: str) -> bool:
    """Import the native binding module and register one FFI target."""
    with _REGISTER_LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        try:
            ffi_module = importlib.import_module(_FFI_MODULE)
            capsule = getattr(ffi_module, capsule_factory)()
        except ImportError:
            logger.warning(
                "CUDA FFI extension %s is unavailable. Run "
                "`pixi install` in an environment where nvcc is on PATH.",
                _FFI_MODULE,
            )
            _REGISTERED[name] = False
            return False
        except Exception as e:
            logger.error("Failed to load CUDA FFI target '%s': %s", name, e)
            _REGISTERED[name] = False
            return False

        try:
            jax.ffi.register_ffi_target(
                name,
                capsule,
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED[name] = True
            logger.info("Registered CUDA FFI target '%s'", name)
            return True
        except Exception as e:
            logger.error("Failed to register CUDA FFI target '%s': %s", name, e)
            _REGISTERED[name] = False
            return False


def register_p2g_inline():
    return _register(_P2G_INLINE_TARGET, "p2g_inline")


def register_p2g_v2_inline():
    return _register(_P2G_V2_INLINE_TARGET, "p2g_v2_inline")


def register_p2g_v3_inline():
    return _register(_P2G_V3_INLINE_TARGET, "p2g_v3_inline")


def register_p2g_v4_inline():
    return _register(_P2G_V4_INLINE_TARGET, "p2g_v4_inline")


def _p2g_ffi_call(
    target_name,
    x,
    v,
    C,
    stress,
    *extra_args,
    num_grids,
    dt,
    vol,
    p_mass,
    inv_dx,
    dx,
    extra_attrs=None,
):
    n_particles = x.shape[0]
    grid_nodes = num_grids**3
    attrs = {
        "G": np.int32(num_grids),
        "dt": np.float32(dt),
        "vol": np.float32(vol),
        "p_mass": np.float32(p_mass),
        "inv_dx": np.float32(inv_dx),
        "dx": np.float32(dx),
        **(extra_attrs or {}),
    }

    return jax.ffi.ffi_call(
        target_name,
        (
            jax.ShapeDtypeStruct((grid_nodes, 3), jnp.float32),
            jax.ShapeDtypeStruct((grid_nodes,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x,
        v,
        C.reshape(n_particles, 9),
        stress.reshape(n_particles, 9),
        *extra_args,
        **attrs,
    )


def _inline_p2g_call(target_name, x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    return _p2g_ffi_call(
        target_name,
        x,
        v,
        C,
        stress,
        num_grids=num_grids,
        dt=dt,
        vol=vol,
        p_mass=p_mass,
        inv_dx=inv_dx,
        dx=dx,
        extra_attrs={"N": np.int32(x.shape[0])},
    )


def cuda_p2g_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G via JAX FFI (backend: cuda_v1).

    Takes per-particle state including precomputed stress (from the JAX-side
    StVK elasticity). One CUDA kernel launch, one thread per particle, with a
    register-resident 27-stencil loop. No (N, 27, *) tensor materialised.

    Stress is computed by the JAX elasticity model; weights and scatter happen
    inside this CUDA kernel.
    """
    return _inline_p2g_call(
        _P2G_INLINE_TARGET,
        x,
        v,
        C,
        stress,
        num_grids,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
    )


def cuda_p2g_v2_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle reduction (backend: cuda_v2).

    Same FFI signature as ``cuda_p2g_inline`` — only the C++ symbol is
    different. The kernel inserts a ``__match_any_sync`` + ``__shfl_sync``
    reduction over each arbitrary peer mask in front of every atomicAdd inside
    the 27-stencil scatter loop, so warp-resident contributions to the same
    grid_idx collapse to a single atomic.
    """
    return _inline_p2g_call(
        _P2G_V2_INLINE_TARGET,
        x,
        v,
        C,
        stress,
        num_grids,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
    )


def cuda_p2g_v3_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle atomic coalescing (backend: cuda_v3).

    Identical kernel-side reduction as ``cuda_p2g_v2_inline``. Designed to
    be called on Morton-sorted particles (see
    :func:`mpm_jax.sort.morton_argsort`) so adjacent warp lanes share
    stencil targets — the sort is what makes the warp reduction productive.
    """
    return _inline_p2g_call(
        _P2G_V3_INLINE_TARGET,
        x,
        v,
        C,
        stress,
        num_grids,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
    )


# Super-cell width for the cuda_v4 backend. With SC=k the kernel launches (G/SC)^3
# blocks (vs G^3) and each block aggregates particles from SC^3 cells into a
# (SC+2)^3 shared-memory tile. The kernel is a template on SC; the FFI handler
# dispatches to the instantiated values in SUPPORTED_SC by a runtime switch, so
# SC is config-selectable (backend.super_cell_width) without recompiling — but
# only among the instantiated widths. SC=4 is the default: 4^3 cells, ~512
# particles/block at the 8-particles/cell benchmark, and a 6^3 grid scratchpad.
SUPPORTED_SC = (2, 4, 8)  # template instantiations compiled into the extension
V4_SUPER_CELL_WIDTH = 4  # default super-cell width


def cuda_p2g_v4_inline(
    x_sorted,
    v_sorted,
    C_sorted,
    stress_sorted,
    cell_start,
    num_grids,
    dt,
    vol,
    p_mass,
    inv_dx,
    dx,
    super_cell=V4_SUPER_CELL_WIDTH,
):
    """Cell-major inline P2G via JAX FFI (backend: cuda_v4).

    The Python wrapper assumes the inputs are already sorted by home
    *super*-cell and that ``cell_start`` is the CSR boundary array of length
    (G/SC)^3 + 1, where SC is ``super_cell`` (one of :data:`SUPPORTED_SC`).

    The kernel uses one CUDA block per super-cell and aggregates each
    super-cell's contributions into a (SC+2)^3 shared-memory tile before
    flushing to HBM. With SC=4, the standard 8 particles/cell benchmark gives
    each non-empty block enough particles to amortize the tile overhead.
    """
    return _p2g_ffi_call(
        _P2G_V4_INLINE_TARGET,
        x_sorted,
        v_sorted,
        C_sorted,
        stress_sorted,
        cell_start,
        num_grids=num_grids,
        dt=dt,
        vol=vol,
        p_mass=p_mass,
        inv_dx=inv_dx,
        dx=dx,
        extra_attrs={"SC": np.int32(super_cell)},
    )


__all__ = [
    # FFI registration
    "register_p2g_inline",
    "register_p2g_v2_inline",
    "register_p2g_v3_inline",
    "register_p2g_v4_inline",
    # FFI op wrappers
    "cuda_p2g_inline",
    "cuda_p2g_v2_inline",
    "cuda_p2g_v3_inline",
    "cuda_p2g_v4_inline",
    # Super-cell helpers
    "V4_SUPER_CELL_WIDTH",
    "SUPPORTED_SC",
]
