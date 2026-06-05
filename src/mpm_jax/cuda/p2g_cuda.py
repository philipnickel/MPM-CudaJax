"""CUDA P2G kernels, integrated via JAX FFI.

The .so files are built at install time by scikit-build-core + CMake (see
CMakeLists.txt) and shipped inside ``mpm_jax/cuda/_lib/``. In editable Pixi
environments, ``editable.rebuild=true`` lets scikit-build-core incrementally
rebuild changed CUDA sources when the packaged artifact is resolved.

Override the CUDA architecture at build time with ``MPM_CUDA_ARCH=sm_86``
(default: ``native``).
"""

import ctypes
import importlib.resources as resources
import importlib.util
import logging
from pathlib import Path
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LIB_DIR = _PACKAGE_DIR / "_lib"
_REGISTERED: dict[str, bool] = {}
_REGISTER_LOCK = Lock()


def _shared_library_path(so_name: str) -> Path:
    return _LIB_DIR / so_name


def _shared_library_candidates(so_name: str) -> list[Path]:
    """Return plausible locations for a packaged CUDA shared library.

    Prefer Python's package/artifact lookup so scikit-build-core's editable
    rebuild hook can run when CUDA sources changed. The source-tree ``_lib``
    path is only a fallback for older installs or direct in-tree builds.
    """
    # Ask Python's import machinery for the installed artifact. In editable
    # scikit-build-core installs, this triggers the native editable rebuild
    # hook for known wheel files when needed and gives us the built .so path.
    candidates = []
    module_name = Path(so_name).stem
    try:
        spec = importlib.util.find_spec(f"mpm_jax.cuda._lib.{module_name}")
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin))

    try:
        resource_path = resources.files("mpm_jax.cuda._lib").joinpath(so_name)
        candidates.append(Path(str(resource_path)))
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    candidates.append(_shared_library_path(so_name))

    return candidates


def _find_shared_library(so_name: str) -> Path | None:
    for candidate in _shared_library_candidates(so_name):
        if candidate.exists():
            return candidate
    return None


def _register(name: str, so_name: str, symbol: str) -> bool:
    """Load .so from the package's _lib/ dir and register the FFI target."""
    with _REGISTER_LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        so_path = _find_shared_library(so_name)
        if so_path is None:
            logger.warning(
                "CUDA kernel %s not found in package resources. Run "
                "`pixi install -e gpu` in an environment where nvcc is on PATH.",
                so_name,
            )
            _REGISTERED[name] = False
            return False

        try:
            lib = ctypes.cdll.LoadLibrary(str(so_path))
            jax.ffi.register_ffi_target(
                name,
                jax.ffi.pycapsule(getattr(lib, symbol)),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED[name] = True
            logger.info("Registered CUDA kernel '%s' from %s", name, so_path)
            return True
        except Exception as e:
            logger.error("Failed to register CUDA kernel '%s': %s", name, e)
            _REGISTERED[name] = False
            return False


def _register_inline():
    return _register("p2g_inline_cuda", "libp2g_inline.so", "P2GInline")


def _register_v2_inline():
    return _register("p2g_v2_inline_cuda", "libp2g_v2_inline.so", "P2GV2Inline")


def _register_v3_inline():
    return _register("p2g_v3_inline_cuda", "libp2g_v3_inline.so", "P2GV3Inline")


def _register_v4_inline():
    return _register("p2g_v4_inline_cuda", "libp2g_v4_inline.so", "P2GV4Inline")


def is_available(kernel="inline"):
    """Check if a prebuilt CUDA kernel can be loaded and registered."""
    if kernel == "inline":
        return _register_inline()
    elif kernel == "v2_inline":
        return _register_v2_inline()
    elif kernel == "v3_inline":
        return _register_v3_inline()
    elif kernel == "v4_inline":
        return _register_v4_inline()
    return False


def cuda_p2g_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G via JAX FFI (cuda_v1_inline).

    Takes per-particle state including precomputed stress (from the JAX-side
    StVK elasticity). One CUDA kernel launch, one thread per particle, with a
    register-resident 27-stencil loop. No (N, 27, *) tensor materialised.

    Stress is computed by the JAX elasticity model; weights and scatter happen
    inside this CUDA kernel.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G**3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x,
        v,
        C_flat,
        stress_flat,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


def cuda_p2g_v2_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle reduction (cuda_v2_inline).

    Same FFI signature as ``cuda_p2g_inline`` — only the C++ symbol is
    different. The kernel inserts a ``__match_any_sync`` + ``__shfl_sync``
    reduction over each arbitrary peer mask in front of every atomicAdd inside
    the 27-stencil scatter loop, so warp-resident contributions to the same
    grid_idx collapse to a single atomic.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G**3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v2_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x,
        v,
        C_flat,
        stress_flat,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


def cuda_p2g_v3_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle atomic coalescing (cuda_v3_inline).

    Identical kernel-side reduction as ``cuda_p2g_v2_inline``. Designed to
    be called on Morton-sorted particles (see
    :func:`mpm_jax.sort.morton_argsort`) so adjacent warp lanes share
    stencil targets — the sort is what makes the warp reduction productive.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G**3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v3_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",  # should we do it like this?
    )(
        x,
        v,
        C_flat,
        stress_flat,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


# Super-cell width for cuda_v4_inline. With SC=k the kernel launches (G/SC)^3
# blocks (vs G^3) and each block aggregates particles from SC^3 cells into a
# (SC+2)^3 shared-memory tile. The kernel is a template on SC; the FFI handler
# dispatches to the instantiated values in SUPPORTED_SC by a runtime switch, so
# SC is config-selectable (backend.super_cell_width) without recompiling — but
# only among the instantiated widths. SC=4 is the default: 4^3 cells, ~512
# particles/block at the 8-particles/cell benchmark, and a 6^3 grid scratchpad.
SUPPORTED_SC = (2, 4, 8)  # template instantiations compiled into the .so
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
    """Cell-major inline P2G via JAX FFI (cuda_v4_inline).

    The Python wrapper assumes the inputs are already sorted by home
    *super*-cell and that ``cell_start`` is the CSR boundary array of length
    (G/SC)^3 + 1, where SC is ``super_cell`` (one of :data:`SUPPORTED_SC`).

    The kernel uses one CUDA block per super-cell and aggregates each
    super-cell's contributions into a (SC+2)^3 shared-memory tile before
    flushing to HBM. With SC=4, the standard 8 particles/cell benchmark gives
    each non-empty block enough particles to amortize the tile overhead.
    """
    N = x_sorted.shape[0]
    G = num_grids
    G3 = G**3
    C_flat = C_sorted.reshape(N, 9)
    stress_flat = stress_sorted.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v4_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x_sorted,
        v_sorted,
        C_flat,
        stress_flat,
        cell_start,
        G=np.int32(G),
        SC=np.int32(super_cell),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


__all__ = [
    # FFI registration
    "is_available",
    # FFI op wrappers
    "cuda_p2g_inline",
    "cuda_p2g_v2_inline",
    "cuda_p2g_v3_inline",
    "cuda_p2g_v4_inline",
    # Super-cell helpers
    "V4_SUPER_CELL_WIDTH",
    "SUPPORTED_SC",
]
