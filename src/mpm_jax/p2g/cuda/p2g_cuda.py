"""CUDA P2G kernels, integrated via JAX FFI."""

import importlib
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

_FFI_MODULE = "mpm_jax.p2g.cuda._p2g_ffi"
_REGISTERED: set[str] = set()
_REGISTER_LOCK = Lock()

_P2G_TARGET = "p2g_v1_cuda"
_P2G_V2_TARGET = "p2g_v2_cuda"
_P2G_V3_TARGET = "p2g_v3_cuda"


class CudaP2GKernel:
    """Callable CUDA P2G FFI target that registers itself on construction."""

    target: str
    capsule_factory: str

    def __init__(self):
        self.register()

    def register(self):
        with _REGISTER_LOCK:
            if self.target not in _REGISTERED:
                ffi_module = importlib.import_module(_FFI_MODULE)
                capsule = getattr(ffi_module, self.capsule_factory)()
                jax.ffi.register_ffi_target(
                    self.target,
                    capsule,
                    platform="CUDA",
                    api_version=1,
                )
                _REGISTERED.add(self.target)
        return True

    def __call__(self, x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
        return _p2g_ffi_call(
            self.target,
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


class CudaV1P2G(CudaP2GKernel):
    """One CUDA thread per particle with global atomics."""

    target = _P2G_TARGET
    capsule_factory = "p2g_v1"


class CudaV2P2G(CudaP2GKernel):
    """Warp-shuffle coalesced scatter for Morton-sorted particles."""

    target = _P2G_V2_TARGET
    capsule_factory = "p2g_v2"


class CudaV3P2G(CudaP2GKernel):
    """Super-cell-owned grid tile scatter."""

    target = _P2G_V3_TARGET
    capsule_factory = "p2g_v3"

    def __init__(self, super_cell=4):
        self.super_cell = int(super_cell)
        super().__init__()

    def __call__(
        self,
        x_sorted,
        v_sorted,
        C_sorted,
        stress_sorted,
        bucket_start,
        num_grids,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
    ):
        return _p2g_ffi_call(
            self.target,
            x_sorted,
            v_sorted,
            C_sorted,
            stress_sorted,
            bucket_start,
            num_grids=num_grids,
            dt=dt,
            vol=vol,
            p_mass=p_mass,
            inv_dx=inv_dx,
            dx=dx,
            extra_attrs={"SC": np.int32(self.super_cell)},
        )


# Compiled super-cell-width instantiations.
SUPPORTED_SC = (2, 4, 8)
V3_SUPER_CELL_WIDTH = 4


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


__all__ = [
    "CudaP2GKernel",
    "CudaV1P2G",
    "CudaV2P2G",
    "CudaV3P2G",
    "V3_SUPER_CELL_WIDTH",
    "SUPPORTED_SC",
]
