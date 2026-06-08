import importlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omegaconf import OmegaConf

from mpm_jax.p2g.backends import (
    CudaV1Backend,
    CudaV2Backend,
    CudaV3Backend,
    CutileV1Backend,
    CutileV3Backend,
    JaxBackend,
)
from mpm_jax.p2g.cuda.p2g_cuda import (
    CudaV1P2G,
    CudaV2P2G,
    CudaV3P2G,
)
from mpm_jax.types import MPMState, MPMParams


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


cuda_tile = _optional_module("cuda.tile")


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _require_cuda_kernels(*kernel_types):
    if not _has_cuda():
        pytest.skip("CUDA P2G kernels require a GPU backend")
    try:
        for kernel_type in kernel_types:
            kernel_type()
    except ImportError as exc:
        pytest.skip(f"CUDA P2G kernels not built: {exc}")


def _cutile_available() -> bool:
    if not _has_cuda():
        return False
    return cuda_tile is not None


def _require_cutile():
    if not _cutile_available():
        pytest.skip("cuTile/JAX backend requires a GPU and cuda-tile")


def _inputs():
    n = 2000
    num_grids = 16
    rng = np.random.RandomState(123)
    state = MPMState(
        x=jnp.array(rng.rand(n, 3).astype(np.float32) * 0.4 + 0.3),
        v=jnp.array(rng.randn(n, 3).astype(np.float32) * 0.1),
        C=jnp.array(rng.randn(n, 3, 3).astype(np.float32) * 0.01),
        F=jnp.tile(jnp.eye(3, dtype=jnp.float32), (n, 1, 1)),
    )
    stress = jnp.array(rng.randn(n, 3, 3).astype(np.float32) * 0.01)
    params = MPMParams(
        OmegaConf.create(
            {
                "n_particles": n,
                "num_grids": num_grids,
                "dt": 3e-4,
                "gravity": [0.0, 0.0, -9.8],
                "rho": 1000.0,
                "clip_bound": 0.5,
                "damping": 1.0,
                "size": [0.4, 0.4, 0.4],
            }
        )
    )
    return params, state, stress


def _p2g_output(backend, params, state, stress):
    @jax.jit
    def run(state, stress):
        prepared = backend.prepare(params, state, stress)
        return backend.scatter(params, prepared)

    out = run(state, stress)
    jax.block_until_ready(out)
    return out


def test_cuda_p2g_variants_match_jax_scan():
    _require_cuda_kernels(
        CudaV1P2G,
        CudaV2P2G,
        CudaV3P2G,
    )
    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(JaxBackend(), params, state, stress)

    for backend in (
        CudaV1Backend(params.num_grids),
        CudaV2Backend(params.num_grids),
        CudaV3Backend(params.num_grids),
    ):
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)
        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0
        )


def test_cuda_v3_supercell_widths_match_jax_scan():
    _require_cuda_kernels(CudaV3P2G)
    # cuda_v3 is templated + dispatched over SUPPORTED_SC; every compiled
    # super-cell width must scatter identically to the JAX baseline.
    from mpm_jax.p2g.backends import CudaV3Backend
    from mpm_jax.p2g.cuda.p2g_cuda import SUPPORTED_SC

    params, state, stress = _inputs()  # num_grids=16, divisible by 2/4/8
    ref_mv, ref_m = _p2g_output(JaxBackend(), params, state, stress)

    for sc in SUPPORTED_SC:
        backend = CudaV3Backend(num_grids=params.num_grids, super_cell_width=sc)
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)
        np.testing.assert_allclose(
            np.asarray(grid_mv),
            np.asarray(ref_mv),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"SC={sc}",
        )
        np.testing.assert_allclose(
            np.asarray(grid_m),
            np.asarray(ref_m),
            atol=1e-5,
            rtol=1e-5,
            err_msg=f"SC={sc}",
        )


def test_cutile_p2g_matches_jax_scan():
    _require_cutile()
    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(JaxBackend(), params, state, stress)

    for backend in (
        CutileV1Backend(params.num_grids),
        CutileV3Backend(params.num_grids),
    ):
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)

        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0
        )
