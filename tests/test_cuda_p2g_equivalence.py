import importlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omegaconf import OmegaConf

from mpm_jax.backends import (
    CudaV1Backend,
    CudaV2Backend,
    CudaV3Backend,
    CudaV4Backend,
    CutileBackend,
    JaxBackend,
)
from mpm_jax.cuda.p2g_cuda import (
    register_p2g_inline,
    register_p2g_v2_inline,
    register_p2g_v3_inline,
    register_p2g_v4_inline,
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


def _require_cuda_kernels(*registrars):
    if not _has_cuda():
        pytest.skip("CUDA P2G kernels require a GPU backend")
    if not all(register() for register in registrars):
        pytest.skip("CUDA P2G kernels not built")


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
        return backend.p2g(params, prepared)

    out = run(state, stress)
    jax.block_until_ready(out)
    return out


def test_cuda_p2g_variants_match_jax_scan():
    _require_cuda_kernels(
        register_p2g_inline,
        register_p2g_v2_inline,
        register_p2g_v3_inline,
        register_p2g_v4_inline,
    )
    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(JaxBackend(), params, state, stress)

    for backend in (
        CudaV1Backend(params.num_grids),
        CudaV2Backend(params.num_grids),
        CudaV3Backend(params.num_grids),
        CudaV4Backend(params.num_grids),
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


def test_cuda_v4_supercell_widths_match_jax_scan():
    _require_cuda_kernels(register_p2g_v4_inline)
    # cuda_v4 is templated + dispatched over SUPPORTED_SC; every compiled
    # super-cell width must scatter identically to the JAX baseline.
    from mpm_jax.backends import CudaV4Backend
    from mpm_jax.cuda.p2g_cuda import SUPPORTED_SC

    params, state, stress = _inputs()  # num_grids=16, divisible by 2/4/8
    ref_mv, ref_m = _p2g_output(JaxBackend(), params, state, stress)

    for sc in SUPPORTED_SC:
        backend = CudaV4Backend(num_grids=params.num_grids, super_cell_width=sc)
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

    for backend in (CutileBackend(params.num_grids),):
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
