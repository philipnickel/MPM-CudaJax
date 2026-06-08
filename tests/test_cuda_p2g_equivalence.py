import importlib

import numpy as np
import pytest

from mpm_jax.p2g.backends import (
    CudaV1Backend,
    CudaV3Backend,
    CudaV4Backend,
    CutileV3Backend,
    JaxBackend,
)
from mpm_jax.p2g.cuda.p2g_cuda import (
    CudaV1P2G,
    CudaV3P2G,
    CudaV4P2G,
)
from tests.cuda_validation import (
    assert_grid_close,
    assert_home_cell_prepared,
    benchmark_validation_size,
    has_cuda,
    make_p2g_inputs,
    p2g_output,
    require_cuda_kernels,
    timed_prepare_scatter,
)


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


cuda_tile = _optional_module("cuda.tile")


def _cutile_available() -> bool:
    if not has_cuda():
        return False
    return cuda_tile is not None


def _require_cutile():
    if not _cutile_available():
        pytest.skip("cuTile/JAX backend requires a GPU and cuda-tile")


def test_cuda_p2g_variants_match_jax_scan():
    require_cuda_kernels(
        CudaV1P2G,
        CudaV3P2G,
        CudaV4P2G,
    )
    params, state, stress = make_p2g_inputs()
    ref = p2g_output(JaxBackend(), params, state, stress)

    for backend in (
        CudaV1Backend(params.num_grids),
        CudaV3Backend(params.num_grids),
        CudaV4Backend(params.num_grids),
    ):
        grid = p2g_output(backend, params, state, stress)
        assert_grid_close(grid, ref, err_msg=backend.name)


def test_cuda_progression_prepare_contract(monkeypatch):
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV1P2G.register", lambda self: True
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV3P2G.register", lambda self: True
    )
    monkeypatch.setattr(
        "mpm_jax.p2g.cuda.p2g_cuda.CudaV4P2G.register", lambda self: True
    )

    params, state, stress = make_p2g_inputs(n=64, num_grids=16)
    v1_prepared = CudaV1Backend(params.num_grids).prepare(params, state, stress)

    np.testing.assert_array_equal(np.asarray(v1_prepared.x), np.asarray(state.x))
    np.testing.assert_array_equal(np.asarray(v1_prepared.v), np.asarray(state.v))
    assert v1_prepared.bucket_start is None
    assert v1_prepared.bucket_bounds is None

    for backend in (
        CudaV3Backend(params.num_grids),
        CudaV4Backend(params.num_grids),
    ):
        assert_home_cell_prepared(backend, params, state, stress)


def test_cuda_progression_optional_benchmark_scale_matches_v1():
    n, num_grids = benchmark_validation_size()
    require_cuda_kernels(CudaV1P2G, CudaV3P2G, CudaV4P2G)

    params, state, stress = make_p2g_inputs(
        n=n, num_grids=num_grids, seed=987, size=(0.8, 0.8, 0.8)
    )
    ref, ref_timing = timed_prepare_scatter(
        CudaV1Backend(params.num_grids), params, state, stress
    )
    timings = [("cuda_v1", ref_timing)]

    for backend in (
        CudaV3Backend(params.num_grids),
        CudaV4Backend(params.num_grids),
    ):
        grid, timing = timed_prepare_scatter(backend, params, state, stress)
        assert_grid_close(
            grid,
            ref,
            atol=5e-5,
            rtol=5e-5,
            sum_atol=1e-3,
            err_msg=backend.name,
        )
        timings.append((backend.name, timing))

    for name, timing in timings:
        assert timing["prepare_ms"] >= 0.0, name
        assert timing["scatter_ms"] >= 0.0, name
        print(
            f"{name}: prepare={timing['prepare_ms']:.3f} ms "
            f"scatter={timing['scatter_ms']:.3f} ms"
        )


def test_cutile_p2g_matches_jax_scan():
    _require_cutile()
    params, state, stress = make_p2g_inputs()
    ref = p2g_output(JaxBackend(), params, state, stress)

    for backend in (
        CutileV3Backend(params.num_grids),
    ):
        grid = p2g_output(backend, params, state, stress)
        assert_grid_close(grid, ref, err_msg=backend.name)
