"""Shared CUDA backend validation helpers for tests."""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.p2g.backends.common import morton_order, supercell_order
from mpm_jax.p2g.cuda.p2g_cuda import V3_SUPER_CELL_WIDTH
from mpm_jax.types import MPMParams, MPMState


BENCHMARK_VALIDATION_ENV = "MPM_CUDA_BENCHMARK_VALIDATION"


def has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def require_cuda_kernels(*kernel_types):
    if not has_cuda():
        pytest.skip("CUDA P2G kernels require a GPU backend")
    try:
        for kernel_type in kernel_types:
            kernel_type()
    except ImportError as exc:
        pytest.skip(f"CUDA P2G kernels not built: {exc}")


def make_p2g_inputs(n=2000, num_grids=16, seed=123, size=(0.4, 0.4, 0.4)):
    rng = np.random.RandomState(seed)
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
                "size": list(size),
            }
        )
    )
    return params, state, stress


def benchmark_validation_size():
    if os.environ.get(BENCHMARK_VALIDATION_ENV) != "1":
        pytest.skip(
            f"set {BENCHMARK_VALIDATION_ENV}=1 to run benchmark-scale CUDA "
            "correctness validation"
        )
    n = int(os.environ.get("MPM_CUDA_BENCHMARK_PARTICLES", "10000000"))
    num_grids = int(os.environ.get("MPM_CUDA_BENCHMARK_GRIDS", "96"))
    return n, num_grids


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def p2g_output(backend, params, state, stress):
    @jax.jit
    def run(state, stress):
        prepared = backend.prepare(params, state, stress)
        return backend.scatter(params, prepared)

    out = run(state, stress)
    return block_until_ready(out)


def timed_prepare_scatter(backend, params, state, stress):
    @jax.jit
    def prepare(state, stress):
        return backend.prepare(params, state, stress)

    @jax.jit
    def scatter(prepared):
        return backend.scatter(params, prepared)

    prepared = block_until_ready(prepare(state, stress))
    out = block_until_ready(scatter(prepared))

    t0 = time.perf_counter()
    prepared = block_until_ready(prepare(state, stress))
    t1 = time.perf_counter()
    out = block_until_ready(scatter(prepared))
    t2 = time.perf_counter()
    return out, {"prepare_ms": (t1 - t0) * 1000, "scatter_ms": (t2 - t1) * 1000}


def assert_grid_close(
    actual,
    expected,
    *,
    atol=1e-5,
    rtol=1e-5,
    sum_atol=None,
    err_msg="",
):
    actual_mv, actual_m = actual
    expected_mv, expected_m = expected
    np.testing.assert_allclose(
        np.asarray(actual_mv),
        np.asarray(expected_mv),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )
    np.testing.assert_allclose(
        np.asarray(actual_m),
        np.asarray(expected_m),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )
    np.testing.assert_allclose(
        float(actual_m.sum()),
        float(expected_m.sum()),
        atol=atol if sum_atol is None else sum_atol,
        rtol=0.0,
        err_msg=err_msg,
    )


def assert_morton_prepared(backend, params, state, stress):
    """cuda_v2 reorders particles by Morton code with no bucket structure."""
    prepared = block_until_ready(backend.prepare(params, state, stress))
    expected = block_until_ready(morton_order(params, state, stress))

    np.testing.assert_array_equal(np.asarray(prepared.x), np.asarray(expected.x))
    np.testing.assert_array_equal(np.asarray(prepared.v), np.asarray(expected.v))
    np.testing.assert_array_equal(np.asarray(prepared.C), np.asarray(expected.C))
    np.testing.assert_array_equal(np.asarray(prepared.F), np.asarray(expected.F))
    np.testing.assert_array_equal(np.asarray(prepared.stress), np.asarray(expected.stress))
    assert prepared.bucket_start is None
    assert prepared.bucket_bounds is None


def assert_supercell_prepared(backend, params, state, stress):
    """cuda_v3 sorts particles by super-cell and emits CSR-style bucket_start."""
    super_cell = getattr(backend, "super_cell", V3_SUPER_CELL_WIDTH)
    prepared = block_until_ready(backend.prepare(params, state, stress))
    expected = block_until_ready(
        supercell_order(params, state, stress, super_cell)
    )

    np.testing.assert_array_equal(np.asarray(prepared.x), np.asarray(expected.x))
    np.testing.assert_array_equal(np.asarray(prepared.v), np.asarray(expected.v))
    np.testing.assert_array_equal(np.asarray(prepared.C), np.asarray(expected.C))
    np.testing.assert_array_equal(np.asarray(prepared.F), np.asarray(expected.F))
    np.testing.assert_array_equal(np.asarray(prepared.stress), np.asarray(expected.stress))
    assert prepared.bucket_bounds is None
    assert prepared.bucket_start is not None
    super_grids = params.num_grids // super_cell
    assert prepared.bucket_start.shape == (super_grids**3 + 1,)
    np.testing.assert_array_equal(
        np.asarray(prepared.bucket_start), np.asarray(expected.bucket_start)
    )
