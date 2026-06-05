import numpy as np
import pytest

import jax
import jax.numpy as jnp

from omegaconf import OmegaConf

from mpm_jax.types import MPMState, MPMParams


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _cuda_kernels_available(*kinds: str) -> bool:
    if not _has_cuda():
        return False
    from mpm_jax.cuda.p2g_cuda import is_available

    return all(is_available(kind) for kind in kinds)


def _cutile_available() -> bool:
    if not _has_cuda():
        return False
    try:
        import cuda.tile  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
        return False
    return True


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
    params = MPMParams(OmegaConf.create({
        "n_particles": n,
        "num_grids": num_grids,
        "dt": 3e-4,
        "gravity": [0.0, 0.0, -9.8],
        "rho": 1000.0,
        "clip_bound": 0.5,
        "damping": 1.0,
        "size": [0.4, 0.4, 0.4],
    }))
    return params, state, stress


def _p2g_output(backend, params, state, stress):
    @jax.jit
    def run(state, stress):
        prepared = backend.prepare(params, state, stress)
        return backend.p2g(params, prepared)

    out = run(state, stress)
    jax.block_until_ready(out)
    return out


@pytest.mark.skipif(
    not _cuda_kernels_available(
        "inline", "v2_inline", "v3_inline", "v4_inline"
    ),
    reason="CUDA P2G kernels not built or no GPU backend available",
)
def test_cuda_p2g_variants_match_jax_scan():
    from mpm_jax.backends import build_backend, jax_baseline_backend

    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(jax_baseline_backend(), params, state, stress)

    for name in ("cuda_v1_inline", "cuda_v2_inline", "cuda_v3_inline", "cuda_v4_inline"):
        backend = build_backend(name, params.num_grids)
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)
        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0)


@pytest.mark.skipif(
    not _cutile_available(),
    reason="cuTile/JAX backend requires a GPU and cuda-tile",
)
def test_cutile_p2g_matches_jax_scan():
    from mpm_jax.backends import build_backend, jax_baseline_backend

    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(jax_baseline_backend(), params, state, stress)

    for backend in (
        build_backend("cutile_v6_atomic_tile", params.num_grids, autotune=False),
    ):
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)

        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0)
