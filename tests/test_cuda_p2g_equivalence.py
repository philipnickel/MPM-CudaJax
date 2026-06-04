import numpy as np
import pytest

import jax
import jax.numpy as jnp

from mpm_jax.types import MPMState, make_params


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
    if not _cuda_kernels_available("g2p_fused"):
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
    params = make_params(
        n_particles=n,
        num_grids=num_grids,
        dt=3e-4,
        center=[0.5, 0.5, 0.5],
        size=[0.4, 0.4, 0.4],
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


@pytest.mark.skipif(
    not _cuda_kernels_available(
        "inline", "v2_inline", "v3_inline", "v4_inline", "g2p_fused"
    ),
    reason="CUDA P2G kernels not built or no GPU backend available",
)
def test_cuda_p2g_variants_match_jax_scan():
    from mpm_jax.backends import (
        cuda_v1_backend,
        cuda_v2_backend,
        cuda_v3_backend,
        cuda_v4_backend,
        jax_v1_5_backend,
    )

    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(jax_v1_5_backend(), params, state, stress)

    for backend in (
        cuda_v1_backend(num_grids=params.num_grids),
        cuda_v2_backend(num_grids=params.num_grids),
        cuda_v3_backend(num_grids=params.num_grids),
        cuda_v4_backend(num_grids=params.num_grids),
    ):
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
    reason="cuTile/JAX backend requires a GPU, cuda-tile, and the CUDA G2P kernel library",
)
def test_cutile_p2g_matches_jax_scan():
    from mpm_jax.backends import (
        cutile_v1_backend,
        cutile_v2_backend,
        cutile_v3_backend,
        cutile_v4_native4_backend,
        cutile_v5_sc4_tiledview_flush_backend,
        cutile_v6_sc4_colored_tiledview_store_backend,
        cutile_v7_sc4_colored_arena256_store_backend,
        jax_v1_5_backend,
    )

    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(jax_v1_5_backend(), params, state, stress)

    for backend in (
        cutile_v1_backend(num_grids=params.num_grids),
        cutile_v2_backend(num_grids=params.num_grids),
        cutile_v3_backend(num_grids=params.num_grids),
        cutile_v4_native4_backend(num_grids=params.num_grids),
        cutile_v5_sc4_tiledview_flush_backend(num_grids=params.num_grids),
        cutile_v6_sc4_colored_tiledview_store_backend(num_grids=params.num_grids),
        cutile_v7_sc4_colored_arena256_store_backend(num_grids=params.num_grids),
    ):
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)

        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0)


@pytest.mark.skipif(
    not _cuda_kernels_available("g2p_fused"),
    reason="Warp/JAX backend requires a GPU and the CUDA G2P kernel library",
)
def test_warp_supercell_p2g_matches_jax_scan():
    from mpm_jax.backends import (
        jax_v1_5_backend,
        warp_v3_supercell_backend,
        warp_v4_hashgrid_backend,
    )

    params, state, stress = _inputs()
    ref_mv, ref_m = _p2g_output(jax_v1_5_backend(), params, state, stress)

    for backend in (
        warp_v3_supercell_backend(num_grids=params.num_grids),
        warp_v4_hashgrid_backend(num_grids=params.num_grids),
    ):
        grid_mv, grid_m = _p2g_output(backend, params, state, stress)

        np.testing.assert_allclose(
            np.asarray(grid_mv), np.asarray(ref_mv), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(grid_m), np.asarray(ref_m), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(float(grid_m.sum()), float(ref_m.sum()), atol=1e-5, rtol=0.0)
