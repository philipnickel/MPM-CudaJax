"""cuda_v2 must match cuda_v1 up to atomic-order f32 drift.

Both kernels are bit-identical in their per-particle math; cuda_v2 adds Morton
ordering plus a warp-shuffle reduction before each atomicAdd. The same drift
sources as the rest of the CUDA equivalence suite apply (non-deterministic
atomicAdd ordering), so we use the same tolerances as the v1-vs-JAX comparison.

Skipped when the native CUDA extension isn't built or there's no GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.p2g.backends import CudaV1Backend, CudaV2Backend
from mpm_jax.p2g.cuda.p2g_cuda import CudaV1P2G, CudaV2P2G
from mpm_jax.types import MPMState, MPMParams


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _require_kernels(*kernel_types):
    if not _has_cuda():
        pytest.skip("cuda_v1 / cuda_v2 require a GPU backend")
    try:
        for kernel_type in kernel_types:
            kernel_type()
    except ImportError as exc:
        pytest.skip(f"cuda_v1 / cuda_v2 native extension not built: {exc}")


def test_cuda_v2_matches_v1():
    """Compare scatter output directly; cuda_v2 changes particle order."""
    _require_kernels(CudaV1P2G, CudaV2P2G)
    n = 2000
    num_grids = 16
    rng = np.random.RandomState(0)
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
                "size": [1.0, 1.0, 1.0],
            }
        )
    )

    def p2g_output(backend):
        @jax.jit
        def run(state, stress):
            prepared = backend.prepare(params, state, stress)
            return backend.scatter(params, prepared)

        out = run(state, stress)
        jax.block_until_ready(out)
        return out

    mv1, m1 = p2g_output(CudaV1Backend(num_grids))
    mv2, m2 = p2g_output(CudaV2Backend(num_grids))

    np.testing.assert_allclose(np.asarray(mv1), np.asarray(mv2), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m2), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(float(m1.sum()), float(m2.sum()), atol=1e-4, rtol=0.0)
