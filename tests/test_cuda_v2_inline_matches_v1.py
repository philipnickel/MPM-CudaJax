"""cuda_v2 must match cuda_v1 up to atomic-order f32 drift.

Both kernels are bit-identical in their per-particle math; only the scatter
strategy differs (v2_inline adds a warp-shuffle reduction before each
atomicAdd). The same drift sources as the rest of the CUDA equivalence
suite apply (non-deterministic atomicAdd ordering), so we use the same
tolerances as the v1-vs-JAX comparison.

Skipped when the native CUDA extension isn't built or there's no GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.backends import CudaV1Backend, CudaV2Backend
from mpm_jax.boundary import StickyPlane, bind_boundaries
from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.cuda.p2g_cuda import register_p2g_inline, register_p2g_v2_inline
from mpm_jax.solver import build_backend_frame
from mpm_jax.types import MPMState, MPMParams


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _require_kernels(*registrars):
    if not _has_cuda():
        pytest.skip("cuda_v1 / cuda_v2 require a GPU backend")
    if not all(registrar() for registrar in registrars):
        pytest.skip("cuda_v1 / cuda_v2 native extension not built")


def test_cuda_v2_matches_v1():
    """Run a short sim under both inline kernels and compare final state."""
    _require_kernels(register_p2g_inline, register_p2g_v2_inline)
    n = 2000
    num_grids = 16
    rng = np.random.RandomState(0)
    x0 = jnp.array(rng.rand(n, 3).astype(np.float32) * 0.4 + 0.3)
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

    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    bcs = [
        StickyPlane(
            point=(1.0, 1.0, 0.02),
            normal=(0.0, 0.0, 1.0),
            start_time=0.0,
            end_time=1e3,
        )
    ]
    post_fn = bind_boundaries(bcs, grid_x, params.dx)

    # jelly material (StVK elasticity, no plasticity): stress stays on the JAX
    # side without a cuSOLVER dependence.
    elasticity_fn = stvk_elasticity_jacobi(E=2e6, nu=0.4)

    state0 = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    steps_per_frame = 5
    num_frames = 3

    jit_v1 = build_backend_frame(
        params,
        elasticity_fn,
        post_fn,
        CudaV1Backend(num_grids),
        steps_per_frame,
    )
    jit_v2 = build_backend_frame(
        params,
        elasticity_fn,
        post_fn,
        CudaV2Backend(num_grids),
        steps_per_frame,
    )

    s1 = state0
    s2 = state0
    for _ in range(num_frames):
        s1 = jit_v1(s1)
        s2 = jit_v2(s2)
    jax.block_until_ready(s1.x)
    jax.block_until_ready(s2.x)

    # Same scatter-only tolerance band as the rest of the CUDA-equivalence
    # suite: positions ~1e-4, velocities ~5e-3, F ~1e-4.
    np.testing.assert_allclose(np.asarray(s1.x), np.asarray(s2.x), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.v), np.asarray(s2.v), atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.F), np.asarray(s2.F), atol=1e-4, rtol=1e-3)
