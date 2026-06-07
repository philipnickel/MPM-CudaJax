"""cuda_v2 must match cuda_v1 up to atomic-order f32 drift.

Both kernels are bit-identical in their per-particle math; only the scatter
strategy differs (v2 adds a warp-shuffle reduction before each atomicAdd).
The same drift sources as the rest of the CUDA equivalence
suite apply (non-deterministic atomicAdd ordering), so we use the same
tolerances as the v1-vs-JAX comparison.

Skipped when the native CUDA extension isn't built or there's no GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from types import SimpleNamespace

from mpm_jax.p2g.backends import CudaV1Backend, CudaV2Backend
from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.p2g.cuda.p2g_cuda import CudaV1P2G, CudaV2P2G
from mpm_jax.solver import MPMSolver, RuntimeConfig
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
    """Run a short sim under both CUDA kernels and compare final state."""
    _require_kernels(CudaV1P2G, CudaV2P2G)
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

    sim = OmegaConf.create(
        {
            "n_particles": n,
            "num_grids": num_grids,
            "dt": params.dt,
            "gravity": [0.0, 0.0, -9.8],
            "rho": 1000.0,
            "clip_bound": 0.5,
            "damping": 1.0,
            "size": [1.0, 1.0, 1.0],
            "initial_velocity": [0.0, 0.0, -0.5],
            "center": [0.5, 0.5, 0.5],
            "steps_per_frame": steps_per_frame,
        }
    )
    material = SimpleNamespace(elasticity=elasticity_fn)
    solver_v1 = MPMSolver(
        RuntimeConfig(material=material, sim=sim, backend=CudaV1Backend(num_grids))
    )
    solver_v2 = MPMSolver(
        RuntimeConfig(material=material, sim=sim, backend=CudaV2Backend(num_grids))
    )
    solver_v1.state = state0
    solver_v2.state = state0

    for _ in range(num_frames):
        s1 = solver_v1._frame(solver_v1.state)
        s2 = solver_v2._frame(solver_v2.state)
        solver_v1.state = s1
        solver_v2.state = s2
    jax.block_until_ready(s1.x)
    jax.block_until_ready(s2.x)

    # Same scatter-only tolerance band as the rest of the CUDA-equivalence
    # suite: positions ~1e-4, velocities ~5e-3, F ~1e-4.
    np.testing.assert_allclose(np.asarray(s1.x), np.asarray(s2.x), atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.v), np.asarray(s2.v), atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.F), np.asarray(s2.F), atol=1e-4, rtol=1e-3)
