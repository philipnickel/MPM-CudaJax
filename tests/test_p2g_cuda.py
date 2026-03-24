"""Tests for CUDA P2G variants. Skipped if no GPU available."""
import pytest
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params

try:
    from mpm_jax.cuda.runtime import CudaRuntime
    _runtime = CudaRuntime()
    _has_cuda = True
except Exception:
    _has_cuda = False

requires_cuda = pytest.mark.skipif(not _has_cuda, reason="No CUDA runtime")

_cfg = OmegaConf.create({
    "material": {
        "elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
        "plasticity": {"name": "IdentityPlasticity"},
    },
    "kernel": {"name": "cuda_naive", "block_size": 256},
})


def _make_state(N, params, deformed=False):
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (N, 3)) * 0.5 + 0.25
    v = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
    C = jnp.zeros((N, 3, 3))
    if deformed:
        # Non-trivial F to exercise SVD (small perturbation from identity)
        F = jnp.tile(jnp.eye(3), (N, 1, 1)) + jax.random.normal(jax.random.PRNGKey(2), (N, 3, 3)) * 0.05
    else:
        F = jnp.tile(jnp.eye(3), (N, 1, 1))
    return MPMState(x=x, v=v, C=C, F=F)


@requires_cuda
def test_cuda_naive_shapes():
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    N = 100
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    p2g_fn = make_cuda_naive_p2g(_cfg, _runtime)
    grid_mv, grid_m = p2g_fn(state, params)
    G = params.num_grids
    assert grid_mv.shape == (G**3, 3)
    assert grid_m.shape == (G**3,)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))


@requires_cuda
def test_cuda_naive_matches_jax():
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    from mpm_jax.p2g.jax import make_jax_p2g
    N = 100
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    jax_p2g = make_jax_p2g(_cfg)
    cuda_p2g = make_cuda_naive_p2g(_cfg, _runtime)
    jax_mv, jax_m = jax_p2g(state, params)
    cuda_mv, cuda_m = cuda_p2g(state, params)
    assert jnp.allclose(jax_m, cuda_m, atol=1e-5)
    assert jnp.allclose(jax_mv, cuda_mv, atol=1e-5)


@requires_cuda
def test_cuda_warp_matches_jax():
    from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
    from mpm_jax.p2g.jax import make_jax_p2g
    N = 100
    params = make_params(n_particles=N, num_grids=8)
    state = _make_state(N, params)
    warp_cfg = OmegaConf.merge(_cfg, {"kernel": {"name": "cuda_warp"}})
    jax_p2g = make_jax_p2g(_cfg)
    cuda_p2g = make_cuda_warp_p2g(warp_cfg, _runtime)
    jax_mv, jax_m = jax_p2g(state, params)
    cuda_mv, cuda_m = cuda_p2g(state, params)
    assert jnp.allclose(jax_m, cuda_m, atol=1e-5)
    assert jnp.allclose(jax_mv, cuda_mv, atol=1e-5)


@requires_cuda
def test_cuda_naive_deformed_matches_jax():
    """Test with non-trivial F to exercise SVD."""
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    from mpm_jax.p2g.jax import make_jax_p2g
    N = 200
    params = make_params(n_particles=N, num_grids=10)
    state = _make_state(N, params, deformed=True)
    jax_p2g = make_jax_p2g(_cfg)
    cuda_p2g = make_cuda_naive_p2g(_cfg, _runtime)
    jax_mv, jax_m = jax_p2g(state, params)
    cuda_mv, cuda_m = cuda_p2g(state, params)
    assert jnp.allclose(jax_m, cuda_m, atol=1e-4)
    assert jnp.allclose(jax_mv, cuda_mv, atol=1e-2), \
        f"max mv diff: {jnp.max(jnp.abs(jax_mv - cuda_mv))}"
