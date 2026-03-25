"""Minimal warp reduction test — just mass scatter."""
import jax.numpy as jnp
from mpm_jax.state import MPMState, make_params
from mpm_jax.cuda.runtime import CudaRuntime, get_ptr
from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
from omegaconf import OmegaConf
import numpy as np

runtime = CudaRuntime()
cfg = OmegaConf.create({
    "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                 "plasticity": {"name": "IdentityPlasticity"}},
    "kernel": {"name": "cuda_naive", "block_size": 256},
})

# Just 32 particles (one warp) at the same position — all should target same grid nodes
N = 32
params = make_params(n_particles=N, num_grids=8)
state = MPMState(
    x=jnp.full((N, 3), 0.5),
    v=jnp.zeros((N, 3)),
    C=jnp.zeros((N, 3, 3)),
    F=jnp.tile(jnp.eye(3), (N, 1, 1)),
)

naive_p2g = make_cuda_naive_p2g(cfg, runtime)
warp_cfg = OmegaConf.merge(cfg, {"kernel": {"name": "cuda_warp"}})
warp_p2g = make_cuda_warp_p2g(warp_cfg, runtime)

nmv, nm = naive_p2g(state, params)
wmv, wm = warp_p2g(state, params)

print(f"32 particles at same position (one warp, all same gids):")
print(f"  Naive mass sum: {nm.sum():.4f}")
print(f"  Warp  mass sum: {wm.sum():.4f}")
print(f"  Expected: {N * params.p_mass:.4f}")
print(f"  Mass max diff: {jnp.max(jnp.abs(nm - wm)):.6f}")
