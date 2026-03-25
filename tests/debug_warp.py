"""Debug warp kernel differences."""
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.cuda.runtime import CudaRuntime
from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
from mpm_jax.p2g.jax import make_jax_p2g

runtime = CudaRuntime()
cfg = OmegaConf.create({
    "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                 "plasticity": {"name": "IdentityPlasticity"}},
    "kernel": {"name": "cuda_naive", "block_size": 256},
})

N = 100
params = make_params(n_particles=N, num_grids=8)
key = jax.random.PRNGKey(42)
state = MPMState(
    x=jax.random.uniform(key, (N, 3)) * 0.5 + 0.25,
    v=jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1,
    C=jnp.zeros((N, 3, 3)),
    F=jnp.tile(jnp.eye(3), (N, 1, 1)),
)

jax_p2g = make_jax_p2g(cfg)
naive_p2g = make_cuda_naive_p2g(cfg, runtime)
warp_cfg = OmegaConf.merge(cfg, {"kernel": {"name": "cuda_warp"}})
warp_p2g = make_cuda_warp_p2g(warp_cfg, runtime)

jmv, jm = jax_p2g(state, params)
nmv, nm = naive_p2g(state, params)
wmv, wm = warp_p2g(state, params)

print(f"Naive vs JAX:  mass max diff = {jnp.max(jnp.abs(jm - nm)):.8f}")
print(f"Warp  vs JAX:  mass max diff = {jnp.max(jnp.abs(jm - wm)):.8f}")
print(f"Warp  vs Naive: mass max diff = {jnp.max(jnp.abs(nm - wm)):.8f}")
print(f"Naive vs JAX:  mv max diff   = {jnp.max(jnp.abs(jmv - nmv)):.8f}")
print(f"Warp  vs JAX:  mv max diff   = {jnp.max(jnp.abs(jmv - wmv)):.8f}")
print(f"Warp  vs Naive: mv max diff  = {jnp.max(jnp.abs(nmv - wmv)):.8f}")
print(f"Mass sum — JAX: {jm.sum():.6f}, Naive: {nm.sum():.6f}, Warp: {wm.sum():.6f}")
