"""Compare full simulation: CUDA naive vs JAX baseline."""
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p

cfg = OmegaConf.create({
    "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                 "plasticity": {"name": "IdentityPlasticity"}},
    "kernel": {"name": "cuda_naive", "block_size": 256},
})

np.random.seed(42)
N = 200
x_np = np.random.uniform(0.25, 0.75, (N, 3)).astype(np.float32)
v_np = np.tile([0, 0, -0.5], (N, 1)).astype(np.float32)
C_np = np.zeros((N, 3, 3), dtype=np.float32)
F_np = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))

params = make_params(n_particles=N, num_grids=25, dt=3e-4, size=[0.5, 0.5, 0.5])

def run_sim(p2g_fn, n_frames):
    state = MPMState(x=jnp.array(x_np), v=jnp.array(v_np), C=jnp.array(C_np), F=jnp.array(F_np))
    for frame in range(n_frames):
        for step in range(10):
            mv, m = p2g_fn(state, params)
            gv = grid_update(mv, m, params)
            state = g2p(state, gv, params)
    return state

p2g_jax = make_jax_p2g(cfg)

from mpm_jax.cuda.runtime import CudaRuntime
from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
runtime = CudaRuntime()
p2g_naive = make_cuda_naive_p2g(cfg, runtime)
warp_cfg = OmegaConf.merge(cfg, {"kernel": {"name": "cuda_warp"}})
p2g_warp = make_cuda_warp_p2g(warp_cfg, runtime)

for n_frames in [10, 50, 150]:
    state_j = run_sim(p2g_jax, n_frames)
    state_n = run_sim(p2g_naive, n_frames)
    state_w = run_sim(p2g_warp, n_frames)

    jx = np.array(state_j.x)
    nx = np.array(state_n.x)
    wx = np.array(state_w.x)

    print(f"After {n_frames} frames ({n_frames*10} steps):")
    print(f"  Naive vs JAX:  x={np.max(np.abs(jx-nx)):.6f}  v={np.max(np.abs(np.array(state_j.v)-np.array(state_n.v))):.6f}  F={np.max(np.abs(np.array(state_j.F)-np.array(state_n.F))):.6f}")
    print(f"  Warp  vs JAX:  x={np.max(np.abs(jx-wx)):.6f}  v={np.max(np.abs(np.array(state_j.v)-np.array(state_w.v))):.6f}  F={np.max(np.abs(np.array(state_j.F)-np.array(state_w.F))):.6f}")
    print(f"  z mean — JAX: {jx[:,2].mean():.4f}, Naive: {nx[:,2].mean():.4f}, Warp: {wx[:,2].mean():.4f}")
    print()
