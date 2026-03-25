"""Benchmark all P2G kernels at various particle counts."""
import time
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulate import get_particles

cfg = OmegaConf.create({
    "material": {"elasticity": {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4},
                 "plasticity": {"name": "IdentityPlasticity"}},
    "kernel": {"name": "jax", "block_size": 256},
})

try:
    from mpm_jax.cuda.runtime import CudaRuntime
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
    runtime = CudaRuntime()
    has_cuda = True
except Exception:
    has_cuda = False

sizes = [2000, 10000, 50000, 100000, 200000, 500000]
n_steps = 50

print(f"{'Kernel':<12} {'N':>8} {'P2G ms':>8} {'Step ms':>8} {'Steps/s':>8}")
print("-" * 52)

for N in sizes:
    state = get_particles(N, [0.5,0.5,0.5], [0.5,0.5,0.5], [0,0,-0.5])
    actual_n = state.x.shape[0]
    params = make_params(n_particles=actual_n, num_grids=25, dt=3e-4, size=[0.5,0.5,0.5])

    kernels = {"jax": make_jax_p2g(cfg)}
    if has_cuda:
        kernels["cuda_naive"] = make_cuda_naive_p2g(cfg, runtime)
        warp_cfg = OmegaConf.merge(cfg, {"kernel": {"name": "cuda_warp"}})
        kernels["cuda_warp"] = make_cuda_warp_p2g(warp_cfg, runtime)

    for name, p2g_fn in kernels.items():
        # warmup
        mv, m = p2g_fn(state, params)
        gv = grid_update(mv, m, params)
        g2p(state, gv, params)
        jax.block_until_ready(gv)

        p2g_times = []
        t0 = time.perf_counter()
        for _ in range(n_steps):
            tp = time.perf_counter()
            mv, m = p2g_fn(state, params)
            jax.block_until_ready((mv, m))
            p2g_times.append(time.perf_counter() - tp)
            gv = grid_update(mv, m, params)
            state_new = g2p(state, gv, params)
            jax.block_until_ready(state_new.x)
        total = time.perf_counter() - t0

        mean_p2g = sum(p2g_times) / len(p2g_times) * 1000
        mean_step = total / n_steps * 1000
        steps_s = n_steps / total
        print(f"{name:<12} {actual_n:>8} {mean_p2g:>8.2f} {mean_step:>8.2f} {steps_s:>8.1f}")
