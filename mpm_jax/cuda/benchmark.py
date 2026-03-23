"""Benchmark: JAX XLA P2G vs hand-written CUDA P2G kernel."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig

from mpm_jax.solver import make_params, OFFSET_27, _compute_weights, _linearize_index, grid_update


def jax_p2g(params, x, v, C, stress):
    """Extract just the P2G portion from the solver for benchmarking."""
    inv_dx = params.inv_dx
    dx = params.dx
    G = params.num_grids
    dt = params.dt
    vol = params.vol
    p_mass = params.p_mass

    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)

    w, dw = _compute_weights(fx)
    weight = jnp.einsum('bi,bj,bk->bijk', w[:, :, 0], w[:, :, 1], w[:, :, 2]).reshape(-1, 27)
    dweight = inv_dx * jnp.stack([
        jnp.einsum('bi,bj,bk->bijk', dw[:, :, 0],  w[:, :, 1],  w[:, :, 2]),
        jnp.einsum('bi,bj,bk->bijk',  w[:, :, 0], dw[:, :, 1],  w[:, :, 2]),
        jnp.einsum('bi,bj,bk->bijk',  w[:, :, 0],  w[:, :, 1], dw[:, :, 2]),
    ], axis=-1).reshape(-1, 27, 3)

    dpos = (OFFSET_27[None, :, :] - fx[:, None, :]) * dx
    index = _linearize_index(base, OFFSET_27, G)

    mv = (
        -dt * vol * jnp.einsum('bij,bkj->bki', stress, dweight)
        + p_mass * weight[:, :, None] * (v[:, None, :] + jnp.einsum('bij,bkj->bki', C, dpos))
    )
    grid_mv = jnp.zeros((G ** 3, 3)).at[index].add(mv.reshape(-1, 3))
    grid_m = jnp.zeros((G ** 3,)).at[index].add((weight * p_mass).reshape(-1))
    return grid_mv, grid_m


jax_p2g_jit = jax.jit(jax_p2g, static_argnums=(0,))


@hydra.main(version_base=None, config_path="../../conf", config_name="benchmark")
def main(cfg: DictConfig):
    N = cfg.n_particles
    G = cfg.num_grids
    num_warmup = cfg.num_warmup
    num_runs = cfg.num_runs

    params = make_params(n_particles=N, num_grids=G)

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    x = jax.random.uniform(keys[0], (N, 3), minval=0.1, maxval=0.9)
    v = jax.random.normal(keys[1], (N, 3)) * 0.01
    C = jnp.zeros((N, 3, 3))
    stress = jax.random.normal(keys[2], (N, 3, 3)) * 100.0

    print(f"\n=== P2G Benchmark: N={N}, G={G} ===\n")

    # Warmup
    for _ in range(num_warmup):
        gmv, gm = jax_p2g_jit(params, x, v, C, stress)
        gmv.block_until_ready()

    # Timed runs
    jax_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        gmv, gm = jax_p2g_jit(params, x, v, C, stress)
        gmv.block_until_ready()
        jax_times.append(time.perf_counter() - t0)

    jax_mean = np.mean(jax_times) * 1000
    jax_std = np.std(jax_times) * 1000
    print(f"JAX XLA:  {jax_mean:.3f} +/- {jax_std:.3f} ms")

    try:
        from mpm_jax.cuda.p2g_custom_op import cuda_p2g
        import pycuda.driver as cuda_drv

        x_np = np.array(x, dtype=np.float64)
        v_np = np.array(v, dtype=np.float64)
        C_np = np.array(C, dtype=np.float64)
        stress_np = np.array(stress, dtype=np.float64)

        for _ in range(num_warmup):
            cuda_p2g(x_np, v_np, C_np, stress_np,
                     N, G, params.dt, params.vol, params.p_mass, params.inv_dx, params.dx)

        cuda_times = []
        for _ in range(num_runs):
            start = cuda_drv.Event()
            end = cuda_drv.Event()
            start.record()
            cuda_p2g(x_np, v_np, C_np, stress_np,
                     N, G, params.dt, params.vol, params.p_mass, params.inv_dx, params.dx)
            end.record()
            end.synchronize()
            cuda_times.append(start.time_till(end))

        cuda_mean = np.mean(cuda_times)
        cuda_std = np.std(cuda_times)
        print(f"CUDA:     {cuda_mean:.3f} +/- {cuda_std:.3f} ms")
        print(f"Speedup:  {jax_mean / cuda_mean:.2f}x (CUDA over JAX)")

    except (ImportError, RuntimeError) as e:
        print(f"\nCUDA benchmark skipped: {e}")
        print("Install pycuda to enable: pip install pycuda")


if __name__ == '__main__':
    main()
