"""CUDA P2G with naive atomicAdd scatter."""
import math
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams
from mpm_jax.cuda.runtime import CudaRuntime, get_ptr


def make_cuda_naive_p2g(cfg, runtime: CudaRuntime):
    """Compile naive scatter kernel and return P2G closure."""
    kernel = runtime.compile_kernel("p2g_scatter_naive.cu", "p2g_scatter_naive")

    mat = cfg.material.elasticity
    E, nu = float(mat.E), float(mat.nu)
    mu_0 = E / (2.0 * (1.0 + nu))
    lambda_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    block_size = cfg.kernel.get("block_size", 256)

    def p2g(state: MPMState, params: MPMParams):
        N = params.n_particles
        G = int(params.num_grids)

        jax.block_until_ready((state.x, state.v, state.C, state.F))

        grid_mv = jnp.zeros((G ** 3, 3), dtype=jnp.float32)
        grid_m = jnp.zeros((G ** 3,), dtype=jnp.float32)

        grid_dim = math.ceil(N / block_size)

        runtime.launch(
            kernel, grid=grid_dim, block=block_size,
            get_ptr(state.x), get_ptr(state.v),
            get_ptr(state.C), get_ptr(state.F),
            get_ptr(grid_mv), get_ptr(grid_m),
            params.dt, params.vol, params.p_mass,
            params.inv_dx, G,
            mu_0, lambda_0,
            N,
        )

        return grid_mv, grid_m

    return p2g
