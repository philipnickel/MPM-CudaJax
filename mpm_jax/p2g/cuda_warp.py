"""CUDA P2G with warp-reduced scatter."""
import math
import numpy as np
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams
from mpm_jax.cuda.runtime import CudaRuntime, get_ptr


def make_cuda_warp_p2g(cfg, runtime: CudaRuntime):
    """Compile warp-reduced scatter kernel and return P2G closure."""
    kernel = runtime.compile_kernel("p2g_scatter_warp.cu", "p2g_scatter_warp")

    mat = cfg.material.elasticity
    E, nu = float(mat.E), float(mat.nu)
    mu_0 = np.float32(E / (2.0 * (1.0 + nu)))
    lambda_0 = np.float32(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    block_size = int(cfg.kernel.get("block_size", 256))

    def p2g(state: MPMState, params: MPMParams):
        N = int(params.n_particles)
        G = int(params.num_grids)

        jax.block_until_ready((state.x, state.v, state.C, state.F))

        grid_mv = jnp.zeros((G ** 3, 3), dtype=jnp.float32)
        grid_m = jnp.zeros((G ** 3,), dtype=jnp.float32)

        grid_dim = math.ceil(N / block_size)

        runtime.launch(
            kernel, grid_dim, block_size,
            get_ptr(state.x), get_ptr(state.v),
            get_ptr(state.C), get_ptr(state.F),
            get_ptr(grid_mv), get_ptr(grid_m),
            np.float32(params.dt), np.float32(params.vol), np.float32(params.p_mass),
            np.float32(params.inv_dx), np.int32(G),
            mu_0, lambda_0,
            np.int32(N),
        )

        return grid_mv, grid_m

    return p2g
