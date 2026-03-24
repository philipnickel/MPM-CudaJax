// p2g_scatter_naive.cu — Fused P2G scatter with naive atomicAdd.
//
// Standalone kernel (no JAX FFI). Each thread handles one particle:
//   1. Calls p2g_compute to get 27 (momentum, mass, grid_idx) contributions
//   2. Scatters all 27 via global atomicAdd (4 atomics per node = 108 total)
//
// Callable from cuda.core via extern "C" __global__.
//
// Grid buffers (grid_mv, grid_m) must be pre-zeroed by the caller.

#include "p2g_compute.cuh"

#define BLOCK_SIZE 256

extern "C"
__global__ void p2g_scatter_naive(
    const float* __restrict__ x,        // (N, 3)
    const float* __restrict__ v,        // (N, 3)
    const float* __restrict__ C,        // (N, 3, 3) row-major
    const float* __restrict__ F,        // (N, 3, 3) row-major
    float* __restrict__ grid_mv,        // (G^3, 3) — must be pre-zeroed
    float* __restrict__ grid_m,         // (G^3,)   — must be pre-zeroed
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    int n_particles
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    ParticleContrib contrib[STENCIL];
    p2g_compute(
        x[pid*3+0], x[pid*3+1], x[pid*3+2],
        v[pid*3+0], v[pid*3+1], v[pid*3+2],
        &C[pid*9], &F[pid*9],
        dt, vol, p_mass, inv_dx, num_grids,
        mu_0, lambda_0, contrib
    );

    #pragma unroll
    for (int i = 0; i < STENCIL; i++) {
        atomicAdd(&grid_mv[contrib[i].grid_idx*3+0], contrib[i].mv[0]);
        atomicAdd(&grid_mv[contrib[i].grid_idx*3+1], contrib[i].mv[1]);
        atomicAdd(&grid_mv[contrib[i].grid_idx*3+2], contrib[i].mv[2]);
        atomicAdd(&grid_m[contrib[i].grid_idx], contrib[i].m);
    }
}
