// p2g_scatter_warp.cu — Fused P2G scatter with warp-level reduction.
//
// Standalone kernel (no JAX FFI). Each thread handles one particle:
//   1. Calls p2g_compute to get 27 (momentum, mass, grid_idx) contributions
//   2. For each contribution, uses __match_any_sync to find warp peers
//      targeting the same grid node
//   3. Reduces via warp_reduce_masked (butterfly __shfl_xor_sync)
//   4. Only the leader (lowest set bit in peer mask) does atomicAdd
//
// This reduces global atomics by a factor of k, where k is the average
// number of warp lanes per unique grid node (typically 2-8 at high density).
//
// Requires sm_70+ for __match_any_sync.
//
// Grid buffers (grid_mv, grid_m) must be pre-zeroed by the caller.

#include "p2g_compute.cuh"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

// ---------------------------------------------------------------------------
// Warp-level reduction helper
// ---------------------------------------------------------------------------

// Reduce `val` across all lanes in `mask` using butterfly shuffle.
// Returns the sum in ALL lanes of the group (not just the leader).
__device__ __forceinline__ float warp_reduce_masked(float val, unsigned mask) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(mask, val, delta);
        // Only add if the other lane is actually in our group
        if (mask & (1u << ((threadIdx.x & 31) ^ delta)))
            val += other;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Fused P2G kernel with warp reduction
// ---------------------------------------------------------------------------

extern "C"
__global__ void p2g_scatter_warp(
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

    int lane = threadIdx.x & 31;

    ParticleContrib contrib[STENCIL];
    p2g_compute(
        x[pid*3+0], x[pid*3+1], x[pid*3+2],
        v[pid*3+0], v[pid*3+1], v[pid*3+2],
        &C[pid*9], &F[pid*9],
        dt, vol, p_mass, inv_dx, num_grids,
        mu_0, lambda_0, contrib
    );

    for (int i = 0; i < STENCIL; i++) {
        int gid = contrib[i].grid_idx;

        // Find all lanes in this warp targeting the same grid node
        unsigned peers = __match_any_sync(FULL_MASK, gid);

        // Reduce contributions across matching lanes
        float mv0  = warp_reduce_masked(contrib[i].mv[0], peers);
        float mv1  = warp_reduce_masked(contrib[i].mv[1], peers);
        float mv2  = warp_reduce_masked(contrib[i].mv[2], peers);
        float mass = warp_reduce_masked(contrib[i].m, peers);

        // Only the leader (lowest lane in group) does the atomic
        int leader = __ffs(peers) - 1;  // __ffs returns 1-indexed
        if (lane == leader) {
            atomicAdd(&grid_mv[gid*3+0], mv0);
            atomicAdd(&grid_mv[gid*3+1], mv1);
            atomicAdd(&grid_mv[gid*3+2], mv2);
            atomicAdd(&grid_m[gid], mass);
        }
    }
}
