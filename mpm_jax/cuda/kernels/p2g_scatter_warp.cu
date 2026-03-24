// p2g_scatter_warp.cu — Fused P2G scatter with warp-level reduction.
//
// Each thread handles one particle, computes 27 contributions via p2g_compute,
// then uses __match_any_sync + warp shuffle reduction so only one lane per
// unique grid node does the atomicAdd.
//
// Requires sm_70+ for __match_any_sync.
// Grid buffers (grid_mv, grid_m) must be pre-zeroed by the caller.

#include "p2g_compute.cuh"

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Warp-level reduction for lanes matching a grid node
// Uses explicit peer iteration: the leader reads each peer's value via
// __shfl_sync. Correct for arbitrary (non-contiguous) peer masks.
// ---------------------------------------------------------------------------

// Reduce val across lanes in `peers` using butterfly XOR shuffle.
// All active lanes get the sum (not just the leader).
// Ported from the original tested p2g_scatter_warp.cu.
__device__ __forceinline__ float warp_reduce_peers(float val, unsigned peers) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(peers, val, delta);
        // Only add if the XOR partner is actually in our peer group
        if (peers & (1u << ((threadIdx.x & 31) ^ delta)))
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

    // Get mask of active lanes in this warp
    unsigned active_mask = __activemask();

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

        // Find all active lanes targeting the same grid node
        unsigned peers = __match_any_sync(active_mask, gid);

        // Reduce contributions across matching lanes
        float mv0  = warp_reduce_peers(contrib[i].mv[0], peers);
        float mv1  = warp_reduce_peers(contrib[i].mv[1], peers);
        float mv2  = warp_reduce_peers(contrib[i].mv[2], peers);
        float mass = warp_reduce_peers(contrib[i].m, peers);

        // Only the leader (lowest lane in group) does the atomic
        int leader = __ffs(peers) - 1;
        if (lane == leader) {
            atomicAdd(&grid_mv[gid*3+0], mv0);
            atomicAdd(&grid_mv[gid*3+1], mv1);
            atomicAdd(&grid_mv[gid*3+2], mv2);
            atomicAdd(&grid_m[gid], mass);
        }
    }
}
