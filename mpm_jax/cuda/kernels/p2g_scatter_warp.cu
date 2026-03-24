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
// Uses __shfl_down_sync with the peer mask. The leader (lowest set bit)
// accumulates the sum.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_peers(float val, unsigned peers) {
    // Iteratively halve the peer group, accumulating into lower lanes
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_down_sync(peers, val, delta);
        val += other;
    }
    return val;  // correct result is in the leader (lowest bit of peers)
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
    int lane = threadIdx.x & 31;

    // Determine which lanes in this warp are active (have valid particles)
    bool active = (pid < n_particles);
    unsigned active_mask = __ballot_sync(0xFFFFFFFF, active);

    if (!active) return;

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
