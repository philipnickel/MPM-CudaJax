// p2g_scatter_warp.cu — Fused P2G scatter with warp-level reduction.
//
// Each thread handles one particle (inactive threads use gid=-1).
// Uses __match_any_sync + __shfl_xor_sync butterfly to reduce
// contributions from warp lanes targeting the same grid node.
//
// Requires sm_70+ for __match_any_sync.
// Grid buffers must be pre-zeroed by the caller.

#include "p2g_compute.cuh"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

// Butterfly XOR reduction across peer lanes.
// All 32 lanes participate in __shfl_xor_sync (FULL_MASK).
// Only values from actual peers are accumulated.
__device__ __forceinline__ float warp_reduce_peers(float val, unsigned peers) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(FULL_MASK, val, delta);
        if (peers & (1u << ((threadIdx.x & 31) ^ delta)))
            val += other;
    }
    return val;
}

extern "C"
__global__ void p2g_scatter_warp(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ C,
    const float* __restrict__ F,
    float* __restrict__ grid_mv,
    float* __restrict__ grid_m,
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    int n_particles
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    bool active = (pid < n_particles);

    // All threads participate in the loop — inactive threads use sentinel gid
    ParticleContrib contrib[STENCIL];

    if (active) {
        p2g_compute(
            x[pid*3+0], x[pid*3+1], x[pid*3+2],
            v[pid*3+0], v[pid*3+1], v[pid*3+2],
            &C[pid*9], &F[pid*9],
            dt, vol, p_mass, inv_dx, num_grids,
            mu_0, lambda_0, contrib
        );
    } else {
        // Zero out contributions — sentinel gid = -1 won't match any active lane
        for (int i = 0; i < STENCIL; i++) {
            contrib[i].mv[0] = contrib[i].mv[1] = contrib[i].mv[2] = 0.0f;
            contrib[i].m = 0.0f;
            contrib[i].grid_idx = -1;
        }
    }

    for (int i = 0; i < STENCIL; i++) {
        int gid = contrib[i].grid_idx;

        // All 32 lanes participate — inactive lanes have gid=-1
        unsigned peers = __match_any_sync(FULL_MASK, gid);

        float mv0  = warp_reduce_peers(contrib[i].mv[0], peers);
        float mv1  = warp_reduce_peers(contrib[i].mv[1], peers);
        float mv2  = warp_reduce_peers(contrib[i].mv[2], peers);
        float mass = warp_reduce_peers(contrib[i].m, peers);

        int leader = __ffs(peers) - 1;
        if (lane == leader && gid >= 0) {
            atomicAdd(&grid_mv[gid*3+0], mv0);
            atomicAdd(&grid_mv[gid*3+1], mv1);
            atomicAdd(&grid_mv[gid*3+2], mv2);
            atomicAdd(&grid_m[gid], mass);
        }
    }
}
