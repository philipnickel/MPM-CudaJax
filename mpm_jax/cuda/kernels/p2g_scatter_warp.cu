// p2g_scatter_warp.cu — Fused P2G scatter with warp-aggregated atomics.
//
// Uses __match_any_sync to find warp lanes targeting the same grid node.
// Each lane still computes its own contribution, but only the leader does
// atomicAdd — accumulating peers' values via __shfl_sync reads.
//
// For mass: leader uses __popc(peers) * per-lane mass (since mass is uniform).
// For momentum: leader reads each peer's value via __shfl_sync and sums.
//
// Requires sm_70+ for __match_any_sync.
// Grid buffers must be pre-zeroed by the caller.

#include "p2g_compute.cuh"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

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

    // Use __activemask for partial warps (last warp may have < 32 active lanes)
    unsigned active = __activemask();

    for (int i = 0; i < STENCIL; i++) {
        int gid = contrib[i].grid_idx;
        unsigned peers = __match_any_sync(active, gid);
        int leader = __ffs(peers) - 1;

        if (lane == leader) {
            // Leader accumulates from all peers
            float mv0 = 0.0f, mv1 = 0.0f, mv2 = 0.0f, m = 0.0f;
            unsigned remaining = peers;
            while (remaining) {
                int src = __ffs(remaining) - 1;
                mv0 += __shfl_sync(peers, contrib[i].mv[0], src);
                mv1 += __shfl_sync(peers, contrib[i].mv[1], src);
                mv2 += __shfl_sync(peers, contrib[i].mv[2], src);
                m   += __shfl_sync(peers, contrib[i].m,      src);
                remaining &= remaining - 1;  // clear lowest bit
            }
            atomicAdd(&grid_mv[gid*3+0], mv0);
            atomicAdd(&grid_mv[gid*3+1], mv1);
            atomicAdd(&grid_mv[gid*3+2], mv2);
            atomicAdd(&grid_m[gid], m);
        } else {
            // Non-leaders: participate in all __shfl_sync calls
            unsigned remaining = peers;
            while (remaining) {
                int src = __ffs(remaining) - 1;
                __shfl_sync(peers, contrib[i].mv[0], src);
                __shfl_sync(peers, contrib[i].mv[1], src);
                __shfl_sync(peers, contrib[i].mv[2], src);
                __shfl_sync(peers, contrib[i].m,      src);
                remaining &= remaining - 1;
            }
        }
    }
}
