// p2g_scatter_warp.cu — Fused P2G scatter with warp-level reduction.
//
// Same compute as naive, but uses __match_any_sync to find warp lanes
// targeting the same grid node, then reduces via sequential __shfl_sync.
// Only the leader lane does the atomicAdd — fewer global atomics.
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
    int lane = threadIdx.x & 31;
    bool active = (pid < n_particles);

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
        for (int i = 0; i < STENCIL; i++) {
            contrib[i].mv[0] = contrib[i].mv[1] = contrib[i].mv[2] = 0.0f;
            contrib[i].m = 0.0f;
            contrib[i].grid_idx = -1;
        }
    }

    for (int i = 0; i < STENCIL; i++) {
        int gid = contrib[i].grid_idx;
        unsigned peers = __match_any_sync(FULL_MASK, gid);
        int leader = __ffs(peers) - 1;

        // Sequential reduction: leader sums values from all peers
        // All peers participate in __shfl_sync (required for correctness)
        float mv0 = contrib[i].mv[0];
        float mv1 = contrib[i].mv[1];
        float mv2 = contrib[i].mv[2];
        float m   = contrib[i].m;

        if (lane == leader) {
            unsigned others = peers & ~(1u << leader);  // exclude self
            while (others) {
                int src = __ffs(others) - 1;
                mv0 += __shfl_sync(peers, contrib[i].mv[0], src);
                mv1 += __shfl_sync(peers, contrib[i].mv[1], src);
                mv2 += __shfl_sync(peers, contrib[i].mv[2], src);
                m   += __shfl_sync(peers, contrib[i].m,      src);
                others &= others - 1;
            }
            if (gid >= 0) {
                atomicAdd(&grid_mv[gid*3+0], mv0);
                atomicAdd(&grid_mv[gid*3+1], mv1);
                atomicAdd(&grid_mv[gid*3+2], mv2);
                atomicAdd(&grid_m[gid], m);
            }
        } else {
            // Non-leader peers: participate in __shfl_sync calls
            // The leader reads from us via __shfl_sync(peers, ..., src)
            // We just need to be present at each __shfl_sync call
            unsigned others = peers & ~(1u << leader);
            while (others) {
                __shfl_sync(peers, contrib[i].mv[0], __ffs(others) - 1);
                __shfl_sync(peers, contrib[i].mv[1], __ffs(others) - 1);
                __shfl_sync(peers, contrib[i].mv[2], __ffs(others) - 1);
                __shfl_sync(peers, contrib[i].m,      __ffs(others) - 1);
                others &= others - 1;
            }
        }
    }
}
