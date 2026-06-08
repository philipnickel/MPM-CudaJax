// Shared inline helpers for CUDA P2G variants.
#pragma once

#include <cstddef>

#include "cuda_runtime_api.h"
#include "xla/ffi/api/ffi.h"

constexpr int P2G_BLOCK_SIZE = 256;
constexpr unsigned P2G_FULL_MASK = 0xFFFFFFFFu;

inline int p2g_grid_nodes(int G) {
    return G * G * G;
}

inline int p2g_launch_blocks(int n_particles) {
    return (n_particles + P2G_BLOCK_SIZE - 1) / P2G_BLOCK_SIZE;
}

inline void p2g_zero_grid(
    float* grid_mv,
    float* grid_m,
    int G,
    cudaStream_t stream
) {
    int nodes = p2g_grid_nodes(G);
    // f32 +0.0 is all-zero bits.
    cudaMemsetAsync(grid_mv, 0, static_cast<size_t>(nodes) * 3 * sizeof(float), stream);
    cudaMemsetAsync(grid_m, 0, static_cast<size_t>(nodes) * sizeof(float), stream);
}

inline xla::ffi::Error p2g_last_launch_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            cudaGetErrorString(err)
        );
    }
    return xla::ffi::Error::Success();
}

__device__ __forceinline__ void p2g_atomic_add_grid(
    float* __restrict__ grid_mv,
    float* __restrict__ grid_m,
    int grid_idx,
    float mv0,
    float mv1,
    float mv2,
    float m_contrib
) {
    atomicAdd(&grid_mv[grid_idx * 3 + 0], mv0);
    atomicAdd(&grid_mv[grid_idx * 3 + 1], mv1);
    atomicAdd(&grid_mv[grid_idx * 3 + 2], mv2);
    atomicAdd(&grid_m[grid_idx],          m_contrib);
}

__device__ __forceinline__ void p2g_atomic_add_grid(
    float* __restrict__ grid_mv,
    float* __restrict__ grid_m,
    int grid_idx,
    const float mv[3],
    float m_contrib
) {
    p2g_atomic_add_grid(grid_mv, grid_m, grid_idx, mv[0], mv[1], mv[2], m_contrib);
}

__device__ __forceinline__ void p2g_load_particle(
    const float* __restrict__ x, const float* __restrict__ v,
    const float* __restrict__ C, const float* __restrict__ stress,
    int pid, float px[3], float pv[3], float pC[9], float pS[9]) {
    for (int d = 0; d < 3; d++) {
        px[d] = x[pid * 3 + d];
        pv[d] = v[pid * 3 + d];
    }
    for (int i = 0; i < 9; i++) {
        pC[i] = C[pid * 9 + i];
        pS[i] = stress[pid * 9 + i];
    }
}

// Base node + fractional offset; must match home_cell_id in sort.py.
__device__ __forceinline__ void p2g_base_fx(
    const float px[3], float inv_dx, int base[3], float fx[3]) {
    for (int d = 0; d < 3; d++) {
        float fpx = px[d] * inv_dx;
        base[d] = (int)floorf(fpx - 0.5f);
        fx[d] = fpx - (float)base[d];
    }
}

__device__ __forceinline__ void p2g_bspline_tables(
    const float fx[3], float w[3][3], float dw[3][3]) {
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
        dw[d][0] = fx[d] - 1.5f;
        dw[d][1] = -2.0f * (fx[d] - 1.0f);
        dw[d][2] = fx[d] - 0.5f;
    }
}

// Per-axis clamp.
__device__ __forceinline__ int p2g_clip_axis(int idx, int G) {
    return max(0, min(idx, G - 1));
}

// MLS-MPM per-node contribution.
__device__ __forceinline__ void p2g_node_contribution(
    int di, int dj, int dk,
    const float w[3][3], const float dw[3][3], const float fx[3],
    const float pC[9], const float pS[9], const float pv[3],
    float inv_dx, float dx, float dt, float vol, float p_mass,
    float mv[3], float* m_contrib) {
    float weight = w[0][di] * w[1][dj] * w[2][dk];

    // weight gradient, scaled to physical units.
    float dweight[3];
    dweight[0] = inv_dx * dw[0][di] * w[1][dj]  * w[2][dk];
    dweight[1] = inv_dx * w[0][di]  * dw[1][dj] * w[2][dk];
    dweight[2] = inv_dx * w[0][di]  * w[1][dj]  * dw[2][dk];

    // dpos = (offset - fx) * dx.
    float dpos[3];
    dpos[0] = ((float)di - fx[0]) * dx;
    dpos[1] = ((float)dj - fx[1]) * dx;
    dpos[2] = ((float)dk - fx[2]) * dx;

    for (int d = 0; d < 3; d++) {
        float s_dw = 0.0f;
        float c_dp = 0.0f;
        for (int j = 0; j < 3; j++) {
            s_dw += pS[d * 3 + j] * dweight[j];
            c_dp += pC[d * 3 + j] * dpos[j];
        }
        mv[d] = -dt * vol * s_dw + p_mass * weight * (pv[d] + c_dp);
    }
    *m_contrib = weight * p_mass;
}

// Sum val across lanes in an arbitrary peer mask.
__device__ __forceinline__ float p2g_warp_reduce_masked(float val, unsigned mask) {
    float sum = 0.0f;
    for (unsigned remaining = mask; remaining; remaining &= (remaining - 1)) {
        int src_lane = __ffs(remaining) - 1;
        sum += __shfl_sync(mask, val, src_lane);
    }
    return sum;
}
