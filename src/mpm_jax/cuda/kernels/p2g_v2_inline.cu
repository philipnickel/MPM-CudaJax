// Inline P2G scatter kernel with warp-shuffle reduction used by the cuda_v2 backend.
//
// Identical to the cuda_v1 backend kernel (p2g_inline.cu) for the per-particle setup —
// one thread per particle, register-resident state, inline B-spline weights,
// no (N, 27, *) materialisation in HBM. The ONLY difference is the 27-stencil
// scatter: before each atomicAdd, lanes inside the warp that target the same
// grid_idx reduce their contributions via __match_any_sync + __shfl_sync,
// and only the leader lane issues the atomic.
//
// Question this kernel answers: now that the (N, 27, *) materialisation is
// gone (cuda_v1 structure), does warp-shuffle reduction on top of
// atomicAdd help, hurt, or wash?
//
// Requires sm_70+ for __match_any_sync (the A10 / sm_86 is fine).
//
// Inputs/outputs identical to p2g_inline.cu — see that file for layout.

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

namespace ffi = xla::ffi;

__global__ void p2g_v2_inline_kernel(
    const float* __restrict__ x,        // (N, 3)
    const float* __restrict__ v,        // (N, 3)
    const float* __restrict__ C,        // (N, 9) row-major
    const float* __restrict__ stress,   // (N, 9) row-major
    float*       __restrict__ grid_mv,  // (G^3, 3)
    float*       __restrict__ grid_m,   // (G^3,)
    int N, int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    // Out-of-range threads must still participate in __match_any_sync (they
    // contribute a sentinel grid_idx of -1, so they form their own group and
    // skip the atomics in the leader check). Otherwise we deadlock the warp.
    bool active = pid < N;

    // Per-particle state + B-spline tables (only meaningful for active lanes;
    // inactive lanes never read them — they carry grid_idx = -1 below).
    float px[3], pv[3], pC[9], pS[9];
    int base[3];
    float fx[3], w[3][3], dw[3][3];
    if (active) {
        p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);
        p2g_base_fx(px, inv_dx, base, fx);
        p2g_bspline_tables(fx, w, dw);
    }

    // Scatter to 27 stencil nodes. All warp lanes must execute the loop
    // (they all participate in __match_any_sync); inactive lanes carry
    // grid_idx = -1 so they form their own match group and the atomic guard
    // (active && lane == leader) keeps them from writing. v2's scatter is the
    // experiment: warp-shuffle reduce same-node lanes, one leader atomic each.
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        float mv0 = 0.0f, mv1 = 0.0f, mv2 = 0.0f, mass_contrib = 0.0f;
        int grid_idx = -1;
        if (active) {
            int gi = p2g_clip_axis(base[0] + di, G);
            int gj = p2g_clip_axis(base[1] + dj, G);
            int gk = p2g_clip_axis(base[2] + dk, G);
            grid_idx = gi * G * G + gj * G + gk;

            float mv[3];
            p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                  inv_dx, dx, dt, vol, p_mass, mv, &mass_contrib);
            mv0 = mv[0]; mv1 = mv[1]; mv2 = mv[2];
        }

        // Warp coalescing: find lanes in this warp targeting the same grid_idx.
        // Inactive lanes (grid_idx = -1) form their own group and don't write.
        unsigned peers = __match_any_sync(FULL_MASK, grid_idx);

        // Sum contributions across matching lanes.
        mv0 = p2g_warp_reduce_masked(mv0, peers);
        mv1 = p2g_warp_reduce_masked(mv1, peers);
        mv2 = p2g_warp_reduce_masked(mv2, peers);
        mass_contrib = p2g_warp_reduce_masked(mass_contrib, peers);

        // Only the leader (lowest lane in group) does the atomic.
        int leader = __ffs(peers) - 1;  // __ffs returns 1-indexed
        if (active && lane == leader) {
            atomicAdd(&grid_mv[grid_idx * 3 + 0], mv0);
            atomicAdd(&grid_mv[grid_idx * 3 + 1], mv1);
            atomicAdd(&grid_mv[grid_idx * 3 + 2], mv2);
            atomicAdd(&grid_m[grid_idx],          mass_contrib);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV2InlineImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    int32_t N,
    int32_t G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int grid_mv_size = G * G * G * 3;
    int grid_m_size = G * G * G;

    // Zero the grid (the kernel only adds into it). f32 +0.0 is all-zero bits,
    // so a byte memset is exactly correct and uses the driver's tuned fill path.
    cudaMemsetAsync(grid_mv->typed_data(), 0,
                    (size_t)grid_mv_size * sizeof(float), stream);
    cudaMemsetAsync(grid_m->typed_data(), 0,
                    (size_t)grid_m_size * sizeof(float), stream);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    p2g_v2_inline_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        stress.typed_data(),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        N, G,
        dt, vol, p_mass, inv_dx, dx
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV2Inline, P2GV2InlineImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x
        .Arg<ffi::Buffer<ffi::F32>>()   // v
        .Arg<ffi::Buffer<ffi::F32>>()   // C
        .Arg<ffi::Buffer<ffi::F32>>()   // stress
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Attr<int32_t>("N")
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
