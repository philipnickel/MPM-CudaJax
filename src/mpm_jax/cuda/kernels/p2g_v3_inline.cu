// Inline P2G scatter kernel with warp-shuffle atomic coalescing used by cuda_v3.
//
// Same structure as the cuda_v1 kernel (p2g_inline.cu): one thread per particle,
// register-resident state, inline B-spline weights, 27-stencil scatter loop.
//
// Difference: before each atomicAdd into the grid, threads in the same warp
// that happen to target the SAME grid node detect each other with
// __match_any_sync, sum their contributions with __shfl_sync, and elect
// a single leader to do one atomic on behalf of the group. When particles
// are spatially sorted (Morton / Z-order) BEFORE this kernel runs, many
// warp lanes hit the same stencil cell and the number of global atomics
// drops dramatically.
//
// Helper warp_reduce_masked coalesces matching stencil targets inside a warp.
//
// Inputs (all float32):
//   x:      (N, 3)        particle positions          (assumed sorted)
//   v:      (N, 3)        particle velocities         (in same order as x)
//   C:      (N, 9)        APIC affine matrix          (in same order as x)
//   stress: (N, 9)        Kirchhoff stress            (in same order as x)
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: N, G, dt, vol, p_mass, inv_dx, dx

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

__global__ void p2g_v3_inline_kernel(
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

    // Out-of-range threads have to participate in the warp sync intrinsics
    // (otherwise __match_any_sync deadlocks). We mark them as "inactive" by
    // giving them a match key of -1 so they never match an in-range lane,
    // and we skip the atomicAdd at the bottom.
    bool active = (pid < N);
    int lane = threadIdx.x & 31;

    // Zero-init so inactive lanes carry defined state through the (unguarded)
    // base/weight computation; the actual load happens only for active lanes.
    float px[3] = {0, 0, 0}, pv[3] = {0, 0, 0}, pC[9] = {0}, pS[9] = {0};
    if (active) {
        p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);
    }

    int base[3];
    float fx[3], w[3][3], dw[3][3];
    p2g_base_fx(px, inv_dx, base, fx);
    p2g_bspline_tables(fx, w, dw);

    // v3's scatter is the experiment: on Morton-sorted particles many warp
    // lanes hit the same node, so warp-shuffle reduce same-node lanes and let
    // one leader atomic per group cut the global atomic count.
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        int gi = p2g_clip_axis(base[0] + di, G);
        int gj = p2g_clip_axis(base[1] + dj, G);
        int gk = p2g_clip_axis(base[2] + dk, G);
        int grid_idx = gi * G * G + gj * G + gk;

        // Inactive lanes get a sentinel that won't match any real index.
        int match_key = active ? grid_idx : -1;

        float mv0 = 0.0f, mv1 = 0.0f, mv2 = 0.0f, m_contrib = 0.0f;
        if (active) {
            float mv[3];
            p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                  inv_dx, dx, dt, vol, p_mass, mv, &m_contrib);
            mv0 = mv[0]; mv1 = mv[1]; mv2 = mv[2];
        }

        // Find all lanes (in this 32-lane warp) targeting the same grid node.
        // Inactive lanes use match_key = -1, so they cluster together and
        // their (zeroed) contributions don't pollute real groups.
        unsigned peers = __match_any_sync(P2G_FULL_MASK, match_key);

        mv0 = p2g_warp_reduce_masked(mv0, peers);
        mv1 = p2g_warp_reduce_masked(mv1, peers);
        mv2 = p2g_warp_reduce_masked(mv2, peers);
        m_contrib = p2g_warp_reduce_masked(m_contrib, peers);

        int leader = __ffs(peers) - 1;
        if (active && lane == leader) {
            p2g_atomic_add_grid(grid_mv, grid_m, grid_idx,
                                mv0, mv1, mv2, m_contrib);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV3InlineImpl(
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
    p2g_zero_grid(grid_mv->typed_data(), grid_m->typed_data(), G, stream);

    p2g_v3_inline_kernel<<<p2g_launch_blocks(N), P2G_BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        stress.typed_data(),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        N, G,
        dt, vol, p_mass, inv_dx, dx
    );

    return p2g_last_launch_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV3Inline, P2GV3InlineImpl,
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
