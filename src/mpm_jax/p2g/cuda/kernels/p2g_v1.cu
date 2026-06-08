// cuda_v1 P2G scatter: one thread per particle, direct global atomics.
// Stress is precomputed by JAX; this kernel only scatters the 27-node stencil
// and avoids materialising (N, 27, *) intermediates.

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

__global__ void p2g_v1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ C,
    const float* __restrict__ stress,
    float*       __restrict__ grid_mv,
    float*       __restrict__ grid_m,
    int N, int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    float px[3], pv[3], pC[9], pS[9];
    p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);

    int base[3];
    float fx[3], w[3][3], dw[3][3];
    p2g_base_fx(px, inv_dx, base, fx);
    p2g_bspline_tables(fx, w, dw);

    // Direct global atomic scatter, no warp or tile aggregation.
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        int gi = p2g_clip_axis(base[0] + di, G);
        int gj = p2g_clip_axis(base[1] + dj, G);
        int gk = p2g_clip_axis(base[2] + dk, G);
        int grid_idx = gi * G * G + gj * G + gk;

        float mv[3], m_contrib;
        p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                              inv_dx, dx, dt, vol, p_mass, mv, &m_contrib);

        p2g_atomic_add_grid(grid_mv, grid_m, grid_idx, mv, m_contrib);
    }
}

ffi::Error P2GV1Impl(
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

    p2g_v1_kernel<<<p2g_launch_blocks(N), P2G_BLOCK_SIZE, 0, stream>>>(
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
    P2GV1, P2GV1Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int32_t>("N")
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
