// Inline P2G scatter kernel used by the cuda_v1 backend.
//
// One thread per particle. Each thread:
//   1. Loads x, v, C, stress into registers (stress is precomputed by JAX).
//   2. Computes B-spline weights for its particle position.
//   3. Loops over 27 stencil nodes:
//        - computes per-stencil weight, dweight, dpos, grid index
//        - computes momentum contribution mv = -dt*vol*stress@dweight
//                                              + p_mass*weight*(v + C@dpos)
//        - atomicAdds mv and mass into grid buffers
//
// No SVD, plasticity, or stress formula lives in this kernel; JAX computes
// stress upstream and the CUDA kernel only scatters it. This keeps the kernel
// model-agnostic and avoids materialising (N, 27, *) intermediates.
//
// Inputs (all float32):
//   x:      (N, 3)        particle positions
//   v:      (N, 3)        particle velocities
//   C:      (N, 9)        APIC affine matrix (row-major)
//   stress: (N, 9)        Kirchhoff stress, precomputed by JAX (row-major)
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: N, G, dt, vol, p_mass, inv_dx, dx

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

__global__ void p2g_inline_kernel(
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
    if (pid >= N) return;

    // Load this particle's state into registers — read ONCE, reused 27x.
    float px[3], pv[3], pC[9], pS[9];
    p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);

    // Quadratic B-spline base node, fractional offset, and weight tables.
    int base[3];
    float fx[3], w[3][3], dw[3][3];
    p2g_base_fx(px, inv_dx, base, fx);
    p2g_bspline_tables(fx, w, dw);

    // Scatter to 27 stencil nodes — register-resident loop, no (N, 27, *)
    // intermediate ever exists in HBM. v1's scatter is the naive one: a direct
    // global atomicAdd per node, no coalescing.
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

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GInlineImpl(
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

    p2g_inline_kernel<<<p2g_launch_blocks(N), P2G_BLOCK_SIZE, 0, stream>>>(
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
    P2GInline, P2GInlineImpl,
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
