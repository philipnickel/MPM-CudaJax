// cuda_v3 P2G scatter: one block per JAX-sorted super-cell. Threads aggregate
// stencil contributions into a (SC+2)^3 shared-memory tile, then flush it to
// the global grid. The kernel keeps P2G math local and avoids materialising
// (N, 27, *) intermediates.
//
// Home-cell convention matches home_super_cell_id in sort.py:
//   base = floor(x * inv_dx - 0.5), home = base + 1.

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

// One block handles one super-cell bucket. SC is a template parameter so the
// tile dimensions stay static; the FFI handler dispatches among the supported
// instantiations. Boundary stencils that spill outside the shared tile fall
// back to global atomics.
template <int SC>
__global__ void p2g_v3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ C,
    const float* __restrict__ stress,
    const int*   __restrict__ bucket_start,
    float*       __restrict__ grid_mv,
    float*       __restrict__ grid_m,
    int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    constexpr int TILE_DIM = SC + 2;       // 1-node stencil apron each side
    constexpr int TILE_SIZE = TILE_DIM * TILE_DIM * TILE_DIM;
    int Gs = G / SC;
    int Gs3 = Gs * Gs * Gs;
    int super_id = blockIdx.x;
    if (super_id >= Gs3) return;

    int p_start = bucket_start[super_id];
    int p_end   = bucket_start[super_id + 1];
    int n_particles = p_end - p_start;

    // Uniform early exit before any barrier; every thread sees the same super_id.
    if (n_particles == 0) return;

    // Decompose super_id to match the JAX-side super-cell id.
    int Si = super_id / (Gs * Gs);
    int Sj = (super_id / Gs) % Gs;
    int Sk = super_id % Gs;

    int base_ci = Si * SC;
    int base_cj = Sj * SC;
    int base_ck = Sk * SC;

    // Start one node before the super-cell so interior stencils stay tile-local.
    int tile_i = base_ci - 1;
    int tile_j = base_cj - 1;
    int tile_k = base_ck - 1;

    __shared__ float tile[TILE_SIZE * 4];

    for (int t = threadIdx.x; t < TILE_SIZE * 4; t += blockDim.x)
        tile[t] = 0.0f;
    __syncthreads();

    for (int p = threadIdx.x; p < n_particles; p += blockDim.x) {
        int pid = p_start + p;

        float px[3], pv[3], pC[9], pS[9];
        p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);

        int base[3];
        float fx[3], w[3][3], dw[3][3];
        p2g_base_fx(px, inv_dx, base, fx);
        p2g_bspline_tables(fx, w, dw);

        for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
        for (int dk = 0; dk < 3; dk++) {
            // Per-axis clamp intentionally differs from JAX's flat-index clip
            // at boundaries.
            int gi = p2g_clip_axis(base[0] + di, G);
            int gj = p2g_clip_axis(base[1] + dj, G);
            int gk = p2g_clip_axis(base[2] + dk, G);

            float mv[3], m_contrib;
            p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                  inv_dx, dx, dt, vol, p_mass, mv, &m_contrib);

            // Use clipped global indices for tile-local lookup, matching the
            // global fallback path.
            int ti = gi - tile_i;
            int tj = gj - tile_j;
            int tk = gk - tile_k;
            if (ti >= 0 && ti < TILE_DIM &&
                tj >= 0 && tj < TILE_DIM &&
                tk >= 0 && tk < TILE_DIM) {
                int tile_idx = (ti * TILE_DIM * TILE_DIM + tj * TILE_DIM + tk) * 4;
                atomicAdd(&tile[tile_idx + 0], mv[0]);
                atomicAdd(&tile[tile_idx + 1], mv[1]);
                atomicAdd(&tile[tile_idx + 2], mv[2]);
                atomicAdd(&tile[tile_idx + 3], m_contrib);
            } else {
                int grid_idx = gi * G * G + gj * G + gk;
                p2g_atomic_add_grid(grid_mv, grid_m, grid_idx, mv, m_contrib);
            }
        }
    }

    __syncthreads();

    // Flush the shared tile to global memory.
    for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
        float smv0 = tile[t * 4 + 0];
        float smv1 = tile[t * 4 + 1];
        float smv2 = tile[t * 4 + 2];
        float sm   = tile[t * 4 + 3];

        if (sm == 0.0f && smv0 == 0.0f && smv1 == 0.0f && smv2 == 0.0f)
            continue;

        int ti = t / (TILE_DIM * TILE_DIM);
        int tj = (t / TILE_DIM) % TILE_DIM;
        int tk = t % TILE_DIM;
        int gi = tile_i + ti;
        int gj = tile_j + tj;
        int gk = tile_k + tk;

        if (gi < 0 || gi >= G || gj < 0 || gj >= G || gk < 0 || gk >= G)
            continue;

        int gid = gi * G * G + gj * G + gk;
        p2g_atomic_add_grid(grid_mv, grid_m, gid, smv0, smv1, smv2, sm);
    }
}

ffi::Error P2GV3Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::Buffer<ffi::S32> bucket_start,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    int32_t G,
    int32_t SC,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    if (SC != 2 && SC != 4 && SC != 8) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "unsupported SC; kernel is built for SC in {2, 4, 8}");
    }
    if (G % SC != 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "G must be divisible by SC (super-cell width)");
    }
    int Gs = G / SC;
    int Gs3 = Gs * Gs * Gs;
    int expected = Gs3 + 1;
    int got = static_cast<int>(bucket_start.dimensions()[0]);
    if (got != expected) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "bucket_start size does not match (G/SC)^3 + 1");
    }

    p2g_zero_grid(grid_mv->typed_data(), grid_m->typed_data(), G, stream);

    // Keep SC cases in sync with SUPPORTED_SC in p2g_cuda.py.
#define MPM_LAUNCH_V3(SCVAL)                                                  \
    p2g_v3_kernel<SCVAL><<<Gs3, P2G_BLOCK_SIZE, 0, stream>>>(          \
        x.typed_data(), v.typed_data(), C.typed_data(), stress.typed_data(), \
        reinterpret_cast<const int*>(bucket_start.typed_data()),             \
        grid_mv->typed_data(), grid_m->typed_data(),                         \
        G, dt, vol, p_mass, inv_dx, dx)
    switch (SC) {
        case 2: MPM_LAUNCH_V3(2); break;
        case 4: MPM_LAUNCH_V3(4); break;
        case 8: MPM_LAUNCH_V3(8); break;
        default: break;
    }
#undef MPM_LAUNCH_V3

    return p2g_last_launch_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV3, P2GV3Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int32_t>("G")
        .Attr<int32_t>("SC")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
