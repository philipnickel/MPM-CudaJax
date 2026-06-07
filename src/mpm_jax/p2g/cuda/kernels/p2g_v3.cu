// Home-cell local-reduction P2G scatter used by the cuda_v3 backend.
//
// Particles are expected to be sorted by home_cell_order on the JAX side. One
// CUDA block owns one home cell, loops over that cell's sorted particle range,
// accumulates the 27-node stencil into a small shared-memory tile, then flushes
// the reduced stencil to the global grid.
//
// Inputs (all float32 unless noted):
//   x:             (N, 3) sorted by home cell
//   v:             (N, 3) sorted by home cell
//   C:             (N, 9) sorted by home cell, row-major
//   stress:        (N, 9) sorted by home cell, row-major
//   bucket_bounds: (G+1, G+1, G+1, 2) int32 start/end into sorted arrays
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

__global__ void p2g_v3_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ C,
    const float* __restrict__ stress,
    const int*   __restrict__ bucket_bounds,
    float*       __restrict__ grid_mv,
    float*       __restrict__ grid_m,
    int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int cells_per_axis = G + 1;
    int cell_id = blockIdx.x;
    int cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    if (cell_id >= cell_count) return;

    int hi = cell_id / (cells_per_axis * cells_per_axis);
    int hj = (cell_id / cells_per_axis) % cells_per_axis;
    int hk = cell_id % cells_per_axis;

    int bounds_idx = cell_id * 2;
    int p_start = bucket_bounds[bounds_idx + 0];
    int p_end = bucket_bounds[bounds_idx + 1];
    int n_particles = p_end - p_start;
    if (n_particles == 0) return;

    int tile_i = hi - 1;
    int tile_j = hj - 1;
    int tile_k = hk - 1;

    __shared__ float tile[P2G_STENCIL_NODES * P2G_GRID_CHANNELS];

    for (int t = threadIdx.x; t < P2G_STENCIL_NODES * P2G_GRID_CHANNELS; t += blockDim.x) {
        tile[t] = 0.0f;
    }
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
            int gi = p2g_clip_axis(base[0] + di, G);
            int gj = p2g_clip_axis(base[1] + dj, G);
            int gk = p2g_clip_axis(base[2] + dk, G);

            float mv[3], m_contrib;
            p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                  inv_dx, dx, dt, vol, p_mass, mv, &m_contrib);

            int ti = gi - tile_i;
            int tj = gj - tile_j;
            int tk = gk - tile_k;
            if (ti >= 0 && ti < 3 && tj >= 0 && tj < 3 && tk >= 0 && tk < 3) {
                int tile_idx = (ti * 9 + tj * 3 + tk) * P2G_GRID_CHANNELS;
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

    for (int t = threadIdx.x; t < P2G_STENCIL_NODES; t += blockDim.x) {
        float smv0 = tile[t * P2G_GRID_CHANNELS + 0];
        float smv1 = tile[t * P2G_GRID_CHANNELS + 1];
        float smv2 = tile[t * P2G_GRID_CHANNELS + 2];
        float sm   = tile[t * P2G_GRID_CHANNELS + 3];
        if (sm == 0.0f && smv0 == 0.0f && smv1 == 0.0f && smv2 == 0.0f) {
            continue;
        }

        int ti = t / 9;
        int tj = (t / 3) % 3;
        int tk = t % 3;
        int gi = tile_i + ti;
        int gj = tile_j + tj;
        int gk = tile_k + tk;
        if (gi < 0 || gi >= G || gj < 0 || gj >= G || gk < 0 || gk >= G) {
            continue;
        }

        int grid_idx = gi * G * G + gj * G + gk;
        p2g_atomic_add_grid(grid_mv, grid_m, grid_idx, smv0, smv1, smv2, sm);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV3Impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::Buffer<ffi::S32> bucket_bounds,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    int32_t G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int cells_per_axis = G + 1;
    auto dims = bucket_bounds.dimensions();
    if (dims.size() != 4 ||
        dims[0] != cells_per_axis ||
        dims[1] != cells_per_axis ||
        dims[2] != cells_per_axis ||
        dims[3] != 2) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "bucket_bounds must have shape (G+1, G+1, G+1, 2)");
    }

    p2g_zero_grid(grid_mv->typed_data(), grid_m->typed_data(), G, stream);

    p2g_v3_kernel<<<p2g_home_cell_count(G), P2G_WARP_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        stress.typed_data(),
        reinterpret_cast<const int*>(bucket_bounds.typed_data()),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        G,
        dt, vol, p_mass, inv_dx, dx
    );

    return p2g_last_launch_error();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV3, P2GV3Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x (home-cell sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // v (home-cell sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // C (home-cell sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // stress (home-cell sorted)
        .Arg<ffi::Buffer<ffi::S32>>()   // bucket_bounds
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
