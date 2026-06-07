// Optimized home-cell local-reduction P2G scatter used by cuda_v4.
//
// This kernel has the same ownership model and FFI contract as cuda_v3:
// one block owns one home cell from home_cell_order. The difference is the
// local accumulation path. cuda_v3 uses shared-memory atomics into a 27-node
// tile; cuda_v4 gives each warp a private 27-node partial tile, reduces each
// stencil offset across active lanes, then combines those warp partials during
// the final global flush. That removes shared-memory atomics from the
// per-particle hot loop for the interior home-cell case.

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

namespace ffi = xla::ffi;

constexpr int P2G_V4_WARPS_PER_BLOCK = 1;
constexpr int P2G_V4_BLOCK_SIZE = P2G_V4_WARPS_PER_BLOCK * P2G_WARP_SIZE;

__device__ __forceinline__ void p2g_v4_add_warp_partial(
    float* __restrict__ warp_tile,
    int warp_id,
    int tile_node,
    float mv0,
    float mv1,
    float mv2,
    float mass
) {
    int base = (warp_id * P2G_STENCIL_NODES + tile_node) * P2G_GRID_CHANNELS;
    warp_tile[base + 0] += mv0;
    warp_tile[base + 1] += mv1;
    warp_tile[base + 2] += mv2;
    warp_tile[base + 3] += mass;
}

__global__ void p2g_v4_kernel(
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

    __shared__ float warp_tile[
        P2G_V4_WARPS_PER_BLOCK * P2G_STENCIL_NODES * P2G_GRID_CHANNELS
    ];

    for (int t = threadIdx.x;
         t < P2G_V4_WARPS_PER_BLOCK * P2G_STENCIL_NODES * P2G_GRID_CHANNELS;
         t += blockDim.x) {
        warp_tile[t] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / P2G_WARP_SIZE;
    int lane = threadIdx.x & (P2G_WARP_SIZE - 1);

    for (int local_start = warp_id * P2G_WARP_SIZE;
         local_start < n_particles;
         local_start += P2G_V4_WARPS_PER_BLOCK * P2G_WARP_SIZE) {
        int p_local = local_start + lane;
        bool active = p_local < n_particles;
        int pid = p_start + p_local;

        float px[3] = {0.0f, 0.0f, 0.0f};
        float pv[3] = {0.0f, 0.0f, 0.0f};
        float pC[9] = {0.0f};
        float pS[9] = {0.0f};
        if (active) {
            p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);
        }

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

            float mv0 = 0.0f;
            float mv1 = 0.0f;
            float mv2 = 0.0f;
            float mass = 0.0f;
            if (active) {
                float mv[3];
                p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                      inv_dx, dx, dt, vol, p_mass, mv, &mass);
                mv0 = mv[0];
                mv1 = mv[1];
                mv2 = mv[2];
            }

            int ti = gi - tile_i;
            int tj = gj - tile_j;
            int tk = gk - tile_k;
            bool local = active &&
                         ti >= 0 && ti < 3 &&
                         tj >= 0 && tj < 3 &&
                         tk >= 0 && tk < 3;
            int tile_node = ti * 9 + tj * 3 + tk;

            unsigned local_mask = __ballot_sync(P2G_FULL_MASK, local);
            if (local_mask != 0) {
                float smv0 = p2g_warp_reduce_masked(local ? mv0 : 0.0f, local_mask);
                float smv1 = p2g_warp_reduce_masked(local ? mv1 : 0.0f, local_mask);
                float smv2 = p2g_warp_reduce_masked(local ? mv2 : 0.0f, local_mask);
                float sm   = p2g_warp_reduce_masked(local ? mass : 0.0f, local_mask);
                int leader = __ffs(local_mask) - 1;
                if (lane == leader) {
                    p2g_v4_add_warp_partial(
                        warp_tile, warp_id, tile_node, smv0, smv1, smv2, sm
                    );
                }
            }

            if (active && !local) {
                int grid_idx = gi * G * G + gj * G + gk;
                p2g_atomic_add_grid(grid_mv, grid_m, grid_idx, mv0, mv1, mv2, mass);
            }
        }
    }

    __syncthreads();

    for (int t = threadIdx.x; t < P2G_STENCIL_NODES; t += blockDim.x) {
        float smv0 = 0.0f;
        float smv1 = 0.0f;
        float smv2 = 0.0f;
        float sm = 0.0f;
        for (int warp = 0; warp < P2G_V4_WARPS_PER_BLOCK; warp++) {
            int idx = (warp * P2G_STENCIL_NODES + t) * P2G_GRID_CHANNELS;
            smv0 += warp_tile[idx + 0];
            smv1 += warp_tile[idx + 1];
            smv2 += warp_tile[idx + 2];
            sm   += warp_tile[idx + 3];
        }
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

ffi::Error P2GV4Impl(
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

    p2g_v4_kernel<<<p2g_home_cell_count(G), P2G_V4_BLOCK_SIZE, 0, stream>>>(
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
    P2GV4, P2GV4Impl,
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
