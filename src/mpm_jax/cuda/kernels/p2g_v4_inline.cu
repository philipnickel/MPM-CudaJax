// Cell-major inline P2G scatter kernel (cuda_v4_inline).
//
// Combines two ideas:
//   * `p2g_inline.cu` (cuda_v1_inline): one thread per particle, inline
//     B-spline weights + 27-stencil scatter computed in registers. No
//     (N, 27, *) tensor is ever materialised in HBM.
//   * Particles are sorted by their home SUPER-cell on the JAX side; one
//     CUDA block per super-cell aggregates
//     its particles' contributions into a 6x6x6 shared-memory tile via fast
//     shmem atomics, then flushes the tile to global memory with one
//     atomicAdd per node.
//
// Super-cells (Approach B from the v4 fix plan):
//   A super-cell of size SC^3 grid cells. One CUDA block per super-cell.
//   SC=4 gives 4^3 cells per CUDA block. At the standard 8 particles/cell
//   benchmark this is roughly 512 particles per non-empty block, which is
//   enough work to amortize tile zero/sync/flush costs.
//
// With SC=4:
//   Each super-cell covers 64 cells. The union of quadratic 3^3 stencils spans
//   (SC + 2)^3 = 6^3 = 216 grid nodes. The larger shared tile increases the
//   flush cost, but reduces the block count by another 8x versus SC=2.
//
// The hypothesis is that the smem aggregation amortises the 27 global
// atomicAdds per particle (108 floats) down to the per-super-cell tile flush
// (~TILE_SIZE = (SC+2)^3 = 216 occupied grid nodes for SC=4). With inline
// weight computation, the (N, 27, *) momentum/mass/index tensors disappear
// from HBM and only x/v/C/stress are loaded once per particle.
//
// Inputs (all float32 unless noted):
//   x:          (N, 3)        particle positions (SORTED by home super-cell)
//   v:          (N, 3)        particle velocities
//   C:          (N, 9)        APIC affine matrix (row-major)
//   stress:     (N, 9)        Kirchhoff stress (row-major)
//   cell_start: ((G/SC)^3 + 1,) int32  CSR boundaries into the sorted arrays
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: dt, vol, p_mass, inv_dx, dx
//
// Home cell convention (matches `home_super_cell_id` in blocks/sort.py):
//   For a particle at x, the stencil base node is
//     base = floor(x * inv_dx - 0.5)
//   and the center stencil node (offset (1,1,1)) is
//     home = base + 1
//   home in [0, G). Super-cell id is (home / SC) collapsed to flat.

#include "xla/ffi/api/ffi.h"

#include "p2g_common.cuh"

#define BLOCK_SIZE 256  // threads per super-cell block

// Super-cell width SC is a compile-time TEMPLATE parameter: TILE_DIM/TILE_SIZE
// derive from it and statically size the __shared__ tile, so each instantiation
// stays fully optimised. The FFI handler dispatches to the instantiated SC by a
// runtime switch (kept in sync with SUPPORTED_SC in p2g_cuda.py), so SC is
// selectable from config without a recompile and without dynamic shared memory.

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Super-cell-tiled inline kernel
// ---------------------------------------------------------------------------
// One block per super-cell. Each thread in the block processes one or more
// particles from that super-cell (grid-stride loop over the in-super-cell
// particle range). Per particle:
//   - load x, v, C, stress
//   - compute base node + per-axis B-spline weight/dweight tables
//   - loop over 27 stencil offsets, atomicAdd momentum + mass into the
//     shared-memory tile (or fall back to global atomicAdd when the stencil
//     clips outside the tile — only happens at grid boundary).
// After all threads finish, the block flushes the TILE_SIZE = (SC+2)^3 = 216
// tile entries to global memory with one atomicAdd per (mv, m).

template <int SC>
__global__ void p2g_v4_inline_kernel(
    const float* __restrict__ x,          // (N, 3) sorted by home super-cell
    const float* __restrict__ v,          // (N, 3) sorted
    const float* __restrict__ C,          // (N, 9) sorted, row-major
    const float* __restrict__ stress,     // (N, 9) sorted, row-major
    const int*   __restrict__ cell_start, // ((G/SC)^3 + 1,)
    float*       __restrict__ grid_mv,    // (G^3, 3)
    float*       __restrict__ grid_m,     // (G^3,)
    int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    constexpr int TILE_DIM = SC + 2;       // stencil-union: 1-node apron each side
    constexpr int TILE_SIZE = TILE_DIM * TILE_DIM * TILE_DIM;
    int Gs = G / SC;                       // super-grid resolution
    int Gs3 = Gs * Gs * Gs;
    int super_id = blockIdx.x;
    if (super_id >= Gs3) return;

    int p_start = cell_start[super_id];
    int p_end   = cell_start[super_id + 1];
    int n_particles = p_end - p_start;

    // Fast empty-super-cell exit. The particle block only occupies a fraction
    // of the domain, so most super-cells are empty. The exit must come
    // *before* any shared-memory allocation or __syncthreads() — all
    // threads in a block see the same super_id, so this return is
    // uniform and no thread is left hanging on a barrier.
    if (n_particles == 0) return;

    // Super-cell 3D coords from flat index (matches the JAX-side super-cell id).
    int Si = super_id / (Gs * Gs);
    int Sj = (super_id / Gs) % Gs;
    int Sk = super_id % Gs;

    // Base cell of the super-cell (in cells, not super-cells).
    int base_ci = Si * SC;
    int base_cj = Sj * SC;
    int base_ck = Sk * SC;

    // Tile origin: (base_cell - 1) so that the union of 3^3 stencils for
    // particles in any of this super-cell's SC^3 cells lands at tile-local
    // indices 0..TILE_DIM-1. For SC=4 this gives tile_dim=6, spanning
    // cells (base_cell - 1) .. (base_cell + 4).
    int tile_i = base_ci - 1;
    int tile_j = base_cj - 1;
    int tile_k = base_ck - 1;

    // Shared-memory tile: 216 nodes, each (mv_x, mv_y, mv_z, mass).
    // 864 floats = 3.4 KB.
    __shared__ float tile[TILE_SIZE * 4];

    // Cooperatively zero the tile.
    for (int t = threadIdx.x; t < TILE_SIZE * 4; t += blockDim.x)
        tile[t] = 0.0f;
    __syncthreads();

    // ---- Per-particle inline scatter into the tile ----
    for (int p = threadIdx.x; p < n_particles; p += blockDim.x) {
        int pid = p_start + p;

        // Register-resident particle state.
        float px[3], pv[3], pC[9], pS[9];
        p2g_load_particle(x, v, C, stress, pid, px, pv, pC, pS);

        // Base node, fractional offset, and B-spline weight tables.
        int base[3];
        float fx[3], w[3][3], dw[3][3];
        p2g_base_fx(px, inv_dx, base, fx);
        p2g_bspline_tables(fx, w, dw);

        // 27-stencil register-resident loop.
        for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
        for (int dk = 0; dk < 3; dk++) {
            // Clamped global grid index (per-axis clamp to [0, G-1]; see
            // p2g_common.cuh — intentionally differs from the JAX flat clip).
            int gi = p2g_clip_axis(base[0] + di, G);
            int gj = p2g_clip_axis(base[1] + dj, G);
            int gk = p2g_clip_axis(base[2] + dk, G);

            float mv[3], m_contrib;
            p2g_node_contribution(di, dj, dk, w, dw, fx, pC, pS, pv,
                                  inv_dx, dx, dt, vol, p_mass, mv, &m_contrib);

            // Try the tile first. Tile-local index uses the (possibly
            // clipped) global index, so a clipped stencil that lands back
            // inside the tile is still tile-local and a stencil that lands
            // outside the tile (or outside the grid) goes via global atomic.
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
                // Out-of-tile fallback: direct global atomic. In practice this
                // only triggers near the grid boundary (cells where the
                // stencil clips). The body of the simulation never falls here.
                int grid_idx = gi * G * G + gj * G + gk;
                atomicAdd(&grid_mv[grid_idx * 3 + 0], mv[0]);
                atomicAdd(&grid_mv[grid_idx * 3 + 1], mv[1]);
                atomicAdd(&grid_mv[grid_idx * 3 + 2], mv[2]);
                atomicAdd(&grid_m[grid_idx], m_contrib);
            }
        }
    }

    __syncthreads();

    // ---- Flush the smem tile to global memory ----
    for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
        float smv0 = tile[t * 4 + 0];
        float smv1 = tile[t * 4 + 1];
        float smv2 = tile[t * 4 + 2];
        float sm   = tile[t * 4 + 3];

        // Skip entries no particle touched.
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
        atomicAdd(&grid_mv[gid * 3 + 0], smv0);
        atomicAdd(&grid_mv[gid * 3 + 1], smv1);
        atomicAdd(&grid_mv[gid * 3 + 2], smv2);
        atomicAdd(&grid_m[gid],          sm);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV4InlineImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::Buffer<ffi::S32> cell_start,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    int32_t G,
    int32_t SC,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    if (G % SC != 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "G must be divisible by SC (super-cell width)");
    }
    // cell_start is ((G/SC)^3 + 1,) ints.
    int Gs = G / SC;
    int Gs3 = Gs * Gs * Gs;
    int expected = Gs3 + 1;
    int got = static_cast<int>(cell_start.dimensions()[0]);
    if (got != expected) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cell_start size does not match (G/SC)^3 + 1");
    }

    int G3 = G * G * G;
    int grid_mv_size = G3 * 3;
    int grid_m_size = G3;

    // Zero the grid (the kernel only adds into it). f32 +0.0 is all-zero bits,
    // so a byte memset is exactly correct and uses the driver's tuned fill path.
    cudaMemsetAsync(grid_mv->typed_data(), 0,
                    (size_t)grid_mv_size * sizeof(float), stream);
    cudaMemsetAsync(grid_m->typed_data(), 0,
                    (size_t)grid_m_size * sizeof(float), stream);

    // One block per super-cell. SC selects the template instantiation (each has
    // its own statically-sized shared tile); keep these cases in sync with
    // SUPPORTED_SC in p2g_cuda.py.
#define MPM_LAUNCH_V4(SCVAL)                                                  \
    p2g_v4_inline_kernel<SCVAL><<<Gs3, BLOCK_SIZE, 0, stream>>>(              \
        x.typed_data(), v.typed_data(), C.typed_data(), stress.typed_data(), \
        reinterpret_cast<const int*>(cell_start.typed_data()),               \
        grid_mv->typed_data(), grid_m->typed_data(),                         \
        G, dt, vol, p_mass, inv_dx, dx)
    switch (SC) {
        case 2: MPM_LAUNCH_V4(2); break;
        case 4: MPM_LAUNCH_V4(4); break;
        case 8: MPM_LAUNCH_V4(8); break;
        default:
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "unsupported SC; kernel is built for SC in {2, 4, 8}");
    }
#undef MPM_LAUNCH_V4

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV4Inline, P2GV4InlineImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // v (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // C (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // stress (sorted)
        .Arg<ffi::Buffer<ffi::S32>>()   // cell_start over super-cells
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Attr<int32_t>("G")
        .Attr<int32_t>("SC")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
