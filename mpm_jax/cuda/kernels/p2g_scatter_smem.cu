// Shared-memory P2G scatter kernel (v4).
//
// Follows Gao et al. (SIGGRAPH Asia 2018) "GPU Optimization of MPM":
//   1. Particles are pre-sorted by cell (done in Python/JAX)
//   2. One CUDA block per grid cell
//   3. Each block has a 4x4x4 shared memory tile covering the cell's
//      stencil neighborhood (offset by -1 in each dimension)
//   4. Particles scatter 27 contributions to shared memory (fast atomics)
//   5. Block flushes 64 tile entries to global memory (coalesced atomics)
//
// This reduces global atomics from 4*27*N (naive) to 4*64*num_occupied_cells,
// and shared memory atomics are ~10x faster than global on modern GPUs.
//
// Inputs:
//   mv_sorted:  (N, 27, 3) momentum contributions (sorted by cell)
//   m_sorted:   (N, 27)    mass contributions (sorted by cell)
//   index_sorted: (N, 27)  flat grid indices (sorted by cell)
//   cell_start: (G^3 + 1,) start index of each cell in sorted particle array
//
// Outputs:
//   grid_mv: (G^3, 3) grid momentum
//   grid_m:  (G^3,)   grid mass

#include "xla/ffi/api/ffi.h"

#define TILE_DIM 4
#define TILE_SIZE (TILE_DIM * TILE_DIM * TILE_DIM)  // 64 nodes
#define TILE_FLOATS (TILE_SIZE * 4)  // 64 * (3 momentum + 1 mass) = 256
#define STENCIL 27
#define BLOCK_SIZE 128  // threads per block — handles up to 128 particles per cell

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

__global__ void p2g_scatter_smem_kernel(
    const float* __restrict__ mv,         // (N, 27, 3) sorted
    const float* __restrict__ m,          // (N, 27) sorted
    const int*   __restrict__ index,      // (N, 27) sorted
    const int*   __restrict__ cell_start, // (G^3 + 1,)
    float*       __restrict__ grid_mv,    // (G^3, 3)
    float*       __restrict__ grid_m,     // (G^3,)
    int G
) {
    // One block per grid cell
    int cell_id = blockIdx.x;
    if (cell_id >= G * G * G) return;

    // How many particles in this cell?
    int p_start = cell_start[cell_id];
    int p_end = cell_start[cell_id + 1];
    int n_particles = p_end - p_start;
    if (n_particles == 0) return;

    // Cell 3D coordinates
    int ci = cell_id / (G * G);
    int cj = (cell_id / G) % G;
    int ck = cell_id % G;

    // Tile origin: offset by -1 so the 3x3x3 stencil of any particle
    // in this cell fits within the 4x4x4 tile
    int tile_i = ci - 1;
    int tile_j = cj - 1;
    int tile_k = ck - 1;

    // Shared memory tile: 4x4x4 nodes, each storing (mv_x, mv_y, mv_z, mass)
    __shared__ float tile[TILE_FLOATS];  // 256 floats = 1KB

    // Zero shared memory cooperatively
    for (int t = threadIdx.x; t < TILE_FLOATS; t += blockDim.x)
        tile[t] = 0.0f;
    __syncthreads();

    // Each thread processes one or more particles
    for (int p = threadIdx.x; p < n_particles; p += blockDim.x) {
        int pid = p_start + p;

        // Scatter 27 stencil contributions to shared memory
        for (int s = 0; s < STENCIL; s++) {
            int offset = pid * STENCIL + s;
            int gid = index[offset];

            // Convert flat grid index to 3D
            int gi = gid / (G * G);
            int gj = (gid / G) % G;
            int gk = gid % G;

            // Tile-relative coordinates
            int ti = gi - tile_i;
            int tj = gj - tile_j;
            int tk = gk - tile_k;

            // Check if within tile bounds (should always be true for
            // sorted particles, but boundary cells may go out of range)
            if (ti >= 0 && ti < TILE_DIM &&
                tj >= 0 && tj < TILE_DIM &&
                tk >= 0 && tk < TILE_DIM) {
                int tile_idx = ti * TILE_DIM * TILE_DIM + tj * TILE_DIM + tk;

                // Shared memory atomics (much faster than global)
                atomicAdd(&tile[tile_idx * 4 + 0], mv[offset * 3 + 0]);
                atomicAdd(&tile[tile_idx * 4 + 1], mv[offset * 3 + 1]);
                atomicAdd(&tile[tile_idx * 4 + 2], mv[offset * 3 + 2]);
                atomicAdd(&tile[tile_idx * 4 + 3], m[offset]);
            } else {
                // Fallback: global atomic for out-of-tile nodes
                atomicAdd(&grid_mv[gid * 3 + 0], mv[offset * 3 + 0]);
                atomicAdd(&grid_mv[gid * 3 + 1], mv[offset * 3 + 1]);
                atomicAdd(&grid_mv[gid * 3 + 2], mv[offset * 3 + 2]);
                atomicAdd(&grid_m[gid],           m[offset]);
            }
        }
    }

    __syncthreads();

    // Flush shared memory tile to global memory
    // 64 tile entries, cooperatively across threads
    for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
        float smv0 = tile[t * 4 + 0];
        float smv1 = tile[t * 4 + 1];
        float smv2 = tile[t * 4 + 2];
        float sm   = tile[t * 4 + 3];

        // Skip empty tile entries
        if (sm == 0.0f && smv0 == 0.0f && smv1 == 0.0f && smv2 == 0.0f)
            continue;

        // Tile entry -> global grid index
        int ti = t / (TILE_DIM * TILE_DIM);
        int tj = (t / TILE_DIM) % TILE_DIM;
        int tk = t % TILE_DIM;
        int gi = tile_i + ti;
        int gj = tile_j + tj;
        int gk = tile_k + tk;

        // Boundary check
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

ffi::Error P2GScatterSmemImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> mv,
    ffi::Buffer<ffi::F32> m,
    ffi::Buffer<ffi::S32> index,
    ffi::Buffer<ffi::S32> cell_start,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m
) {
    // Infer G from cell_start size: cell_start has G^3 + 1 elements
    int G3_plus_1 = static_cast<int>(cell_start.dimensions()[0]);
    int G3 = G3_plus_1 - 1;
    // Compute G = cbrt(G3)
    int G = 1;
    while (G * G * G < G3) G++;
    if (G * G * G != G3) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cell_start size is not G^3 + 1 for integer G");
    }

    // Zero output grid
    int zero_blocks = (G3 * 3 + 255) / 256;
    zero_kernel<<<zero_blocks, 256, 0, stream>>>(grid_mv->typed_data(), G3 * 3);
    zero_kernel<<<(G3 + 255) / 256, 256, 0, stream>>>(grid_m->typed_data(), G3);

    // Launch: one block per cell
    p2g_scatter_smem_kernel<<<G3, BLOCK_SIZE, 0, stream>>>(
        mv.typed_data(),
        m.typed_data(),
        reinterpret_cast<const int*>(index.typed_data()),
        reinterpret_cast<const int*>(cell_start.typed_data()),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        G
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GScatterSmem, P2GScatterSmemImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // mv (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // m (sorted)
        .Arg<ffi::Buffer<ffi::S32>>()   // index (sorted)
        .Arg<ffi::Buffer<ffi::S32>>()   // cell_start
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
);
