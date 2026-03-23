// Warp-reduction P2G scatter kernel (v3).
//
// Same interface as v1 (takes precomputed mv, m, index), but before each
// atomicAdd, threads within a warp that target the same grid node reduce
// their contributions via __match_any_sync + __shfl_down_sync.
//
// This reduces the number of global atomics from 4*27*N to roughly
// 4*27*N/k where k is the average number of warp lanes per unique node.
// At high particle density (many particles in same cell), k can be 4-8,
// giving 4-8x fewer atomics.
//
// Requires sm_70+ for __match_any_sync.

#include "xla/ffi/api/ffi.h"

#define BLOCK_SIZE 256
#define STENCIL 27
#define FULL_MASK 0xFFFFFFFFu

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Warp-level reduction helper
// ---------------------------------------------------------------------------

// Reduce `val` across all lanes in `mask` using butterfly shuffle.
// Returns the sum in ALL lanes of the group (not just the leader).
__device__ __forceinline__ float warp_reduce_masked(float val, unsigned mask) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(mask, val, delta);
        // Only add if the other lane is actually in our group
        if (mask & (1u << ((threadIdx.x & 31) ^ delta)))
            val += other;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

__global__ void p2g_scatter_warp_kernel(
    const float* __restrict__ mv,       // (N, 27, 3)
    const float* __restrict__ m,        // (N, 27)
    const int*   __restrict__ index,    // (N, 27)
    float*       __restrict__ grid_mv,  // (G^3, 3)
    float*       __restrict__ grid_m,   // (G^3,)
    int N
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    int lane = threadIdx.x & 31;

    for (int s = 0; s < STENCIL; s++) {
        int offset = pid * STENCIL + s;
        int gid = index[offset];

        // Load this thread's contributions
        float mv0 = mv[offset * 3 + 0];
        float mv1 = mv[offset * 3 + 1];
        float mv2 = mv[offset * 3 + 2];
        float mass = m[offset];

        // Find all lanes in this warp targeting the same grid node
        unsigned peers = __match_any_sync(FULL_MASK, gid);

        // Reduce contributions across matching lanes
        mv0 = warp_reduce_masked(mv0, peers);
        mv1 = warp_reduce_masked(mv1, peers);
        mv2 = warp_reduce_masked(mv2, peers);
        mass = warp_reduce_masked(mass, peers);

        // Only the leader (lowest lane in group) does the atomic
        int leader = __ffs(peers) - 1;  // __ffs returns 1-indexed
        if (lane == leader) {
            atomicAdd(&grid_mv[gid * 3 + 0], mv0);
            atomicAdd(&grid_mv[gid * 3 + 1], mv1);
            atomicAdd(&grid_mv[gid * 3 + 2], mv2);
            atomicAdd(&grid_m[gid], mass);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GScatterWarpImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> mv,
    ffi::Buffer<ffi::F32> m,
    ffi::Buffer<ffi::S32> index,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m
) {
    auto mv_dims = mv.dimensions();
    int N = static_cast<int>(mv_dims[0]);

    auto grid_mv_dims = grid_mv->dimensions();
    int grid_mv_size = 1;
    for (auto d : grid_mv_dims) grid_mv_size *= static_cast<int>(d);

    auto grid_m_dims = grid_m->dimensions();
    int grid_m_size = 1;
    for (auto d : grid_m_dims) grid_m_size *= static_cast<int>(d);

    // Zero output buffers
    int zero_blocks = (grid_mv_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_mv->typed_data(), grid_mv_size);
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_m->typed_data(), grid_m_size);

    // Launch warp-reduction scatter
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    p2g_scatter_warp_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        mv.typed_data(),
        m.typed_data(),
        reinterpret_cast<const int*>(index.typed_data()),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GScatterWarp, P2GScatterWarpImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // mv
        .Arg<ffi::Buffer<ffi::F32>>()   // m
        .Arg<ffi::Buffer<ffi::S32>>()   // index
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
);
