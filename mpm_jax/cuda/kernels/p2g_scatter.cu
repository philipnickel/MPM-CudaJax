// Naive P2G scatter kernel (v1).
//
// One thread per particle.  Each thread loops over its 27 stencil nodes and
// performs 4 global atomicAdd operations per node (3 momentum + 1 mass).
// This is the baseline for measuring atomic contention; v2 will add shared
// memory staging and spatial sorting.
//
// The per-particle *compute* (stress, weights, affine matrix) stays in JAX.
// This kernel only does the scatter: it takes precomputed momentum/mass
// contributions and the flat grid indices, and atomicAdds them onto the grid.
//
// Build:  nvcc -shared -o libp2g_scatter.so p2g_scatter.cu \
//              -arch=sm_90 -O3 --use_fast_math -Xcompiler -fPIC \
//              $(python -c "import jax; print(jax.ffi.include_dir())"  2>/dev/null | xargs -I{} echo -I{})
//
// Inputs (all float32):
//   mv:    (N, 27, 3) — momentum contribution per particle per stencil node
//   m:     (N, 27)    — mass contribution per particle per stencil node
//   index: (N, 27)    — flat grid index per particle per stencil node (int32)
//
// Outputs (all float32):
//   grid_mv: (G^3, 3) — grid momentum (zeroed, then scattered into)
//   grid_m:  (G^3,)   — grid mass

#include "xla/ffi/api/ffi.h"

#define BLOCK_SIZE 256
#define STENCIL 27

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Zero a float buffer (grid-stride loop).
__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

// Naive P2G scatter: one thread per particle, 27 atomicAdds.
__global__ void p2g_scatter_kernel(
    const float* __restrict__ mv,       // (N, 27, 3)
    const float* __restrict__ m,        // (N, 27)
    const int*   __restrict__ index,    // (N, 27)
    float*       __restrict__ grid_mv,  // (G^3, 3)
    float*       __restrict__ grid_m,   // (G^3,)
    int N
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    for (int s = 0; s < STENCIL; s++) {
        int offset = pid * STENCIL + s;
        int gid = index[offset];

        // 3 momentum components + 1 mass = 4 global atomicAdds per node
        atomicAdd(&grid_mv[gid * 3 + 0], mv[offset * 3 + 0]);
        atomicAdd(&grid_mv[gid * 3 + 1], mv[offset * 3 + 1]);
        atomicAdd(&grid_mv[gid * 3 + 2], mv[offset * 3 + 2]);
        atomicAdd(&grid_m[gid],           m[offset]);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GScatterImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> mv,
    ffi::Buffer<ffi::F32> m,
    ffi::Buffer<ffi::S32> index,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m
) {
    // mv is (N, 27, 3) → total elements = N * 27 * 3
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

    // Launch scatter
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    p2g_scatter_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        mv.typed_data(),
        m.typed_data(),
        reinterpret_cast<const int*>(index.typed_data()),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        N
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(
            ffi::ErrorCode::kInternal,
            cudaGetErrorString(err)
        );
    }

    return ffi::Error::Success();
}

// Register the FFI handler
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GScatter, P2GScatterImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // mv
        .Arg<ffi::Buffer<ffi::F32>>()   // m
        .Arg<ffi::Buffer<ffi::S32>>()   // index
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
);
