// Fused P2G kernel (v2).
//
// One thread per particle does the FULL P2G pipeline:
//   1. Compute B-spline weights from position
//   2. Compute Kirchhoff stress from deformation gradient (SVD + plasticity)
//   3. Build affine momentum matrix
//   4. Scatter momentum + mass to 27 stencil nodes via atomicAdd
//
// Everything stays in registers — no global memory round-trips for
// intermediate results (stress, weights, affine matrix).
//
// Inputs:
//   x:   (N, 3)    particle positions
//   v:   (N, 3)    particle velocities
//   C:   (N, 3, 3) APIC affine matrix
//   F:   (N, 3, 3) elastic deformation gradient
//
// Outputs:
//   grid_mv: (G^3, 3)  grid momentum
//   grid_m:  (G^3,)    grid mass
//   F_out:   (N, 3, 3) plasticity-corrected deformation gradient
//
// Scalar attributes: N, G, dt, vol, p_mass, inv_dx, dx, mu_0, lambda_0

#include "xla/ffi/api/ffi.h"
#include <math.h>

#define BLOCK_SIZE 256

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Device helpers: 3x3 matrix operations (register-resident)
// ---------------------------------------------------------------------------

struct Mat3 {
    float m[9];  // row-major: m[i*3+j]
};

__device__ __forceinline__ Mat3 mat3_load(const float* p, int stride) {
    Mat3 A;
    for (int i = 0; i < 9; i++) A.m[i] = p[stride * i];
    return A;
}

__device__ __forceinline__ Mat3 mat3_load_row(const float* p) {
    Mat3 A;
    for (int i = 0; i < 9; i++) A.m[i] = p[i];
    return A;
}

__device__ __forceinline__ void mat3_store(float* p, const Mat3& A) {
    for (int i = 0; i < 9; i++) p[i] = A.m[i];
}

__device__ __forceinline__ Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++)
                s += A.m[i*3+k] * B.m[k*3+j];
            C.m[i*3+j] = s;
        }
    return C;
}

__device__ __forceinline__ Mat3 mat3_transpose(const Mat3& A) {
    Mat3 T;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            T.m[i*3+j] = A.m[j*3+i];
    return T;
}

__device__ __forceinline__ Mat3 mat3_eye() {
    Mat3 I;
    for (int i = 0; i < 9; i++) I.m[i] = 0.0f;
    I.m[0] = I.m[4] = I.m[8] = 1.0f;
    return I;
}

__device__ __forceinline__ Mat3 mat3_scale(const Mat3& A, float s) {
    Mat3 B;
    for (int i = 0; i < 9; i++) B.m[i] = A.m[i] * s;
    return B;
}

__device__ __forceinline__ Mat3 mat3_add(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 9; i++) C.m[i] = A.m[i] + B.m[i];
    return C;
}

__device__ __forceinline__ Mat3 mat3_sub(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 9; i++) C.m[i] = A.m[i] - B.m[i];
    return C;
}

__device__ __forceinline__ float mat3_det(const Mat3& A) {
    return A.m[0] * (A.m[4]*A.m[8] - A.m[5]*A.m[7])
         - A.m[1] * (A.m[3]*A.m[8] - A.m[5]*A.m[6])
         + A.m[2] * (A.m[3]*A.m[7] - A.m[4]*A.m[6]);
}

// ---------------------------------------------------------------------------
// SVD for 3x3 matrices (Jacobi iteration, McAdams et al. 2011)
//
// Computes F = U * diag(sigma) * V^T
// Uses 4 sweeps of 3 Jacobi rotations for convergence at float32.
// ---------------------------------------------------------------------------

__device__ void givens_coeffs(float a_pp, float a_pq, float a_qq,
                               float& c, float& s) {
    // Compute Givens rotation to zero a_pq in symmetric 2x2 [a_pp, a_pq; a_pq, a_qq]
    if (fabsf(a_pq) < 1e-10f) {
        c = 1.0f; s = 0.0f;
        return;
    }
    float tau = (a_qq - a_pp) / (2.0f * a_pq);
    float t = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
    c = rsqrtf(1.0f + t * t);
    s = t * c;
}

__device__ void jacobi_svd3(const Mat3& F, Mat3& U, float sigma[3], Mat3& V) {
    // Compute F^T F
    Mat3 Ft = mat3_transpose(F);
    Mat3 FtF = mat3_mul(Ft, F);

    // Jacobi eigendecomposition of F^T F → V * diag(sigma^2) * V^T
    V = mat3_eye();
    float S[9];
    for (int i = 0; i < 9; i++) S[i] = FtF.m[i];

    // 4 sweeps, 3 rotations per sweep (pairs: 01, 02, 12)
    for (int sweep = 0; sweep < 4; sweep++) {
        // Pair (0, 1)
        {
            float c, s;
            givens_coeffs(S[0], S[1], S[4], c, s);
            // Rotate S
            float S00 = c*c*S[0] + 2*c*s*S[1] + s*s*S[4];
            float S11 = s*s*S[0] - 2*c*s*S[1] + c*c*S[4];
            float S01 = 0.0f;
            float S02 = c*S[2] + s*S[5];
            float S12 = -s*S[2] + c*S[5];
            S[0] = S00; S[4] = S11; S[1] = S01; S[3] = S01;
            S[2] = S02; S[6] = S02; S[5] = S12; S[7] = S12;
            // Rotate V
            for (int i = 0; i < 3; i++) {
                float v0 = V.m[i*3+0], v1 = V.m[i*3+1];
                V.m[i*3+0] = c*v0 + s*v1;
                V.m[i*3+1] = -s*v0 + c*v1;
            }
        }
        // Pair (0, 2)
        {
            float c, s;
            givens_coeffs(S[0], S[2], S[8], c, s);
            float S00 = c*c*S[0] + 2*c*s*S[2] + s*s*S[8];
            float S22 = s*s*S[0] - 2*c*s*S[2] + c*c*S[8];
            float S02 = 0.0f;
            float S01 = c*S[1] + s*S[7];
            float S12 = -s*S[1] + c*S[7];
            S[0] = S00; S[8] = S22; S[2] = S02; S[6] = S02;
            S[1] = S01; S[3] = S01; S[5] = S12; S[7] = S12;
            for (int i = 0; i < 3; i++) {
                float v0 = V.m[i*3+0], v2 = V.m[i*3+2];
                V.m[i*3+0] = c*v0 + s*v2;
                V.m[i*3+2] = -s*v0 + c*v2;
            }
        }
        // Pair (1, 2)
        {
            float c, s;
            givens_coeffs(S[4], S[5], S[8], c, s);
            float S11 = c*c*S[4] + 2*c*s*S[5] + s*s*S[8];
            float S22 = s*s*S[4] - 2*c*s*S[5] + c*c*S[8];
            float S12 = 0.0f;
            float S01 = c*S[3] + s*S[6];
            float S02 = -s*S[3] + c*S[6];
            S[4] = S11; S[8] = S22; S[5] = S12; S[7] = S12;
            S[3] = S01; S[1] = S01; S[6] = S02; S[2] = S02;
            for (int i = 0; i < 3; i++) {
                float v1 = V.m[i*3+1], v2 = V.m[i*3+2];
                V.m[i*3+1] = c*v1 + s*v2;
                V.m[i*3+2] = -s*v1 + c*v2;
            }
        }
    }

    // sigma = sqrt(eigenvalues of F^T F)
    for (int i = 0; i < 3; i++)
        sigma[i] = sqrtf(fmaxf(S[i*4], 0.0f));

    // U = F * V * diag(1/sigma)
    Mat3 FV = mat3_mul(F, V);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            U.m[i*3+j] = (sigma[j] > 1e-10f) ? FV.m[i*3+j] / sigma[j] : 0.0f;
}

// ---------------------------------------------------------------------------
// Zero kernel
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Fused P2G kernel: stress + weights + compute + scatter
// ---------------------------------------------------------------------------

__global__ void p2g_fused_kernel(
    const float* __restrict__ x,        // (N, 3)
    const float* __restrict__ v,        // (N, 3)
    const float* __restrict__ C,        // (N, 9) row-major
    const float* __restrict__ F,        // (N, 9) row-major
    float*       __restrict__ grid_mv,  // (G^3, 3)
    float*       __restrict__ grid_m,   // (G^3,)
    float*       __restrict__ F_out,    // (N, 9) corrected F after plasticity
    int N, int G,
    float dt, float vol, float p_mass, float inv_dx, float dx,
    float mu_0, float lambda_0,
    float theta_c, float theta_s, float hardening_coeff
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    // Load particle state into registers
    float px[3], pv[3];
    for (int d = 0; d < 3; d++) {
        px[d] = x[pid * 3 + d];
        pv[d] = v[pid * 3 + d];
    }

    Mat3 pC, pF;
    for (int i = 0; i < 9; i++) {
        pC.m[i] = C[pid * 9 + i];
        pF.m[i] = F[pid * 9 + i];
    }

    // --- SVD of deformation gradient ---
    Mat3 U, V;
    float sigma[3];
    jacobi_svd3(pF, U, sigma, V);

    // --- Plasticity: clamp singular values ---
    float sig_clamped[3];
    for (int i = 0; i < 3; i++)
        sig_clamped[i] = fminf(fmaxf(sigma[i], 1.0f - theta_c), 1.0f + theta_s);

    // Reconstruct corrected F = U * diag(sig_clamped) * V^T
    Mat3 Vt = mat3_transpose(V);
    Mat3 sig_diag = mat3_eye();
    sig_diag.m[0] = sig_clamped[0];
    sig_diag.m[4] = sig_clamped[1];
    sig_diag.m[8] = sig_clamped[2];
    Mat3 F_new = mat3_mul(mat3_mul(U, sig_diag), Vt);

    // Store corrected F
    mat3_store(&F_out[pid * 9], F_new);

    // --- Hardening ---
    float J_new = sig_clamped[0] * sig_clamped[1] * sig_clamped[2];
    // Note: for the simple corotated model without snow hardening,
    // mu = mu_0 and lambda = lambda_0 (hardening_coeff=0 gives h=1)
    float h = expf(hardening_coeff * (1.0f - J_new));
    float mu = mu_0 * h;
    float lam = lambda_0 * h;

    // --- Kirchhoff stress: tau = 2*mu*(F_new - R)*F_new^T + lambda*(J-1)*J*I ---
    Mat3 R = mat3_mul(U, Vt);  // rotation
    Mat3 F_new_t = mat3_transpose(F_new);
    Mat3 tau = mat3_add(
        mat3_scale(mat3_mul(mat3_sub(F_new, R), F_new_t), 2.0f * mu),
        mat3_scale(mat3_eye(), lam * (J_new - 1.0f) * J_new)
    );

    // --- Affine matrix: A = -dt * vol * 4*inv_dx^2 * tau + p_mass * C ---
    float coeff = -dt * vol * 4.0f * inv_dx * inv_dx;
    Mat3 affine = mat3_add(mat3_scale(tau, coeff), mat3_scale(pC, p_mass));

    // --- B-spline weights ---
    float fpx[3], fx[3];
    int base[3];
    for (int d = 0; d < 3; d++) {
        fpx[d] = px[d] * inv_dx;
        base[d] = (int)floorf(fpx[d] - 0.5f);
        fx[d] = fpx[d] - (float)base[d];
    }

    float w[3][3];
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
    }

    // --- Scatter to 27 stencil nodes ---
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        float weight = w[0][di] * w[1][dj] * w[2][dk];

        float dpos[3];
        dpos[0] = ((float)di - fx[0]) * dx;
        dpos[1] = ((float)dj - fx[1]) * dx;
        dpos[2] = ((float)dk - fx[2]) * dx;

        int gi = base[0] + di;
        int gj = base[1] + dj;
        int gk = base[2] + dk;
        gi = max(0, min(gi, G - 1));
        gj = max(0, min(gj, G - 1));
        gk = max(0, min(gk, G - 1));
        int grid_idx = gi * G * G + gj * G + gk;

        // mv = weight * (p_mass * v + affine @ dpos)
        float mv[3];
        for (int d = 0; d < 3; d++) {
            float a_dpos = 0.0f;
            for (int j = 0; j < 3; j++)
                a_dpos += affine.m[d*3+j] * dpos[j];
            mv[d] = weight * (p_mass * pv[d] + a_dpos);
        }

        for (int d = 0; d < 3; d++)
            atomicAdd(&grid_mv[grid_idx * 3 + d], mv[d]);
        atomicAdd(&grid_m[grid_idx], weight * p_mass);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GFusedImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> F,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    ffi::ResultBuffer<ffi::F32> F_out,
    int32_t N,
    int32_t G,
    float dt, float vol, float p_mass, float inv_dx, float dx,
    float mu_0, float lambda_0,
    float theta_c, float theta_s, float hardening_coeff
) {
    // Sizes
    int grid_mv_size = G * G * G * 3;
    int grid_m_size = G * G * G;

    // Zero output grid buffers
    int zero_blocks = (grid_mv_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_mv->typed_data(), grid_mv_size);
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_m->typed_data(), grid_m_size);

    // Launch fused P2G
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    p2g_fused_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        F.typed_data(),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        F_out->typed_data(),
        N, G,
        dt, vol, p_mass, inv_dx, dx,
        mu_0, lambda_0,
        theta_c, theta_s, hardening_coeff
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GFused, P2GFusedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x
        .Arg<ffi::Buffer<ffi::F32>>()   // v
        .Arg<ffi::Buffer<ffi::F32>>()   // C
        .Arg<ffi::Buffer<ffi::F32>>()   // F
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Ret<ffi::Buffer<ffi::F32>>()   // F_out
        .Attr<int32_t>("N")
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
        .Attr<float>("mu_0")
        .Attr<float>("lambda_0")
        .Attr<float>("theta_c")
        .Attr<float>("theta_s")
        .Attr<float>("hardening_coeff")
);
