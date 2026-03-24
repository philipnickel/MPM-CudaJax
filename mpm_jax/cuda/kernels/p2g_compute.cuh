// p2g_compute.cuh — Shared device functions for fused P2G kernels.
//
// Standalone header (no JAX FFI dependency). Each thread computes weights,
// stress, and APIC momentum for one particle, producing 27 ParticleContrib
// entries ready for scatter.
//
// Usage:
//   #include "p2g_compute.cuh"
//   ParticleContrib contrib[STENCIL];
//   p2g_compute(px, py, pz, vx, vy, vz, C, F, dt, vol, p_mass,
//               inv_dx, num_grids, mu_0, lambda_0, contrib);

#pragma once

// No #include <math.h> — NVRTC provides CUDA math intrinsics (floorf, sqrtf, etc.)

#define STENCIL 27

// ---------------------------------------------------------------------------
// ParticleContrib: per-node output of p2g_compute
// ---------------------------------------------------------------------------

struct ParticleContrib {
    float mv[3];    // momentum contribution
    float m;        // mass contribution
    int   grid_idx; // flat grid index
};

// ---------------------------------------------------------------------------
// 3x3 matrix helpers (register-resident, row-major)
// ---------------------------------------------------------------------------

struct Mat3 {
    float m[9];  // row-major: m[i*3+j]
};

__device__ __forceinline__ Mat3 mat3_eye() {
    Mat3 I;
    #pragma unroll
    for (int i = 0; i < 9; i++) I.m[i] = 0.0f;
    I.m[0] = I.m[4] = I.m[8] = 1.0f;
    return I;
}

__device__ __forceinline__ Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 3; k++)
                s += A.m[i*3+k] * B.m[k*3+j];
            C.m[i*3+j] = s;
        }
    return C;
}

__device__ __forceinline__ Mat3 mat3_transpose(const Mat3& A) {
    Mat3 T;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            T.m[i*3+j] = A.m[j*3+i];
    return T;
}

__device__ __forceinline__ Mat3 mat3_scale(const Mat3& A, float s) {
    Mat3 B;
    #pragma unroll
    for (int i = 0; i < 9; i++) B.m[i] = A.m[i] * s;
    return B;
}

__device__ __forceinline__ Mat3 mat3_add(const Mat3& A, const Mat3& B) {
    Mat3 C;
    #pragma unroll
    for (int i = 0; i < 9; i++) C.m[i] = A.m[i] + B.m[i];
    return C;
}

__device__ __forceinline__ Mat3 mat3_sub(const Mat3& A, const Mat3& B) {
    Mat3 C;
    #pragma unroll
    for (int i = 0; i < 9; i++) C.m[i] = A.m[i] - B.m[i];
    return C;
}

// ---------------------------------------------------------------------------
// SVD placeholder for 3x3 matrices
//
// TODO: Implement proper Jacobi SVD (McAdams et al. 2011) or port the
// full implementation from p2g_fused.cu. This placeholder sets:
//   U = I,  sigma = diag(F),  V = I
// which is INCORRECT for off-diagonal F. It will produce wrong stress
// values. Replace with real SVD before validation on HPC.
// ---------------------------------------------------------------------------

__device__ void svd3x3(const Mat3& F, Mat3& U, float sigma[3], Mat3& V) {
    // PLACEHOLDER — treats F as if it were already diagonal.
    // Correct only when F is diagonal; wrong otherwise.
    U = mat3_eye();
    V = mat3_eye();
    sigma[0] = F.m[0];  // F[0,0]
    sigma[1] = F.m[4];  // F[1,1]
    sigma[2] = F.m[8];  // F[2,2]
}

// ---------------------------------------------------------------------------
// p2g_compute: full per-particle P2G computation
//
// Computes B-spline weights, corotated stress, and APIC momentum for one
// particle. Fills 27 ParticleContrib entries (one per stencil node).
//
// Parameters:
//   px, py, pz     — particle position
//   vx, vy, vz     — particle velocity
//   C_ptr           — pointer to 9 floats (3x3 APIC matrix, row-major)
//   F_ptr           — pointer to 9 floats (3x3 deformation gradient, row-major)
//   dt              — timestep
//   vol             — particle volume
//   p_mass          — particle mass
//   inv_dx          — 1/dx (grid spacing inverse)
//   num_grids       — grid dimension (G)
//   mu_0, lambda_0  — Lame parameters
//   out             — output array of 27 ParticleContrib
// ---------------------------------------------------------------------------

__device__ void p2g_compute(
    float px, float py, float pz,
    float vx, float vy, float vz,
    const float* __restrict__ C_ptr,
    const float* __restrict__ F_ptr,
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    ParticleContrib out[STENCIL]
) {
    float dx = 1.0f / inv_dx;

    // Load APIC matrix C and deformation gradient F
    Mat3 pC, pF;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        pC.m[i] = C_ptr[i];
        pF.m[i] = F_ptr[i];
    }

    // --- SVD of deformation gradient ---
    Mat3 U, V;
    float sigma[3];
    svd3x3(pF, U, sigma, V);

    // --- Corotated elasticity stress ---
    // P = 2*mu*(F - R) + lambda*(J-1)*J * F^{-T}
    // For corotated model with SVD:
    //   stress_diag[i] = 2*mu*(sigma[i] - 1)*sigma[i] + lambda*(J - 1)*J
    //   Kirchhoff stress tau = U @ diag(stress_diag) @ V^T
    float J = sigma[0] * sigma[1] * sigma[2];

    float stress_diag[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        stress_diag[i] = 2.0f * mu_0 * (sigma[i] - 1.0f) * sigma[i]
                       + lambda_0 * (J - 1.0f) * J;
    }

    // tau = U @ diag(stress_diag) @ V^T
    Mat3 Vt = mat3_transpose(V);
    Mat3 sig_diag = mat3_eye();
    sig_diag.m[0] = stress_diag[0];
    sig_diag.m[4] = stress_diag[1];
    sig_diag.m[8] = stress_diag[2];
    Mat3 tau = mat3_mul(mat3_mul(U, sig_diag), Vt);

    // --- Affine matrix: A = -dt * vol * 4*inv_dx^2 * tau + p_mass * C ---
    float coeff = -dt * vol * 4.0f * inv_dx * inv_dx;
    Mat3 affine = mat3_add(mat3_scale(tau, coeff), mat3_scale(pC, p_mass));

    // --- B-spline weights ---
    float fpx[3], fx[3];
    int base[3];
    fpx[0] = px * inv_dx;
    fpx[1] = py * inv_dx;
    fpx[2] = pz * inv_dx;

    #pragma unroll
    for (int d = 0; d < 3; d++) {
        base[d] = (int)floorf(fpx[d] - 0.5f);
        fx[d] = fpx[d] - (float)base[d];
    }

    float w[3][3];
    #pragma unroll
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
    }

    // --- Compute 27 contributions ---
    int idx = 0;
    #pragma unroll
    for (int di = 0; di < 3; di++)
    #pragma unroll
    for (int dj = 0; dj < 3; dj++)
    #pragma unroll
    for (int dk = 0; dk < 3; dk++) {
        float weight = w[0][di] * w[1][dj] * w[2][dk];

        float dpos[3];
        dpos[0] = ((float)di - fx[0]) * dx;
        dpos[1] = ((float)dj - fx[1]) * dx;
        dpos[2] = ((float)dk - fx[2]) * dx;

        // Grid index with clamping
        int gi = base[0] + di;
        int gj = base[1] + dj;
        int gk = base[2] + dk;
        gi = max(0, min(gi, num_grids - 1));
        gj = max(0, min(gj, num_grids - 1));
        gk = max(0, min(gk, num_grids - 1));
        int grid_idx = gi * num_grids * num_grids + gj * num_grids + gk;

        // mv[d] = weight * (p_mass * v[d] + affine[d,:] . dpos)
        float mv_out[3];
        float pv[3] = {vx, vy, vz};
        #pragma unroll
        for (int d = 0; d < 3; d++) {
            float a_dpos = 0.0f;
            #pragma unroll
            for (int j = 0; j < 3; j++)
                a_dpos += affine.m[d*3+j] * dpos[j];
            mv_out[d] = weight * (p_mass * pv[d] + a_dpos);
        }

        out[idx].mv[0] = mv_out[0];
        out[idx].mv[1] = mv_out[1];
        out[idx].mv[2] = mv_out[2];
        out[idx].m = weight * p_mass;
        out[idx].grid_idx = grid_idx;
        idx++;
    }
}
