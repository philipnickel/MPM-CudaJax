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
// 3x3 SVD via one-sided Jacobi rotations
//
// Decomposes F = U * diag(sigma) * V^T using:
//   1. Symmetric eigendecomposition of F^T F = V * diag(sigma^2) * V^T
//      via cyclic Jacobi rotations (4 sweeps, each sweep = 3 off-diagonal pairs)
//   2. U = F * V * diag(1/sigma)
//
// Reference: Golub & Van Loan, "Matrix Computations", Section 8.4
// Suitable for MPM where F is typically close to rotation (well-conditioned).
// ---------------------------------------------------------------------------

// Apply Givens rotation to columns p, q of a 3x3 matrix (in-place).
// Rotates by angle with cos(theta)=c, sin(theta)=s.
__device__ __forceinline__ void givens_rotate_cols(Mat3& A, int p, int q, float c, float s) {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        float ap = A.m[i*3+p];
        float aq = A.m[i*3+q];
        A.m[i*3+p] = c * ap + s * aq;
        A.m[i*3+q] = -s * ap + c * aq;
    }
}

// Compute Givens rotation to zero S[p][q] in symmetric matrix S.
// Returns (c, s) such that the (p,q) element of G^T S G is zero.
__device__ __forceinline__ void sym_jacobi_rotation(const Mat3& S, int p, int q, float& c, float& s) {
    float spq = S.m[p*3+q];
    if (fabsf(spq) < 1e-10f) {
        c = 1.0f;
        s = 0.0f;
        return;
    }
    float tau = (S.m[q*3+q] - S.m[p*3+p]) / (2.0f * spq);
    float t;
    if (tau >= 0.0f)
        t = 1.0f / (tau + sqrtf(1.0f + tau * tau));
    else
        t = -1.0f / (-tau + sqrtf(1.0f + tau * tau));
    c = rsqrtf(1.0f + t * t);
    s = t * c;
}

// Apply Jacobi rotation to symmetric matrix S (both sides): S <- G^T S G
__device__ __forceinline__ void sym_rotate(Mat3& S, int p, int q, float c, float s) {
    float spp = S.m[p*3+p], sqq = S.m[q*3+q], spq = S.m[p*3+q];
    S.m[p*3+p] = c*c*spp - 2.0f*c*s*spq + s*s*sqq;
    S.m[q*3+q] = s*s*spp + 2.0f*c*s*spq + c*c*sqq;
    S.m[p*3+q] = S.m[q*3+p] = 0.0f;  // zeroed by construction

    // Update remaining off-diagonal entries
    int r = 3 - p - q;  // the third index
    float srp = S.m[r*3+p], srq = S.m[r*3+q];
    S.m[r*3+p] = S.m[p*3+r] = c * srp - s * srq;
    S.m[r*3+q] = S.m[q*3+r] = s * srp + c * srq;
}

__device__ void svd3x3(const Mat3& F, Mat3& U, float sigma[3], Mat3& V) {
    // Step 1: S = F^T F (symmetric positive semi-definite)
    Mat3 Ft = mat3_transpose(F);
    Mat3 S = mat3_mul(Ft, F);

    // Step 2: Eigendecomposition of S via Jacobi rotations
    // V accumulates the rotation matrices
    V = mat3_eye();

    // 6 sweeps of cyclic Jacobi (pairs: (0,1), (0,2), (1,2))
    // 4 sweeps gives ~1e-2 accuracy, 6 sweeps gives ~1e-6
    #pragma unroll
    for (int sweep = 0; sweep < 6; sweep++) {
        float c, s;

        // Pair (0, 1)
        sym_jacobi_rotation(S, 0, 1, c, s);
        sym_rotate(S, 0, 1, c, s);
        givens_rotate_cols(V, 0, 1, c, s);

        // Pair (0, 2)
        sym_jacobi_rotation(S, 0, 2, c, s);
        sym_rotate(S, 0, 2, c, s);
        givens_rotate_cols(V, 0, 2, c, s);

        // Pair (1, 2)
        sym_jacobi_rotation(S, 1, 2, c, s);
        sym_rotate(S, 1, 2, c, s);
        givens_rotate_cols(V, 1, 2, c, s);
    }

    // Step 3: sigma = sqrt(eigenvalues of S)
    // S is now approximately diagonal
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        sigma[i] = sqrtf(fmaxf(S.m[i*3+i], 0.0f));
    }

    // Step 4: U = F * V * diag(1/sigma)
    U = mat3_mul(F, V);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        float inv_s = (sigma[i] > 1e-8f) ? (1.0f / sigma[i]) : 0.0f;
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            U.m[j*3+i] *= inv_s;  // scale column i of U
        }
    }
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

    // --- Corotated elasticity Kirchhoff stress ---
    // Matches JAX: tau = 2*mu*(F - R) @ F^T + la*J*(J-1)*I
    // where R = U @ V^T (rotation from polar decomposition)
    float J = sigma[0] * sigma[1] * sigma[2];

    // R = U @ V^T
    Mat3 Vt = mat3_transpose(V);
    Mat3 R = mat3_mul(U, Vt);

    // F^T
    Mat3 Ft = mat3_transpose(pF);

    // corotated = 2*mu*(F - R) @ F^T
    Mat3 F_minus_R = mat3_sub(pF, R);
    Mat3 corotated = mat3_scale(mat3_mul(F_minus_R, Ft), 2.0f * mu_0);

    // volume = la * J * (J - 1) * I
    float vol_scalar = lambda_0 * J * (J - 1.0f);
    Mat3 vol_term = mat3_eye();
    vol_term = mat3_scale(vol_term, vol_scalar);

    // tau = corotated + volume
    Mat3 tau = mat3_add(corotated, vol_term);

    // --- B-spline weights and weight gradients ---
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

    // w[dim][offset] = quadratic B-spline weight
    float w[3][3];
    // dw[dim][offset] = weight derivative (before inv_dx scaling)
    float dw[3][3];
    #pragma unroll
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
        dw[d][0] = fx[d] - 1.5f;
        dw[d][1] = -2.0f * (fx[d] - 1.0f);
        dw[d][2] = fx[d] - 0.5f;
    }

    // --- Compute 27 contributions ---
    // JAX formula per node i:
    //   mv[i] = -dt*vol * stress @ dweight[i] + p_mass * weight[i] * (v + C @ dpos[i])
    int idx = 0;
    float pv[3] = {vx, vy, vz};

    #pragma unroll
    for (int di = 0; di < 3; di++)
    #pragma unroll
    for (int dj = 0; dj < 3; dj++)
    #pragma unroll
    for (int dk = 0; dk < 3; dk++) {
        float weight = w[0][di] * w[1][dj] * w[2][dk];

        // dweight[d] = inv_dx * (product of w/dw per dimension)
        // Matches JAX: dweight = inv_dx * [dw_x*w_y*w_z, w_x*dw_y*w_z, w_x*w_y*dw_z]
        float dwt[3];
        dwt[0] = inv_dx * dw[0][di] *  w[1][dj] *  w[2][dk];
        dwt[1] = inv_dx *  w[0][di] * dw[1][dj] *  w[2][dk];
        dwt[2] = inv_dx *  w[0][di] *  w[1][dj] * dw[2][dk];

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

        // mv[d] = -dt*vol * (tau @ dweight)[d] + p_mass * weight * (v[d] + (C @ dpos)[d])
        float mv_out[3];
        #pragma unroll
        for (int d = 0; d < 3; d++) {
            // stress term: tau[d,:] . dweight
            float stress_dw = 0.0f;
            // APIC term: C[d,:] . dpos
            float c_dpos = 0.0f;
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                stress_dw += tau.m[d*3+j] * dwt[j];
                c_dpos += pC.m[d*3+j] * dpos[j];
            }
            mv_out[d] = -dt * vol * stress_dw + p_mass * weight * (pv[d] + c_dpos);
        }

        out[idx].mv[0] = mv_out[0];
        out[idx].mv[1] = mv_out[1];
        out[idx].mv[2] = mv_out[2];
        out[idx].m = weight * p_mass;
        out[idx].grid_idx = grid_idx;
        idx++;
    }
}
