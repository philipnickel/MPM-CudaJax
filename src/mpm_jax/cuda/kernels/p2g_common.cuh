// Shared device helpers for the inline P2G kernels (cuda_v1..v4).
//
// Only the pieces that are IDENTICAL across every variant by design live here:
// the per-particle register loads, the quadratic B-spline base/weight tables,
// the per-axis index clamp, and the per-node MLS-MPM contribution math. Each
// kernel keeps its OWN 27-stencil loop and scatter store (global atomicAdd /
// warp-reduced atomic / shared-tile atomic) — that scatter strategy is exactly
// what distinguishes the variants, so it stays local to each .cu. All helpers
// are __device__ __forceinline__, so codegen is unchanged vs. the inline form.
#pragma once

// Load one particle's AoS state into registers (read ONCE, reused 27x).
__device__ __forceinline__ void p2g_load_particle(
    const float* __restrict__ x, const float* __restrict__ v,
    const float* __restrict__ C, const float* __restrict__ stress,
    int pid, float px[3], float pv[3], float pC[9], float pS[9]) {
    for (int d = 0; d < 3; d++) {
        px[d] = x[pid * 3 + d];
        pv[d] = v[pid * 3 + d];
    }
    for (int i = 0; i < 9; i++) {
        pC[i] = C[pid * 9 + i];
        pS[i] = stress[pid * 9 + i];
    }
}

// Quadratic B-spline base node + fractional offset for a particle position.
// home/base convention must stay matched to home_super_cell_id in sort.py.
__device__ __forceinline__ void p2g_base_fx(
    const float px[3], float inv_dx, int base[3], float fx[3]) {
    for (int d = 0; d < 3; d++) {
        float fpx = px[d] * inv_dx;
        base[d] = (int)floorf(fpx - 0.5f);
        fx[d] = fpx - (float)base[d];
    }
}

// Per-axis quadratic B-spline weight (w) and weight-gradient (dw) tables,
// 3 entries each.
__device__ __forceinline__ void p2g_bspline_tables(
    const float fx[3], float w[3][3], float dw[3][3]) {
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
        dw[d][0] = fx[d] - 1.5f;
        dw[d][1] = -2.0f * (fx[d] - 1.0f);
        dw[d][2] = fx[d] - 0.5f;
    }
}

// Per-axis clamp of a stencil node index to [0, G-1]. NOTE: this is a per-axis
// clamp; the JAX reference (p2g_scan.py / g2p_scan.py) clips the flat index, so
// the two only diverge for out-of-domain stencil nodes (never at the interior
// benchmark). Keep both conventions in sync if a particle can reach the wall.
__device__ __forceinline__ int p2g_clip_axis(int idx, int G) {
    return max(0, min(idx, G - 1));
}

// MLS-MPM per-node contribution for stencil offset (di,dj,dk): the momentum
// mv[3] and the mass m_contrib. Identical in every variant; only the scatter
// store that consumes these differs.
//   mv = -dt*vol*(stress @ dweight) + p_mass*weight*(v + C @ dpos)
//   m  = weight * p_mass
__device__ __forceinline__ void p2g_node_contribution(
    int di, int dj, int dk,
    const float w[3][3], const float dw[3][3], const float fx[3],
    const float pC[9], const float pS[9], const float pv[3],
    float inv_dx, float dx, float dt, float vol, float p_mass,
    float mv[3], float* m_contrib) {
    float weight = w[0][di] * w[1][dj] * w[2][dk];

    // dweight = inv_dx * gradient of the weight along each axis.
    float dweight[3];
    dweight[0] = inv_dx * dw[0][di] * w[1][dj]  * w[2][dk];
    dweight[1] = inv_dx * w[0][di]  * dw[1][dj] * w[2][dk];
    dweight[2] = inv_dx * w[0][di]  * w[1][dj]  * dw[2][dk];

    // dpos = (offset - fx) * dx
    float dpos[3];
    dpos[0] = ((float)di - fx[0]) * dx;
    dpos[1] = ((float)dj - fx[1]) * dx;
    dpos[2] = ((float)dk - fx[2]) * dx;

    for (int d = 0; d < 3; d++) {
        float s_dw = 0.0f;
        float c_dp = 0.0f;
        for (int j = 0; j < 3; j++) {
            s_dw += pS[d * 3 + j] * dweight[j];
            c_dp += pC[d * 3 + j] * dpos[j];
        }
        mv[d] = -dt * vol * s_dw + p_mass * weight * (pv[d] + c_dp);
    }
    *m_contrib = weight * p_mass;
}

// Warp reduction over an arbitrary __match_any_sync peer mask (used by v2/v3):
// sum `val` across all lanes in `mask`, returning the sum in ALL lanes. Walk
// the set bits rather than an XOR butterfly, since match groups are not
// power-of-two aligned.
__device__ __forceinline__ float p2g_warp_reduce_masked(float val, unsigned mask) {
    float sum = 0.0f;
    for (unsigned remaining = mask; remaining; remaining &= (remaining - 1)) {
        int src_lane = __ffs(remaining) - 1;
        sum += __shfl_sync(mask, val, src_lane);
    }
    return sum;
}
