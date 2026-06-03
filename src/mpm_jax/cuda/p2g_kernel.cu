// Hand-written P2G scatter kernel for benchmarking against JAX XLA.
// Each thread handles one particle: computes B-spline weights over 3x3x3
// neighborhood and atomicAdds momentum and mass to the grid.

extern "C" {

__global__ void p2g_kernel(
    const double* __restrict__ x,
    const double* __restrict__ v,
    const double* __restrict__ C,
    const double* __restrict__ stress,
    double* __restrict__ grid_mv,
    double* __restrict__ grid_m,
    int N,
    int G,
    double dt,
    double vol,
    double p_mass,
    double inv_dx,
    double dx
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    double px[3];
    for (int d = 0; d < 3; d++)
        px[d] = x[pid * 3 + d] * inv_dx;

    int base[3];
    double fx[3];
    for (int d = 0; d < 3; d++) {
        base[d] = (int)floor(px[d] - 0.5);
        fx[d] = px[d] - (double)base[d];
    }

    double w[3][3], dw[3][3];
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5 * (1.5 - fx[d]) * (1.5 - fx[d]);
        w[d][1] = 0.75 - (fx[d] - 1.0) * (fx[d] - 1.0);
        w[d][2] = 0.5 * (fx[d] - 0.5) * (fx[d] - 0.5);
        dw[d][0] = fx[d] - 1.5;
        dw[d][1] = -2.0 * (fx[d] - 1.0);
        dw[d][2] = fx[d] - 0.5;
    }

    double pv[3], pC[3][3], pstress[3][3];
    for (int d = 0; d < 3; d++)
        pv[d] = v[pid * 3 + d];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            pC[i][j] = C[pid * 9 + i * 3 + j];
            pstress[i][j] = stress[pid * 9 + i * 3 + j];
        }

    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        double weight = w[0][di] * w[1][dj] * w[2][dk];

        double dweight[3];
        dweight[0] = inv_dx * dw[0][di] * w[1][dj] * w[2][dk];
        dweight[1] = inv_dx * w[0][di] * dw[1][dj] * w[2][dk];
        dweight[2] = inv_dx * w[0][di] * w[1][dj] * dw[2][dk];

        double dpos[3];
        dpos[0] = ((double)di - fx[0]) * dx;
        dpos[1] = ((double)dj - fx[1]) * dx;
        dpos[2] = ((double)dk - fx[2]) * dx;

        int gi = base[0] + di;
        int gj = base[1] + dj;
        int gk = base[2] + dk;
        if (gi < 0) gi = 0; if (gi >= G) gi = G - 1;
        if (gj < 0) gj = 0; if (gj >= G) gj = G - 1;
        if (gk < 0) gk = 0; if (gk >= G) gk = G - 1;
        int grid_idx = gi * G * G + gj * G + gk;

        double mv[3];
        for (int d = 0; d < 3; d++) {
            double stress_term = 0.0;
            for (int j = 0; j < 3; j++)
                stress_term += pstress[d][j] * dweight[j];

            double c_dpos = 0.0;
            for (int j = 0; j < 3; j++)
                c_dpos += pC[d][j] * dpos[j];

            mv[d] = -dt * vol * stress_term + p_mass * weight * (pv[d] + c_dpos);
        }

        for (int d = 0; d < 3; d++)
            atomicAdd(&grid_mv[grid_idx * 3 + d], mv[d]);
        atomicAdd(&grid_m[grid_idx], weight * p_mass);
    }
}

} // extern "C"
