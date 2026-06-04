"""Branch-free Jacobi SVD for batched 3x3 matrices, written in JAX.

The motivation is fusion: `jnp.linalg.svd` lowers to a cuSOLVER call, which is
a host-dispatched external library routine. XLA cannot fuse downstream
producers (stress, weights, momentum) into a single kernel across that
boundary, so the `(N, 27, 3)` momentum intermediate has to materialise in
HBM before the scatter kernel can consume it. A pure-JAX Jacobi SVD has only
pointwise ops + tiny stack/index manipulations, which XLA *can* fuse — at
least in principle.

Implementation follows McAdams et al. 2011: form S = F^T F, run 4 sweeps of
3 Givens rotations (pairs 01, 02, 12) to diagonalise S, then recover
sigma = sqrt(diag(S)) and U = F V diag(1/sigma).

The decomposition is carried *scatter-free*: S (symmetric) and V are held as
3x3 nested lists of per-batch (...,) leaf arrays rather than (..., 3, 3)
tensors, so the 12 Givens rotations contain no ``.at[].set()`` updates. This
is the difference between XLA fusing the whole SVD into a handful of
elementwise kernels (S/V register-resident) and emitting ~60
``dynamic-update-slice`` launches, each round-tripping S and V through HBM —
which, in the sand return-map, made plasticity the single biggest G2P kernel
(~1.4x slower per substep at 8M).

Matches `jnp.linalg.svd(F, full_matrices=False)` returning (U, sigma, Vh)
where Vh = V^T.
"""

import jax
import jax.numpy as jnp


def _givens_cs(a_pp, a_pq, a_qq, eps=1e-10):
    """Branch-free Givens coefficients zeroing a_pq in symmetric 2x2.

    Uses tau = (a_pp - a_qq) / (2 a_pq), the sign for the rotation
    G = [[c, -s], [s, c]] used by the symmetric update below.
    """
    small = jnp.abs(a_pq) < eps
    safe_pq = jnp.where(small, jnp.ones_like(a_pq), a_pq)
    tau = (a_pp - a_qq) / (2.0 * safe_pq)
    t = jnp.sign(tau) / (jnp.abs(tau) + jnp.sqrt(1.0 + tau * tau))
    c = jax.lax.rsqrt(1.0 + t * t)
    s = t * c
    c = jnp.where(small, jnp.ones_like(c), c)
    s = jnp.where(small, jnp.zeros_like(s), s)
    return c, s


def _rotate_sf(S, V, p, q, r):
    """Scatter-free Jacobi rotation: S, V are 3x3 nested lists of (...,) arrays.

    Every matrix entry is a separate leaf array, so the standard symmetric
    Jacobi update has no ``.at[].set()`` writes. (p, q, r) is a static
    permutation of (0, 1, 2), so the list indexing is resolved at trace time and
    XLA sees one flat elementwise dataflow it can fuse into a single kernel --
    instead of the ``dynamic-update-slice`` chain an array-valued S/V update
    lowers to, which fragments the SVD into dozens of HBM-bound launches.
    """
    Spp, Spq, Sqq = S[p][p], S[p][q], S[q][q]
    Spr, Sqr = S[p][r], S[q][r]

    c, s = _givens_cs(Spp, Spq, Sqq)

    new_Spp = c * c * Spp + 2.0 * c * s * Spq + s * s * Sqq
    new_Sqq = s * s * Spp - 2.0 * c * s * Spq + c * c * Sqq
    new_Spr = c * Spr + s * Sqr
    new_Sqr = -s * Spr + c * Sqr
    zero = jnp.zeros_like(Spq)

    NS = [list(row) for row in S]
    NS[p][p] = new_Spp
    NS[q][q] = new_Sqq
    NS[p][q] = NS[q][p] = zero
    NS[p][r] = NS[r][p] = new_Spr
    NS[q][r] = NS[r][q] = new_Sqr

    NV = [list(row) for row in V]
    for i in range(3):
        Vip, Viq = V[i][p], V[i][q]
        NV[i][p] = c * Vip + s * Viq
        NV[i][q] = -s * Vip + c * Viq
    return NS, NV


def jacobi_svd_3x3(F, num_sweeps=4):
    """SVD of batched 3x3 matrices via Jacobi iteration on F^T F.

    Scatter-free: S (symmetric) and V are carried as 3x3 nested lists of
    per-batch (...,) leaf arrays rather than (..., 3, 3) tensors, so the 12
    Givens rotations contain no ``.at[].set()`` updates. XLA fuses the whole
    decomposition into a handful of elementwise kernels with S/V resident in
    registers, instead of the ~60 ``dynamic-update-slice`` launches the
    array-valued formulation emits (each round-tripping S and V through HBM).
    Numerically equivalent to the array-valued Jacobi SVD / ``jnp.linalg.svd``.

    Parameters
    ----------
    F : (..., 3, 3) array
    num_sweeps : int
        Number of full sweeps (each sweep = 3 Givens rotations). 4 is enough
        for f32 convergence on physically reasonable deformation gradients
        (per McAdams 2011).

    Returns
    -------
    U : (..., 3, 3)
    sigma : (..., 3)
    Vh : (..., 3, 3)   (= V^T)
    """
    FtF = jnp.einsum("...ji,...jk->...ik", F, F)
    S = [[FtF[..., i, j] for j in range(3)] for i in range(3)]
    zero = jnp.zeros(F.shape[:-2], dtype=F.dtype)
    one = jnp.ones(F.shape[:-2], dtype=F.dtype)
    V = [[one if i == j else zero for j in range(3)] for i in range(3)]

    for _ in range(num_sweeps):
        S, V = _rotate_sf(S, V, 0, 1, 2)
        S, V = _rotate_sf(S, V, 0, 2, 1)
        S, V = _rotate_sf(S, V, 1, 2, 0)

    sigma2 = jnp.stack([S[0][0], S[1][1], S[2][2]], axis=-1)
    sigma = jnp.sqrt(jnp.maximum(sigma2, 0.0))

    Vmat = jnp.stack([jnp.stack(row, axis=-1) for row in V], axis=-2)
    FV = jnp.einsum("...ij,...jk->...ik", F, Vmat)
    inv_sigma = jnp.where(sigma > 1e-10, 1.0 / sigma, jnp.zeros_like(sigma))
    U = FV * inv_sigma[..., None, :]

    Vh = jnp.swapaxes(Vmat, -2, -1)
    return U, sigma, Vh
