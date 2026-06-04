import math

import jax
import jax.numpy as jnp

from mpm_jax.blocks.svd import jacobi_svd_3x3


def _lame_params(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, la


def _det3x3(F):
    """Closed-form determinant of batched 3x3 matrices.

    Fully fusable (~9 mul + adds, pointwise per matrix) -- unlike
    ``jnp.linalg.det``, which lowers to a cuSOLVER LU custom-call (host-
    dispatched, fusion-breaking).
    """
    a, b, c = F[..., 0, 0], F[..., 0, 1], F[..., 0, 2]
    d, e, f = F[..., 1, 0], F[..., 1, 1], F[..., 1, 2]
    g, h, i = F[..., 2, 0], F[..., 2, 1], F[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def stvk_elasticity_jacobi(E=2e6, nu=0.4):
    mu, la = _lame_params(E, nu)

    def compute_stress(F):
        Ft = jnp.swapaxes(F, -2, -1)
        FtF = Ft @ F
        I = jnp.eye(3, dtype=F.dtype)
        E_strain = 0.5 * (FtF - I)
        stvk = 2.0 * mu * (F @ E_strain)
        # J = det(F), the StVK Jacobian. The old code took prod(singular values)
        # = |det(F)| from a full SVD whose U, Vh were thrown away -- a whole SVD
        # per substep for one scalar. For non-inverted F (det>0, the physical
        # regime) the closed-form determinant is identical, and it fuses.
        J = _det3x3(F).reshape(-1, 1, 1)
        volume = la * J * (J - 1.0) * I
        return stvk + volume

    return compute_stress


def drucker_prager_plasticity_jacobi(E=2e6, nu=0.4, friction_angle=25.0, cohesion=0.0):
    mu, la = _lame_params(E, nu)
    sin_phi = jnp.sin(jnp.deg2rad(friction_angle))
    alpha = math.sqrt(2.0 / 3.0) * 2.0 * float(sin_phi) / (3.0 - float(sin_phi))

    def apply(F):
        U, sigma, Vh = jacobi_svd_3x3(F)
        sigma = jnp.clip(sigma, 0.05)
        epsilon = jnp.log(sigma)
        trace = epsilon.sum(axis=-1, keepdims=True)
        epsilon_hat = epsilon - trace / 3.0
        epsilon_hat_norm = jnp.clip(jnp.linalg.norm(epsilon_hat, axis=-1, keepdims=True), 1e-10)

        expand_epsilon = jnp.ones_like(epsilon) * cohesion
        shifted_trace = trace - cohesion * 3.0
        cond_yield = (shifted_trace < 0).reshape(-1, 1)

        delta_gamma = (
            epsilon_hat_norm
            + (3.0 * la + 2.0 * mu) / (2.0 * mu) * shifted_trace * alpha
        )
        compress_epsilon = epsilon - (jnp.clip(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = jnp.where(cond_yield, compress_epsilon, expand_epsilon)
        diag_exp = jax.vmap(jnp.diag)(jnp.exp(epsilon))
        return U @ diag_exp @ Vh

    return apply


ELASTICITY = {
    "StVKElasticityJacobi": stvk_elasticity_jacobi,
}

PLASTICITY = {
    "DruckerPragerPlasticityJacobi": drucker_prager_plasticity_jacobi,
}

REGISTRY = {**ELASTICITY, **PLASTICITY}


def get_constitutive(cfg):
    params = {k: v for k, v in cfg.items() if k != "name"}
    return REGISTRY[cfg.name](**params)
