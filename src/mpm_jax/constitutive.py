import math

import jax
import jax.numpy as jnp

from mpm_jax.blocks.svd import jacobi_svd_3x3


def _lame_params(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, la


def _det3x3(F):
    """Determinant of a single 3x3 (``vmap`` for batches): the scalar triple
    product ``c0 . (c1 x c2)`` of its columns. Closed-form and fusable, unlike
    ``jnp.linalg.det`` which lowers to a fusion-breaking cuSOLVER LU custom-call.
    """
    return F[:, 0] @ jnp.cross(F[:, 1], F[:, 2])


def stvk_elasticity_jacobi(E=2e6, nu=0.4):
    mu, la = _lame_params(E, nu)

    def stress_single(F):
        """StVK first Piola-Kirchhoff stress for one 3x3 deformation gradient."""
        I = jnp.eye(3, dtype=F.dtype)
        E_strain = 0.5 * (F.T @ F - I)
        # J = det(F) via the closed-form det: no cuSOLVER, and it fuses. (The old
        # code took prod(singular values) = |det(F)| from a full SVD whose U, Vh
        # were discarded -- a whole SVD per substep for one scalar. Identical for
        # the physical non-inverted regime, det>0.)
        J = _det3x3(F)
        return 2.0 * mu * (F @ E_strain) + la * J * (J - 1.0) * I

    return jax.vmap(stress_single)


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
        epsilon_hat_norm = jnp.clip(
            jnp.linalg.norm(epsilon_hat, axis=-1, keepdims=True), 1e-10
        )

        expand_epsilon = jnp.ones_like(epsilon) * cohesion
        shifted_trace = trace - cohesion * 3.0
        cond_yield = (shifted_trace < 0).reshape(-1, 1)

        delta_gamma = (
            epsilon_hat_norm
            + (3.0 * la + 2.0 * mu) / (2.0 * mu) * shifted_trace * alpha
        )
        compress_epsilon = (
            epsilon - (jnp.clip(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat
        )

        epsilon = jnp.where(cond_yield, compress_epsilon, expand_epsilon)
        diag_exp = jax.vmap(jnp.diag)(jnp.exp(epsilon))
        return U @ diag_exp @ Vh

    return apply


# Name -> constitutive-factory registry. sand_jacobi wires StVK elasticity +
# Drucker-Prager plasticity; jelly wires StVK elasticity with no plasticity.
# (A neo-Hookean elastic stress was tried for jelly but dropped: its log(J)/F^-T
# singularity blows up at fine grids on impact; StVK is polynomial in F and stable.)
# Add a function and a config entry to extend it.
REGISTRY = {
    "StVKElasticityJacobi": stvk_elasticity_jacobi,
    "DruckerPragerPlasticityJacobi": drucker_prager_plasticity_jacobi,
}


def get_constitutive(cfg):
    """Construct a constitutive fn from a config node: look up ``cfg['name']`` in
    REGISTRY and pass the remaining keys as kwargs. ``dict(cfg)`` materialises
    both a plain dict and an OmegaConf DictConfig."""
    params = dict(cfg)
    name = params.pop("name")
    return REGISTRY[name](**params)
