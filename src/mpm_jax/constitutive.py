import jax
import jax.numpy as jnp


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
        # Closed-form det avoids cuSOLVER/SVD and matches the non-inverted regime.
        J = _det3x3(F)
        return 2.0 * mu * (F @ E_strain) + la * J * (J - 1.0) * I

    return jax.vmap(stress_single)
