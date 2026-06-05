import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from mpm_jax.constitutive import (
    REGISTRY,
    get_constitutive,
    stvk_elasticity_jacobi,
)


def _make_F_batch(N=10):
    return jnp.tile(jnp.eye(3), (N, 1, 1))


def test_stvk_jacobi_identity_F_gives_zero_stress():
    fn = stvk_elasticity_jacobi(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(10))
    assert stress.shape == (10, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)


def test_stvk_jacobi_stretched_F_gives_nonzero_stress():
    fn = stvk_elasticity_jacobi(E=2e6, nu=0.4)
    F = _make_F_batch(5).at[:, 0, 0].set(1.1)
    stress = fn(F)
    assert not jnp.allclose(stress, 0.0)


def test_registry_is_jelly_only():
    assert REGISTRY == {"StVKElasticityJacobi": stvk_elasticity_jacobi}


def test_get_constitutive_elasticity():
    cfg = OmegaConf.create({"name": "StVKElasticityJacobi", "E": 1e4, "nu": 0.4})
    fn = get_constitutive(cfg)
    stress = fn(_make_F_batch(5))
    assert stress.shape == (5, 3, 3)


def test_removed_material_names_raise_keyerror():
    # Drucker-Prager plasticity was removed with the SVD/sand path.
    cfg = OmegaConf.create({"name": "DruckerPragerPlasticityJacobi"})
    with pytest.raises(KeyError):
        get_constitutive(cfg)
