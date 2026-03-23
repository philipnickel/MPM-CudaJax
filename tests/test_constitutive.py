import jax.numpy as jnp
from mpm_jax.constitutive import (
    corotated_elasticity, sigma_elasticity, stvk_elasticity,
    fluid_elasticity, volume_elasticity,
    ELASTICITY,
)

def _make_F_batch(N=10):
    return jnp.tile(jnp.eye(3), (N, 1, 1))

def test_corotated_identity_F_gives_zero_stress():
    fn = corotated_elasticity(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(20))
    assert stress.shape == (20, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)

def test_sigma_elasticity_identity_F_gives_zero_stress():
    fn = sigma_elasticity(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(10))
    assert stress.shape == (10, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)

def test_stvk_identity_F_gives_zero_stress():
    fn = stvk_elasticity(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(10))
    assert stress.shape == (10, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)

def test_fluid_identity_F_gives_zero_stress():
    fn = fluid_elasticity(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(10))
    assert stress.shape == (10, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)

def test_volume_identity_F_gives_zero_stress():
    fn = volume_elasticity(E=2e6, nu=0.4)
    stress = fn(_make_F_batch(10))
    assert stress.shape == (10, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)

def test_corotated_stretched_F_gives_nonzero_stress():
    fn = corotated_elasticity(E=2e6, nu=0.4)
    F = _make_F_batch(5).at[:, 0, 0].set(1.1)
    stress = fn(F)
    assert not jnp.allclose(stress, 0.0)

def test_elasticity_registry():
    assert "CorotatedElasticity" in ELASTICITY
    assert "SigmaElasticity" in ELASTICITY
    assert "StVKElasticity" in ELASTICITY
    assert "FluidElasticity" in ELASTICITY
    assert "VolumeElasticity" in ELASTICITY

from mpm_jax.constitutive import (
    identity_plasticity, drucker_prager_plasticity,
    von_mises_plasticity, sigma_plasticity,
    PLASTICITY, get_constitutive,
)

def test_identity_plasticity_returns_F_unchanged():
    F = _make_F_batch(10)
    fn = identity_plasticity()
    result = fn(F)
    assert jnp.allclose(result, F)

def test_drucker_prager_preserves_shape():
    fn = drucker_prager_plasticity(E=2e6, nu=0.4, friction_angle=25.0, cohesion=0.0)
    F = _make_F_batch(10)
    result = fn(F)
    assert result.shape == (10, 3, 3)
    assert jnp.allclose(result, F, atol=1e-2)

def test_von_mises_preserves_shape():
    fn = von_mises_plasticity(E=2e6, nu=0.4, sigma_y=1e3)
    F = _make_F_batch(10)
    result = fn(F)
    assert result.shape == (10, 3, 3)

def test_sigma_plasticity_clamps_jacobian():
    fn = sigma_plasticity()
    F = _make_F_batch(5) * 2.0
    result = fn(F)
    J = jnp.linalg.det(result)
    assert jnp.all(J <= 1.21)
    assert jnp.all(J >= 0.04)

def test_plasticity_registry():
    assert "IdentityPlasticity" in PLASTICITY
    assert "DruckerPragerPlasticity" in PLASTICITY
    assert "VonMisesPlasticity" in PLASTICITY
    assert "SigmaPlasticity" in PLASTICITY

def test_get_constitutive_elasticity():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4})
    fn = get_constitutive(cfg)
    F = _make_F_batch(5)
    stress = fn(F)
    assert stress.shape == (5, 3, 3)

def test_get_constitutive_plasticity():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"name": "IdentityPlasticity"})
    fn = get_constitutive(cfg)
    F = _make_F_batch(5)
    assert jnp.allclose(fn(F), F)
