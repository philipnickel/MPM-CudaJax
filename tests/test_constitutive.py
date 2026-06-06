from pathlib import Path

import hydra
import jax.numpy as jnp
from omegaconf import OmegaConf

from mpm_jax.constitutive import stvk_elasticity_jacobi


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


def test_jelly_material_config_builds_constitutive():
    cfg = OmegaConf.load(Path("conf/material/jelly.yaml"))
    fn = hydra.utils.instantiate(cfg.elasticity)
    stress = fn(_make_F_batch(5))
    assert stress.shape == (5, 3, 3)
    assert jnp.allclose(stress, 0.0, atol=1e-3)
