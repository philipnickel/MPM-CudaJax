"""jax_v1_5 P2G keeps the 27-stencil loop inside the outer frame JIT."""

import jax.numpy as jnp

from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.p2g_scan import _p2g_scan
from mpm_jax.p2g_vmap import _p2g_vmap


def test_p2g_scan_shapes_and_mass_are_finite():
    n = 8
    G = 8
    x = jnp.ones((n, 3), dtype=jnp.float32) * 0.5
    v = jnp.zeros((n, 3), dtype=jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    stress = jnp.zeros((n, 3, 3), dtype=jnp.float32)

    grid_mv, grid_m = _p2g_scan(
        x, v, C, stress,
        dt=3e-4,
        vol=1.0 / n,
        p_mass=1.0 / n,
        dx=1.0 / G,
        inv_dx=float(G),
        num_grids=G,
    )

    assert grid_mv.shape == (G ** 3, 3)
    assert grid_m.shape == (G ** 3,)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))
    assert jnp.isclose(grid_m.sum(), 1.0, atol=1e-5)


def test_p2g_vmap_matches_scan_baseline():
    n = 5
    G = 8
    x = jnp.array([
        [0.35, 0.35, 0.35],
        [0.42, 0.52, 0.46],
        [0.55, 0.44, 0.62],
        [0.63, 0.59, 0.40],
        [0.47, 0.66, 0.58],
    ], dtype=jnp.float32)
    v = jnp.array([
        [0.1, -0.2, 0.0],
        [0.0, 0.2, -0.1],
        [-0.1, 0.0, 0.3],
        [0.3, -0.1, 0.1],
        [-0.2, 0.1, -0.3],
    ], dtype=jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    stress = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    dt = 3e-4
    vol = 1.0 / n
    p_mass = 1.0 / n
    dx = 1.0 / G
    inv_dx = float(G)

    weight, dweight, dpos, index = compute_weights_and_indices(x, inv_dx, dx, G)
    scan_mv, scan_m = _p2g_scan(x, v, C, stress, dt, vol, p_mass, dx, inv_dx, G)
    vmap_mv, vmap_m = _p2g_vmap(
        weight, dweight, dpos, index, v, C, stress, dt, vol, p_mass, G)

    assert jnp.allclose(vmap_mv, scan_mv, atol=1e-6)
    assert jnp.allclose(vmap_m, scan_m, atol=1e-6)
