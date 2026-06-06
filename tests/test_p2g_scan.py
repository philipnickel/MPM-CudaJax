"""The JAX baseline P2G (_p2g_scan): lax.scan over the 27 stencil offsets."""

import jax.numpy as jnp

from mpm_jax.p2g.scan import _p2g_scan
from mpm_jax.p2g.stencil import flat_grid_index


def test_p2g_scan_shapes_and_mass_are_finite():
    n = 8
    G = 8
    x = jnp.ones((n, 3), dtype=jnp.float32) * 0.5
    v = jnp.zeros((n, 3), dtype=jnp.float32)
    C = jnp.zeros((n, 3, 3), dtype=jnp.float32)
    stress = jnp.zeros((n, 3, 3), dtype=jnp.float32)

    grid_mv, grid_m = _p2g_scan(
        x,
        v,
        C,
        stress,
        dt=3e-4,
        vol=1.0 / n,
        p_mass=1.0 / n,
        dx=1.0 / G,
        inv_dx=float(G),
        num_grids=G,
    )

    assert grid_mv.shape == (G**3, 3)
    assert grid_m.shape == (G**3,)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))
    assert jnp.isclose(grid_m.sum(), 1.0, atol=1e-5)


def test_flat_grid_index_preserves_flat_clip_boundary_semantics():
    G = 4
    base = jnp.array(
        [
            [-1, G - 1, G - 1],
            [G, 0, 0],
        ],
        dtype=jnp.int32,
    )
    offset = jnp.zeros((3,), dtype=jnp.int32)

    idx = flat_grid_index(base, offset, G)

    assert idx[0] == 0
    assert idx[1] == G**3 - 1
