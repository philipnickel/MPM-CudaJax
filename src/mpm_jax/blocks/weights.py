import jax.numpy as jnp

# The 27 integer offsets of the 3x3x3 quadratic B-spline support, enumerated in
# lexicographic (i, j, k) order over {0,1,2}^3. The order is load-bearing: the
# lax.scan and per-axis weight indexing in p2g_scan.py / g2p_scan.py depend on
# it. (The dense per-particle weight/index helper that used to live here was
# removed once both P2G and G2P moved to inline per-stencil recompute.)
OFFSET_27 = jnp.indices((3, 3, 3)).reshape(3, -1).T.astype(jnp.int32)  # (27, 3)
