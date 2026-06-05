import jax.numpy as jnp


# 27 offsets for the 3x3x3 quadratic B-spline support around each particle.
# (The dense per-particle weight/index helper that used to live here was removed
# once both P2G and G2P moved to inline per-stencil recompute — only this
# constant is still shared, by p2g_scan.py and g2p_scan.py.)
OFFSET_27 = jnp.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float32,
)  # (27, 3)
