"""Pure-JAX backend implementation."""

from hydra_zen import store

from mpm_jax.p2g.backends.common import P2GBackend, jax_scan_p2g


@store(name="jax", group="backend", num_grids="${sim.num_grids}")
class JaxBackend(P2GBackend):
    """JAX/XLA baseline: identity order, JAX scan P2G, shared MLS-MPM G2P."""

    name = "jax"

    def scatter(self, params, prepared):
        return jax_scan_p2g(params, prepared)
