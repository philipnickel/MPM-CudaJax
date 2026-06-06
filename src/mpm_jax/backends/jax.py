"""Pure-JAX backend implementation."""

from hydra_zen import store

from mpm_jax.backends.common import BaseBackend, jax_scan_p2g


@store(name="jax", group="backend", num_grids="${sim.num_grids}")
class JaxBackend(BaseBackend):
    """JAX/XLA baseline: identity order, JAX scan P2G, shared MLS-MPM G2P."""

    name = "jax"

    def p2g(self, params, prepared):
        return jax_scan_p2g(params, prepared)
