"""cuTile P2G backend implementation."""

from hydra_zen import store

from mpm_jax.p2g.backends.common import P2GBackend, home_cell_order


def _cutile_module():
    from mpm_jax.p2g import cutile

    return cutile


@store(name="cutile_v3", group="backend", num_grids="${sim.num_grids}")
class CutileV3Backend(P2GBackend):
    """cuTile one-home-cell scatter with a locally reduced 27-node stencil."""

    name = "cutile_v3"

    def __init__(self, num_grids=None):
        cutile = _cutile_module()
        self.kernel = cutile.cutile_p2g_v3
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        return home_cell_order(params, state, stress)

    def scatter(self, params, prepared):
        return self.kernel(
            prepared.x,
            prepared.v,
            prepared.C,
            prepared.stress,
            prepared.bucket_bounds,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
        )
