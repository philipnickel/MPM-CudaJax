"""cuTile P2G backend implementation."""

from hydra_zen import store
import jax.numpy as jnp

from mpm_jax.backends.common import BaseBackend, supercell_order


def _cutile_module():
    from mpm_jax import cutile_p2g

    return cutile_p2g


@store(name="cutile_v1", group="backend", num_grids="${sim.num_grids}")
class CutileV1Backend(BaseBackend):
    """cuTile direct 27-stencil scatter with global atomics."""

    name = "cutile_v1"

    def __init__(self, num_grids=None):
        cutile = _cutile_module()
        self.kernel = cutile.cutile_p2g_v1
        super().__init__(num_grids=num_grids)

    def p2g(self, params, prepared):
        return self.kernel(
            prepared.x,
            prepared.v,
            prepared.C,
            prepared.stress,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
        )


@store(name="cutile_v2", group="backend", num_grids="${sim.num_grids}")
class CutileV2Backend(BaseBackend):
    """cuTile arena scatter; occupancy is left to the cuTile compiler default."""

    name = "cutile_v2"

    def __init__(self, num_grids=None):
        cutile = _cutile_module()
        self.super_cell = cutile.ARENA_SC
        self.kernel = cutile.cutile_p2g_v2
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        prepared = supercell_order(params, state, stress, self.super_cell)
        grids_per_super_cell = params.num_grids // self.super_cell
        starts = prepared.cell_start[:-1].reshape(
            (grids_per_super_cell, grids_per_super_cell, grids_per_super_cell)
        )
        ends = prepared.cell_start[1:].reshape(
            (grids_per_super_cell, grids_per_super_cell, grids_per_super_cell)
        )
        return prepared._replace(cell_start=jnp.stack((starts, ends), axis=-1))

    def p2g(self, params, prepared):
        return self.kernel(
            prepared.x,
            prepared.v,
            prepared.C,
            prepared.stress,
            prepared.cell_start,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
        )

    def grid_divisor(self):
        return self.super_cell
