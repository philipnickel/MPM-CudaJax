"""cuTile P2G backend implementation."""

from hydra_zen import store

from mpm_jax.backends.common import BaseBackend, supercell_order


def load_cutile_kernels():
    import mpm_jax.cutile_p2g  # noqa: F401

    return None


def arena_super_cell():
    from mpm_jax.cutile_p2g import ARENA_SC

    return ARENA_SC


@store(name="cutile", group="backend", num_grids="${sim.num_grids}")
class CutileBackend(BaseBackend):
    """cuTile arena scatter; occupancy is left to the cuTile compiler default."""

    name = "cutile"

    def __init__(self, num_grids=None):
        load_cutile_kernels()
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        return supercell_order(params, state, stress, arena_super_cell())

    def p2g(self, params, prepared):
        from mpm_jax.cutile_p2g import cutile_p2g_atomic_tile

        return cutile_p2g_atomic_tile(
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
        return arena_super_cell()
