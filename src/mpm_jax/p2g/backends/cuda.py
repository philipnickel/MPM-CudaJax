"""Hand-written CUDA P2G backend implementations."""

from hydra_zen import store

from mpm_jax.p2g.backends.common import P2GBackend, morton_order, supercell_order
from mpm_jax.p2g.cuda.p2g_cuda import (
    SUPPORTED_SC,
    V3_SUPER_CELL_WIDTH,
    CudaP2GKernel,
    CudaV1P2G,
    CudaV2P2G,
    CudaV3P2G,
)


class CudaP2GBackend(P2GBackend):
    """Backend backed by a callable CUDA P2G FFI target."""

    kernel_type: type[CudaP2GKernel]

    def __init__(self, num_grids=None):
        self.kernel = self.kernel_type()
        super().__init__(num_grids=num_grids)

    def scatter(self, params, prepared):
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


@store(name="cuda_v1", group="backend", num_grids="${sim.num_grids}")
class CudaV1Backend(CudaP2GBackend):
    name = "cuda_v1"
    kernel_type = CudaV1P2G


@store(name="cuda_v2", group="backend", num_grids="${sim.num_grids}")
class CudaV2Backend(CudaP2GBackend):
    name = "cuda_v2"
    kernel_type = CudaV2P2G

    def prepare(self, params, state, stress):
        return morton_order(params, state, stress)


@store(
    name="cuda_v3",
    group="backend",
    num_grids="${sim.num_grids}",
    super_cell_width=4,
)
class CudaV3Backend(P2GBackend):
    name = "cuda_v3"

    def __init__(self, num_grids=None, super_cell_width=None):
        super_cell = (
            V3_SUPER_CELL_WIDTH if super_cell_width is None else int(super_cell_width)
        )
        if super_cell not in SUPPORTED_SC:
            raise ValueError(
                f"cuda_v3 super_cell_width={super_cell} is not a compiled "
                f"instantiation; the kernel is built for {SUPPORTED_SC}."
            )
        self.super_cell = super_cell
        self.kernel = CudaV3P2G(super_cell)
        super().__init__(num_grids=num_grids)

    def prepare(self, params, state, stress):
        return supercell_order(params, state, stress, self.super_cell)

    def scatter(self, params, prepared):
        return self.kernel(
            prepared.x,
            prepared.v,
            prepared.C,
            prepared.stress,
            prepared.bucket_start,
            params.num_grids,
            params.dt,
            params.vol,
            params.p_mass,
            params.inv_dx,
            params.dx,
        )

    def grid_divisor(self):
        return self.super_cell
