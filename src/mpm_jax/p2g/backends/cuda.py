"""Hand-written CUDA P2G backend implementations."""

from hydra_zen import store

from mpm_jax.p2g.backends.common import P2GBackend, home_cell_order
from mpm_jax.p2g.cuda.p2g_cuda import (
    CudaP2GKernel,
    CudaV1P2G,
    CudaV3P2G,
    CudaV4P2G,
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


@store(name="cuda_v3", group="backend", num_grids="${sim.num_grids}")
class CudaV3Backend(P2GBackend):
    name = "cuda_v3"

    def __init__(self, num_grids=None):
        self.kernel = CudaV3P2G()
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


@store(name="cuda_v4", group="backend", num_grids="${sim.num_grids}")
class CudaV4Backend(P2GBackend):
    name = "cuda_v4"

    def __init__(self, num_grids=None):
        self.kernel = CudaV4P2G()
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
