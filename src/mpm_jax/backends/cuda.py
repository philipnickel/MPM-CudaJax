"""Hand-written CUDA P2G backend implementations."""

from hydra_zen import store

from mpm_jax.backends.common import BaseBackend, morton_order, supercell_order
from mpm_jax.cuda.p2g_cuda import (
    SUPPORTED_SC,
    V4_SUPER_CELL_WIDTH,
    cuda_p2g_inline,
    cuda_p2g_v2_inline,
    cuda_p2g_v3_inline,
    cuda_p2g_v4_inline,
    register_p2g_inline,
    register_p2g_v2_inline,
    register_p2g_v3_inline,
    register_p2g_v4_inline,
)


def _call_inline_p2g(kernel, params, prepared):
    return kernel(
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


class CudaInlineBackend(BaseBackend):
    """One CUDA inline-scatter launch."""

    def __init__(self, num_grids=None):
        self.register_kernel()
        super().__init__(num_grids=num_grids)

    def register_kernel(self):
        raise NotImplementedError

    def p2g(self, params, prepared):
        raise NotImplementedError


@store(name="cuda_v1", group="backend", num_grids="${sim.num_grids}")
class CudaV1Backend(CudaInlineBackend):
    name = "cuda_v1"

    def register_kernel(self):
        register_p2g_inline()

    def p2g(self, params, prepared):
        return _call_inline_p2g(cuda_p2g_inline, params, prepared)


@store(name="cuda_v2", group="backend", num_grids="${sim.num_grids}")
class CudaV2Backend(CudaInlineBackend):
    name = "cuda_v2"

    def register_kernel(self):
        register_p2g_v2_inline()

    def p2g(self, params, prepared):
        return _call_inline_p2g(cuda_p2g_v2_inline, params, prepared)


@store(name="cuda_v3", group="backend", num_grids="${sim.num_grids}")
class CudaV3Backend(CudaInlineBackend):
    name = "cuda_v3"

    def prepare(self, params, state, stress):
        return morton_order(params, state, stress)

    def register_kernel(self):
        register_p2g_v3_inline()

    def p2g(self, params, prepared):
        return _call_inline_p2g(cuda_p2g_v3_inline, params, prepared)


@store(
    name="cuda_v4",
    group="backend",
    num_grids="${sim.num_grids}",
    super_cell_width=4,
)
class CudaV4Backend(BaseBackend):
    name = "cuda_v4"

    def __init__(self, num_grids=None, super_cell_width=None):
        super_cell = (
            V4_SUPER_CELL_WIDTH if super_cell_width is None else int(super_cell_width)
        )
        if super_cell not in SUPPORTED_SC:
            raise ValueError(
                f"cuda_v4 super_cell_width={super_cell} is not a compiled "
                f"instantiation; the kernel is built for {SUPPORTED_SC}."
            )
        self.super_cell = super_cell
        super().__init__(num_grids=num_grids)
        register_p2g_v4_inline()

    def prepare(self, params, state, stress):
        return supercell_order(params, state, stress, self.super_cell)

    def p2g(self, params, prepared):
        return cuda_p2g_v4_inline(
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
            super_cell=self.super_cell,
        )

    def grid_divisor(self):
        return self.super_cell
