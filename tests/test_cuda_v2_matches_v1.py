"""cuda_v2 must match cuda_v1 on scatter output.

The final cuda_v2 contract sorts particles by home cell before warp-coalesced
global atomics. A full solver state can therefore have a different particle
order from cuda_v1; the grid scatter output is the intended equivalence target.
"""

from mpm_jax.p2g.backends import CudaV1Backend, CudaV2Backend
from mpm_jax.p2g.cuda.p2g_cuda import CudaV1P2G, CudaV2P2G
from tests.cuda_validation import (
    assert_grid_close,
    make_p2g_inputs,
    p2g_output,
    require_cuda_kernels,
)


def test_home_sorted_cuda_v2_scatter_matches_v1():
    require_cuda_kernels(CudaV1P2G, CudaV2P2G)
    params, state, stress = make_p2g_inputs(seed=0)

    v1 = p2g_output(CudaV1Backend(params.num_grids), params, state, stress)
    v2 = p2g_output(CudaV2Backend(params.num_grids), params, state, stress)

    assert_grid_close(v2, v1, atol=1e-5, rtol=1e-5)
