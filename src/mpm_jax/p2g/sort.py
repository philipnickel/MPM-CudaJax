"""3D Morton (Z-order) coding for spatial sorting of MPM particles.

The intent: sorting particles by Morton code along the Z-order space-filling
curve clusters spatially close particles next to each other in memory. When
those particles are then processed by the same CUDA warp (32 lanes ->
32 consecutive after-sort particles), more warp lanes target the same
27-stencil grid node, so a warp-shuffle reduction (`__match_any_sync` +
`__shfl_sync`) coalesces more atomics into one. See p2g_v3.cu.

The bit-interleave handles up to 10 bits per axis (`num_grids` <= 1024),
which more than covers the project's typical G in [32, 256].
"""

import jax.numpy as jnp


def _spread_bits(x):
    """Spread the lowest 10 bits of ``x`` into bits 0, 3, 6, 9, ...

    The output sits in the low 30 bits of a uint32. The classic
    Magic-Number Method (cf. graphics gems / Hu 2018) — five mask-and-shift
    rounds map each bit ``i`` to position ``3*i``.
    """
    x = x & jnp.uint32(0x000003FF)  # keep 10 bits
    x = (x | (x << jnp.uint32(16))) & jnp.uint32(0x030000FF)
    x = (x | (x << jnp.uint32(8))) & jnp.uint32(0x0300F00F)
    x = (x | (x << jnp.uint32(4))) & jnp.uint32(0x030C30C3)
    x = (x | (x << jnp.uint32(2))) & jnp.uint32(0x09249249)
    return x


def morton_code_3d(cx, cy, cz):
    """3D Morton code (Z-order curve) from per-axis cell coords.

    Each input must be uint32 in [0, 1023]. Output is a uint32 where bits
    are interleaved as ``... z2 y2 x2 z1 y1 x1 z0 y0 x0`` from MSB to LSB.
    """
    return (
        _spread_bits(cz)
        | (_spread_bits(cy) << jnp.uint32(1))
        | (_spread_bits(cx) << jnp.uint32(2))
    )


# can this be done in a more native way?
def morton_argsort(x, inv_dx, num_grids):
    """Argsort particle indices by 3D Morton code of their cell.

    Args:
        x: (N, 3) float32 particle positions in [0, 1]
        inv_dx: scalar — 1 / grid spacing (= num_grids for unit-cube domain)
        num_grids: int — grid resolution along one axis

    Returns:
        (N,) int32 array of argsort indices.
    """
    cells = jnp.clip(jnp.floor(x * inv_dx).astype(jnp.int32), 0, num_grids - 1)
    cx = cells[:, 0].astype(jnp.uint32)
    cy = cells[:, 1].astype(jnp.uint32)
    cz = cells[:, 2].astype(jnp.uint32)
    codes = morton_code_3d(cx, cy, cz)
    return jnp.argsort(codes)


def home_super_cell_id(x, inv_dx, num_grids, super_cell_width):
    """Super-cell id for the quadratic B-spline home node.

    A particle's quadratic B-spline stencil is centered on ``floor(x / dx - 0.5) + 1``.
    CUDA v3 sorts particles by the super-cell containing that home node before
    building CSR-style ``bucket_start`` boundaries.
    """
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = jnp.clip(base + 1, 0, num_grids - 1)
    super_grids = num_grids // super_cell_width
    si = home[:, 0] // super_cell_width
    sj = home[:, 1] // super_cell_width
    sk = home[:, 2] // super_cell_width
    return (si * (super_grids * super_grids) + sj * super_grids + sk).astype(jnp.int32)


def home_cell_id(x, inv_dx, num_grids):
    """Cell id for the unclipped quadratic B-spline home node.

    The one-cell cuTile backend owns the exact ``base + {0,1,2}`` stencil, so it
    uses ``base + 1`` as the home cell and allows the upper boundary home ``G``.
    """
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = jnp.clip(base + 1, 0, num_grids)
    cells_per_axis = num_grids + 1
    return (
        home[:, 0] * (cells_per_axis * cells_per_axis)
        + home[:, 1] * cells_per_axis
        + home[:, 2]
    ).astype(jnp.int32)
