"""Morton/Z-order sorting helpers for CUDA scatter locality."""

import jax.numpy as jnp


def _spread_bits(x):
    """Spread the lowest 10 bits of ``x`` into Morton bit positions."""
    x = x & jnp.uint32(0x000003FF)  # keep low 10 bits
    x = (x | (x << jnp.uint32(16))) & jnp.uint32(0x030000FF)
    x = (x | (x << jnp.uint32(8))) & jnp.uint32(0x0300F00F)
    x = (x | (x << jnp.uint32(4))) & jnp.uint32(0x030C30C3)
    x = (x | (x << jnp.uint32(2))) & jnp.uint32(0x09249249)
    return x


def morton_code_3d(cx, cy, cz):
    """3D Morton code (Z-order curve) from per-axis cell coords."""
    return (
        _spread_bits(cz)
        | (_spread_bits(cy) << jnp.uint32(1))
        | (_spread_bits(cx) << jnp.uint32(2))
    )


def morton_argsort(x, inv_dx, num_grids):
    """Argsort particle indices by 3D Morton code of their cell."""
    cells = jnp.clip(jnp.floor(x * inv_dx).astype(jnp.int32), 0, num_grids - 1)
    cx = cells[:, 0].astype(jnp.uint32)
    cy = cells[:, 1].astype(jnp.uint32)
    cz = cells[:, 2].astype(jnp.uint32)
    codes = morton_code_3d(cx, cy, cz)
    return jnp.argsort(codes)


def home_super_cell_id(x, inv_dx, num_grids, super_cell_width):
    """Super-cell id for the quadratic B-spline home node."""
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = jnp.clip(base + 1, 0, num_grids - 1)
    super_grids = num_grids // super_cell_width
    si = home[:, 0] // super_cell_width
    sj = home[:, 1] // super_cell_width
    sk = home[:, 2] // super_cell_width
    return (si * (super_grids * super_grids) + sj * super_grids + sk).astype(jnp.int32)


def home_cell_id(x, inv_dx, num_grids):
    """Cell id for the unclipped quadratic B-spline home node."""
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = jnp.clip(base + 1, 0, num_grids)
    cells_per_axis = num_grids + 1
    return (
        home[:, 0] * (cells_per_axis * cells_per_axis)
        + home[:, 1] * cells_per_axis
        + home[:, 2]
    ).astype(jnp.int32)
