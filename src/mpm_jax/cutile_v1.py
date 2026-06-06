"""Direct cuTile P2G scatter used as a comparison backend."""

import jax.numpy as jnp

import cuda.tile as ct
from cuda.tile.jax import InputOutput, cutile_call

from mpm_jax.cutile_common import _node_contribution, _node_xyz


DIRECT_STENCIL_TILE = 32
DIRECT_PARTICLE_TILE = 16


def _load_particle_chunk(x, v, C, stress, start, stop, inv_dx, particle_tile):
    x_p = (
        ct.load(
            x.slice(0, start, stop),
            (0, 0),
            (particle_tile, 4),
            padding_mode=ct.PaddingMode.ZERO,
        )
        * inv_dx
    )
    v_p = ct.load(
        v.slice(0, start, stop),
        (0, 0),
        (particle_tile, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    C_p = ct.load(
        C.slice(0, start, stop),
        (0, 0, 0),
        (particle_tile, 4, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    stress_p = ct.load(
        stress.slice(0, start, stop),
        (0, 0, 0),
        (particle_tile, 4, 4),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b = ct.astype(ct.floor(x_p - 0.5), ct.int32)
    return b, x_p - ct.astype(b, ct.float32), v_p, C_p, stress_p


@ct.kernel
def _p2g_direct_kernel(
    x,
    v,
    C,
    stress,
    grid,
    n_particles: ct.Constant[int],
    G: ct.Constant[int],
    dt: ct.Constant[float],
    vol: ct.Constant[float],
    p_mass: ct.Constant[float],
    inv_dx: ct.Constant[float],
    dx: ct.Constant[float],
    particle_tile: ct.Constant[int],
    stencil_tile: ct.Constant[int],
):
    block = ct.bid(0)
    p_lane = ct.arange(particle_tile, dtype=ct.int32)
    chunk_start = block * particle_tile
    p = chunk_start + p_lane
    in_bounds = p < n_particles
    particles = _load_particle_chunk(
        x,
        v,
        C,
        stress,
        chunk_start,
        n_particles,
        inv_dx,
        particle_tile,
    )

    active_lane = ct.reshape(in_bounds, (particle_tile, 1))
    offset = ct.reshape(ct.arange(stencil_tile, dtype=ct.int32), (1, stencil_tile))
    oi = offset // 9
    oj = (offset // 3) % 3
    ok = offset % 3
    oi_vec = ct.reshape(oi, (1, stencil_tile, 1))
    oj_vec = ct.reshape(oj, (1, stencil_tile, 1))
    ok_vec = ct.reshape(ok, (1, stencil_tile, 1))

    b = particles[0]
    base = ct.reshape(b, (particle_tile, 1, 4))
    node = _node_xyz(
        ct.extract(base, (0, 0, 0), (particle_tile, 1, 1)) + oi_vec,
        ct.extract(base, (0, 0, 1), (particle_tile, 1, 1)) + oj_vec,
        ct.extract(base, (0, 0, 2), (particle_tile, 1, 1)) + ok_vec,
        (particle_tile, stencil_tile, 4),
    )
    contrib, contributes = _node_contribution(
        particles,
        node,
        dt,
        vol,
        p_mass,
        inv_dx,
        dx,
        particle_tile,
        stencil_tile,
    )

    valid_stencil = offset < 27
    mask = contributes & active_lane & valid_stencil
    flat = (
        ct.extract(node, (0, 0, 0), (particle_tile, stencil_tile, 1))
        * (G * G)
        + ct.extract(node, (0, 0, 1), (particle_tile, stencil_tile, 1)) * G
        + ct.extract(node, (0, 0, 2), (particle_tile, stencil_tile, 1))
    )
    flat = ct.reshape(flat, (particle_tile, stencil_tile))
    flat = ct.maximum(0, ct.minimum(flat, G * G * G - 1))

    channel = ct.reshape(ct.arange(4, dtype=ct.int32), (1, 1, 4))
    flat = ct.reshape(flat, (particle_tile, stencil_tile, 1))
    mask = ct.reshape(mask, (particle_tile, stencil_tile, 1))
    ct.atomic_add(grid, (flat, channel), ct.where(mask, contrib, 0.0))


def cutile_p2g_v1(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Direct cuTile scatter for comparison with the arena backend."""
    n = int(x.shape[0])
    g = int(num_grids)
    g3 = g**3
    blocks = (n + DIRECT_PARTICLE_TILE - 1) // DIRECT_PARTICLE_TILE

    grid = jnp.zeros((g3, 4), dtype=jnp.float32)
    grid = cutile_call(
        (blocks,),
        _p2g_direct_kernel,
        (
            x,
            v,
            C,
            stress,
            InputOutput(grid),
            n,
            g,
            float(dt),
            float(vol),
            float(p_mass),
            float(inv_dx),
            float(dx),
            DIRECT_PARTICLE_TILE,
            DIRECT_STENCIL_TILE,
        ),
    )
    return grid[:, :3], grid[:, 3]
