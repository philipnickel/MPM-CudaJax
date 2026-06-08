"""Shared P2G backend interface and ordering helpers."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from mpm_jax.g2p_scan import _g2p_scan_mls
from mpm_jax.p2g.scan import _p2g_scan
from mpm_jax.p2g.sort import home_cell_id, home_super_cell_id, morton_argsort


class PreparedP2G(NamedTuple):
    x: Any
    v: Any
    C: Any
    F: Any
    stress: Any
    bucket_start: Any = None
    bucket_bounds: Any = None


class P2GBackend:
    """Particle-to-grid strategy consumed by the shared solver frame."""

    name: str

    def __init__(self, num_grids=None):
        self.validate_num_grids(num_grids)

    def validate_num_grids(self, num_grids):
        divisor = self.grid_divisor()
        if divisor is not None and num_grids is not None and num_grids % divisor != 0:
            raise RuntimeError(
                f"{self.name} requires num_grids ({num_grids}) divisible by "
                f"{divisor}."
            )

    def prepare(self, params, state, stress):
        return PreparedP2G(state.x, state.v, state.C, state.F, stress)

    def scatter(self, params, prepared):
        raise NotImplementedError

    def grid_divisor(self):
        return None


def _bucket_start(bucket_id, bucket_count):
    counts = jnp.bincount(bucket_id, length=bucket_count).astype(jnp.int32)
    return jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(counts).astype(jnp.int32)]
    )


def _bucket_bounds(bucket_start, bucket_shape):
    starts = bucket_start[:-1].reshape(bucket_shape)
    ends = bucket_start[1:].reshape(bucket_shape)
    return jnp.stack((starts, ends), axis=-1)


def morton_order(params, state, stress):
    order = morton_argsort(state.x, params.inv_dx, params.num_grids)
    return PreparedP2G(
        state.x[order], state.v[order], state.C[order], state.F[order], stress[order]
    )


def supercell_order(params, state, stress, super_cell_width):
    bucket_id = home_super_cell_id(
        state.x, params.inv_dx, params.num_grids, super_cell_width
    )
    order = jnp.argsort(bucket_id)
    grids_per_super_cell = params.num_grids // super_cell_width
    return PreparedP2G(
        state.x[order],
        state.v[order],
        state.C[order],
        state.F[order],
        stress[order],
        bucket_start=_bucket_start(bucket_id, grids_per_super_cell**3),
    )


def home_cell_order(params, state, stress):
    bucket_id = home_cell_id(state.x, params.inv_dx, params.num_grids)
    order = jnp.argsort(bucket_id)
    cells_per_axis = params.num_grids + 1
    bucket_shape = (cells_per_axis, cells_per_axis, cells_per_axis)
    bucket_start = _bucket_start(bucket_id, cells_per_axis**3)
    return PreparedP2G(
        state.x[order],
        state.v[order],
        state.C[order],
        state.F[order],
        stress[order],
        bucket_bounds=_bucket_bounds(bucket_start, bucket_shape),
    )


def jax_scan_p2g(params, prepared):
    return _p2g_scan(
        prepared.x,
        prepared.v,
        prepared.C,
        prepared.stress,
        params.dt,
        params.vol,
        params.p_mass,
        params.dx,
        params.inv_dx,
        params.num_grids,
    )


def g2p_mls(params, prepared, grid_v):
    return _g2p_scan_mls(
        grid_v,
        prepared.x,
        prepared.F,
        params.dt,
        params.inv_dx,
        params.dx,
        params.num_grids,
        params.clip_bound,
    )
