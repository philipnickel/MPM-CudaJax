"""Shared backend interface and math helpers."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from mpm_jax.g2p_scan import _g2p_scan_mls
from mpm_jax.p2g_scan import _p2g_scan
from mpm_jax.sort import home_super_cell_id, morton_argsort


class PreparedSubstep(NamedTuple):
    x: Any
    v: Any
    C: Any
    F: Any
    stress: Any
    cell_start: Any = None


class BaseBackend:
    """Interface consumed by the shared JAX-owned frame loop."""

    name: str

    def __init__(self, num_grids=None):
        self.validate_num_grids(num_grids)

    def validate_num_grids(self, num_grids):
        divisor = self.grid_divisor()
        if divisor is not None and num_grids is not None and num_grids % divisor != 0:
            raise RuntimeError(
                f"{self.name} requires num_grids ({num_grids}) divisible by "
                f"super-cell width ({divisor})."
            )

    def prepare(self, params, state, stress):
        return identity_order(state, stress)

    def p2g(self, params, prepared):
        raise NotImplementedError

    def step(self, params, state, stress):
        prepared = self.prepare(params, state, stress)
        grid_mv, grid_m = self.p2g(params, prepared)
        return prepared, grid_mv, grid_m

    def g2p(self, params, prepared, grid_v):
        return jax_scan_g2p_mls(params, prepared, grid_v)

    def grid_divisor(self):
        return None


def identity_order(state, stress):
    return PreparedSubstep(state.x, state.v, state.C, state.F, stress)


def morton_order(params, state, stress):
    order = morton_argsort(state.x, params.inv_dx, params.num_grids)
    return PreparedSubstep(
        state.x[order], state.v[order], state.C[order], state.F[order], stress[order]
    )


def supercell_boundaries(params, super_cell_width):
    grids_per_super_cell = params.num_grids // super_cell_width
    return jnp.arange(grids_per_super_cell**3 + 1, dtype=jnp.int32)


def supercell_order(params, state, stress, super_cell_width):
    super_id = home_super_cell_id(
        state.x, params.inv_dx, params.num_grids, super_cell_width
    )
    order = jnp.argsort(super_id)
    cell_start = jnp.searchsorted(
        super_id[order], supercell_boundaries(params, super_cell_width)
    ).astype(jnp.int32)
    return PreparedSubstep(
        state.x[order],
        state.v[order],
        state.C[order],
        state.F[order],
        stress[order],
        cell_start=cell_start,
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


def jax_scan_g2p_mls(params, prepared, grid_v):
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
