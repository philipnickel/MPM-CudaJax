"""Pure JAX P2G — MLS-MPM formulation (Hu et al. 2018)."""
import functools
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27
from mpm_jax.constitutive import get_constitutive
from mpm_jax.weights import compute_weights_batch


def make_jax_p2g(cfg):
    elasticity_fn = get_constitutive(cfg.material.elasticity)
    plasticity_fn = get_constitutive(cfg.material.plasticity)

    @functools.partial(jax.jit, static_argnums=(1,))
    def _p2g(state, num_grids, inv_dx, dx, dt, vol, p_mass):
        F = plasticity_fn(state.F)
        stress = elasticity_fn(F)

        # MLS-MPM affine: -(dt*vol)*4*inv_dx^2 * stress + p_mass * C
        affine = -(dt * vol) * 4.0 * inv_dx * inv_dx * stress + p_mass * state.C

        weight, dpos, index = compute_weights_batch(state.x, inv_dx, dx, num_grids)

        # mv_i = weight_i * (p_mass * v + affine @ dpos_i)
        mv = weight[..., None] * (
            p_mass * state.v[:, None, :] + jnp.einsum('pij,pkj->pki', affine, dpos)
        )
        m = weight * p_mass

        grid_mv = jnp.zeros((num_grids**3, 3)).at[index.ravel()].add(mv.reshape(-1, 3))
        grid_m = jnp.zeros((num_grids**3,)).at[index.ravel()].add(m.ravel())
        return grid_mv, grid_m

    def p2g(state: MPMState, params: MPMParams):
        return _p2g(
            state, int(params.num_grids),
            params.inv_dx, params.dx, params.dt, params.vol, params.p_mass,
        )

    return p2g
