"""Pure JAX P2G using MLS-MPM formulation (Hu et al. 2018).

P2G scatter (no dweight needed):
    Dinv = 4 * inv_dx^2
    stress = -(dt * vol) * Dinv * Kirchhoff_stress
    affine = stress + p_mass * C
    mv_i = weight_i * (p_mass * v + affine @ dpos_i)

dpos is in real space: (offset - fx) * dx.
"""
import functools
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27
from mpm_jax.constitutive import get_constitutive


def make_jax_p2g(cfg):
    elasticity_fn = get_constitutive(cfg.material.elasticity)
    plasticity_fn = get_constitutive(cfg.material.plasticity)

    def _single_particle_weights(x_p, inv_dx, dx, num_grids):
        px = x_p * inv_dx
        base = jnp.floor(px - 0.5).astype(int)
        fx = px - base.astype(jnp.float32)
        w = jnp.stack([0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2])
        offsets = OFFSET_27.astype(int)
        weight = w[offsets[:,0],0] * w[offsets[:,1],1] * w[offsets[:,2],2]
        dpos = (OFFSET_27 - fx[None,:]) * dx
        idx_3d = base[None,:] + offsets
        index = idx_3d[:,0]*num_grids*num_grids + idx_3d[:,1]*num_grids + idx_3d[:,2]
        index = jnp.clip(index, 0, num_grids**3 - 1)
        return weight, dpos, index

    def _single_particle_scatter(v_p, affine_p, weight, dpos, p_mass):
        mv = weight[:, None] * (p_mass * v_p[None, :] + (affine_p @ dpos.T).T)
        m = weight * p_mass
        return mv, m

    compute_weights = jax.vmap(_single_particle_weights, in_axes=(0, None, None, None))
    scatter_vmap = jax.vmap(_single_particle_scatter, in_axes=(0, 0, 0, 0, None))

    @functools.partial(jax.jit, static_argnums=(1,))
    def _p2g_jitted(state, num_grids, inv_dx, dx, dt, vol, p_mass):
        F = plasticity_fn(state.F)
        stress = elasticity_fn(F)  # Kirchhoff stress (N, 3, 3)

        # MLS-MPM: stress_term = -(dt * vol) * Dinv * stress
        Dinv = 4.0 * inv_dx * inv_dx
        stress_term = -(dt * vol) * Dinv * stress

        # affine = stress_term + p_mass * C
        affine = stress_term + p_mass * state.C

        weight, dpos, index = compute_weights(state.x, inv_dx, dx, num_grids)
        mv, m = scatter_vmap(state.v, affine, weight, dpos, p_mass)

        grid_mv = jnp.zeros((num_grids**3, 3)).at[index.ravel()].add(mv.reshape(-1, 3))
        grid_m = jnp.zeros((num_grids**3,)).at[index.ravel()].add(m.ravel())
        return grid_mv, grid_m

    def p2g(state: MPMState, params: MPMParams):
        return _p2g_jitted(
            state, int(params.num_grids),
            params.inv_dx, params.dx, params.dt, params.vol, params.p_mass,
        )

    return p2g
