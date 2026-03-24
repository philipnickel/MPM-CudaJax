"""Pure JAX P2G baseline implementation."""
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
        dw = jnp.stack([fx-1.5, -2.0*(fx-1.0), fx-0.5])
        dweight = inv_dx * jnp.stack([
            dw[offsets[:,0],0]*w[offsets[:,1],1]*w[offsets[:,2],2],
            w[offsets[:,0],0]*dw[offsets[:,1],1]*w[offsets[:,2],2],
            w[offsets[:,0],0]*w[offsets[:,1],1]*dw[offsets[:,2],2],
        ], axis=-1)
        dpos = (OFFSET_27 - fx[None,:]) * dx
        idx_3d = base[None,:] + offsets
        index = idx_3d[:,0]*num_grids*num_grids + idx_3d[:,1]*num_grids + idx_3d[:,2]
        index = jnp.clip(index, 0, num_grids**3 - 1)
        return weight, dweight, dpos, index

    def _single_particle_p2g(v_p, C_p, stress_p, weight, dweight, dpos, dt, vol, p_mass):
        mv = (-dt*vol*(stress_p @ dweight.T).T + p_mass*weight[:,None]*(v_p[None,:] + (C_p @ dpos.T).T))
        m = weight * p_mass
        return mv, m

    compute_weights = jax.vmap(_single_particle_weights, in_axes=(0, None, None, None))
    p2g_compute = jax.vmap(_single_particle_p2g, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))

    @functools.partial(jax.jit, static_argnums=(1,))
    def _p2g_jitted(state, num_grids, inv_dx, dx, dt, vol, p_mass):
        F = plasticity_fn(state.F)
        stress = elasticity_fn(F)
        weight, dweight, dpos, index = compute_weights(state.x, inv_dx, dx, num_grids)
        mv, m = p2g_compute(state.v, state.C, stress, weight, dweight, dpos, dt, vol, p_mass)
        grid_mv = jnp.zeros((num_grids**3, 3)).at[index.ravel()].add(mv.reshape(-1, 3))
        grid_m = jnp.zeros((num_grids**3,)).at[index.ravel()].add(m.ravel())
        return grid_mv, grid_m

    def p2g(state: MPMState, params: MPMParams):
        return _p2g_jitted(
            state, int(params.num_grids),
            params.inv_dx, params.dx, params.dt, params.vol, params.p_mass,
        )

    return p2g
