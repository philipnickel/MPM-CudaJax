import jax
import jax.numpy as jnp


def _single_particle_p2g(v_p, C_p, stress_p, weight, dweight, dpos, dt, vol, p_mass):
    """Compute P2G contributions for one particle (one CUDA thread).

    Args:
        v_p:      (3,)   particle velocity
        C_p:      (3, 3) APIC affine matrix
        stress_p: (3, 3) Kirchhoff stress
        weight:   (27,)  B-spline weights
        dweight:  (27, 3) weight gradients
        dpos:     (27, 3) particle-to-node offsets
        dt, vol, p_mass: scalars

    Returns:
        mv: (27, 3) momentum contribution per stencil node
        m:  (27,)   mass contribution per stencil node
    """
    # Affine momentum: stress term + APIC term
    mv = (
        -dt * vol * (stress_p @ dweight.T).T              # (27, 3)
        + p_mass * weight[:, None] * (v_p[None, :] + (C_p @ dpos.T).T)  # (27, 3)
    )
    m = weight * p_mass  # (27,)
    return mv, m


def p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass):
    """Per-particle P2G computation via vmap (embarrassingly parallel).

    Returns:
        mv: (N, 27, 3) momentum contributions per particle per stencil node
        m:  (N, 27)    mass contributions per particle per stencil node
    """
    return jax.vmap(
        _single_particle_p2g,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)


def p2g_scatter(mv, m, index, num_grids):
    """Scatter particle contributions onto the grid (the reduction).

    This is the only non-embarrassingly-parallel operation in the timestep.
    XLA lowers this to atomicAdd on GPU — the primary target for CUDA
    optimisation (shared memory staging, spatial sorting, warp reductions).

    Returns:
        grid_mv: (G^3, 3) grid momentum
        grid_m:  (G^3,)   grid mass
    """
    G = num_grids
    grid_mv = jnp.zeros((G ** 3, 3)).at[index.ravel()].add(mv.reshape(-1, 3))
    grid_m = jnp.zeros((G ** 3,)).at[index.ravel()].add(m.ravel())
    return grid_mv, grid_m


def p2g(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
    """Full P2G: compute + scatter. Drop-in compatible with existing interface."""
    mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
    return p2g_scatter(mv, m, index, num_grids)
