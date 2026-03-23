from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import numpy as np

class MPMState(NamedTuple):
    x: jax.Array      # (N, 3) positions
    v: jax.Array      # (N, 3) velocities
    C: jax.Array      # (N, 3, 3) APIC affine matrix
    F: jax.Array      # (N, 3, 3) deformation gradient

class MPMParams(NamedTuple):
    num_grids: int
    dt: float
    gravity: jax.Array
    dx: float
    inv_dx: float
    clip_bound: float
    damping: float
    vol: float
    p_mass: float
    n_particles: int

# Precomputed 27 offsets for 3x3x3 neighborhood
OFFSET_27 = jnp.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float32,
)  # (27, 3)

def make_params(
    n_particles: int,
    num_grids: int = 25,
    dt: float = 3e-4,
    gravity: list = [0.0, 0.0, -9.8],
    rho: float = 1000.0,
    clip_bound: float = 0.5,
    damping: float = 1.0,
    center: list = [0.5, 0.5, 0.5],
    size: list = [1.0, 1.0, 1.0],
) -> MPMParams:
    dx = 1.0 / num_grids
    vol = float(np.prod(size)) / n_particles
    return MPMParams(
        num_grids=num_grids,
        dt=dt,
        gravity=jnp.array(gravity, dtype=jnp.float32),
        dx=dx,
        inv_dx=float(num_grids),
        clip_bound=clip_bound * dx,
        damping=damping,
        vol=vol,
        p_mass=rho * vol,
        n_particles=n_particles,
    )


def _compute_weights(fx):
    """Compute quadratic B-spline weights and weight gradients.

    Args:
        fx: (N, 3) fractional position within cell

    Returns:
        w: (N, 3, 3) — axis 0=particle, axis 1=node offset (0,1,2), axis 2=spatial dim (x,y,z)
        dw: (N, 3, 3) — same layout for weight gradients
    """
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    w = jnp.stack([w0, w1, w2], axis=1)

    dw0 = fx - 1.5
    dw1 = -2.0 * (fx - 1.0)
    dw2 = fx - 0.5
    dw = jnp.stack([dw0, dw1, dw2], axis=1)

    return w, dw


def _linearize_index(base, offset, num_grids):
    """Convert 3D grid indices to flat 1D indices."""
    idx = base[:, None, :] + offset[None, :, :].astype(int)  # (N, 27, 3)
    flat = idx[:, :, 0] * num_grids * num_grids + idx[:, :, 1] * num_grids + idx[:, :, 2]
    flat = jnp.clip(flat, 0, num_grids ** 3 - 1)
    return flat.reshape(-1)  # (N*27,)


# ---------------------------------------------------------------------------
# Individually JIT-able kernels
# ---------------------------------------------------------------------------

def compute_weights_and_indices(x, inv_dx, dx, num_grids):
    """Compute B-spline weights, weight gradients, dpos, and grid indices.

    Returns everything the P2G and G2P steps need, so weight computation
    is shared between them.
    """
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)

    w, dw = _compute_weights(fx)

    weight = jnp.einsum('bi,bj,bk->bijk', w[:, :, 0], w[:, :, 1], w[:, :, 2]).reshape(-1, 27)

    dweight = inv_dx * jnp.stack([
        jnp.einsum('bi,bj,bk->bijk', dw[:, :, 0],  w[:, :, 1],  w[:, :, 2]),
        jnp.einsum('bi,bj,bk->bijk',  w[:, :, 0], dw[:, :, 1],  w[:, :, 2]),
        jnp.einsum('bi,bj,bk->bijk',  w[:, :, 0],  w[:, :, 1], dw[:, :, 2]),
    ], axis=-1).reshape(-1, 27, 3)

    dpos = (OFFSET_27[None, :, :] - fx[:, None, :]) * dx
    index = _linearize_index(base, OFFSET_27, num_grids)

    return weight, dweight, dpos, index


def p2g(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
    """Particle-to-grid scatter. Returns (grid_mv, grid_m)."""
    G = num_grids
    mv = (
        -dt * vol * jnp.einsum('bij,bkj->bki', stress, dweight)
        + p_mass * weight[:, :, None] * (v[:, None, :] + jnp.einsum('bij,bkj->bki', C, dpos))
    )
    grid_mv = jnp.zeros((G ** 3, 3)).at[index].add(mv.reshape(-1, 3))
    grid_m = jnp.zeros((G ** 3,)).at[index].add((weight * p_mass).reshape(-1))
    return grid_mv, grid_m


@jax.jit
def grid_update(grid_mv, grid_m, gravity, dt, damping):
    """Normalize momentum by mass, apply gravity and damping."""
    valid = grid_m > 1e-15
    grid_mv = jnp.where(valid[:, None], grid_mv / grid_m[:, None], grid_mv)
    grid_mv = damping * (grid_mv + dt * gravity)
    return grid_mv


@jax.jit
def g2p(grid_mv, weight, dweight, dpos, index, F, x, dt, inv_dx, clip_bound):
    """Grid-to-particle gather. Returns new (x, v, C, F)."""
    gv = grid_mv[index].reshape(-1, 27, 3)
    new_v = (weight[:, :, None] * gv).sum(axis=1)
    new_C = 4.0 * inv_dx * inv_dx * (weight[:, :, None, None] * jnp.einsum('bij,bik->bijk', gv, dpos)).sum(axis=1)
    grad_v = jnp.einsum('bij,bik->bijk', gv, dweight).sum(axis=1)

    new_x = jnp.clip(x + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F + dt * grad_v @ F, -2.0, 2.0)

    return new_x, new_v, new_C, new_F


# ---------------------------------------------------------------------------
# Orchestrator (not jitted — calls jitted pieces + Python-level BCs)
# ---------------------------------------------------------------------------

def step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=None):
    """One full P2G2P step. Optionally accepts a custom p2g_fn to swap implementations."""
    dt = params.dt
    vol = params.vol
    p_mass = params.p_mass
    dx = params.dx
    inv_dx = params.inv_dx
    G = params.num_grids
    clip_bound = params.clip_bound

    # Pre-particle BCs (Python-level)
    x, v = pre_particle_fn(state.x, state.v, time)

    # Compute shared weights/indices
    weight, dweight, dpos, index = compute_weights_and_indices(x, inv_dx, dx, G)

    # P2G — swappable with CUDA kernel
    if p2g_fn is not None:
        grid_mv, grid_m = p2g_fn(v, state.C, stress, weight, dweight, dpos, index,
                                  dt, vol, p_mass, G)
    else:
        grid_mv, grid_m = p2g(v, state.C, stress, weight, dweight, dpos, index,
                              dt, vol, p_mass, G)

    # Grid update
    grid_mv = grid_update(grid_mv, grid_m, params.gravity, dt, params.damping)

    # Post-grid BCs (Python-level)
    grid_mv = post_grid_fn(grid_mv, grid_m, time)

    # G2P
    new_x, new_v, new_C, new_F = g2p(grid_mv, weight, dweight, dpos, index,
                                       state.F, x, dt, inv_dx, clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)


def simulate_frame(params, state, elasticity_fn, plasticity_fn, pre_particle_fn, post_grid_fn, steps_per_frame, time, p2g_fn=None):
    """Run multiple substeps for one frame."""
    for _ in range(steps_per_frame):
        stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=p2g_fn)
        state = state._replace(F=plasticity_fn(state.F))
        time += params.dt
    return state, time
