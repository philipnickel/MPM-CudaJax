"""MPM state and parameter definitions."""
from typing import NamedTuple
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


# 27 stencil offsets for 3x3x3 neighborhood
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
