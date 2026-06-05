from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class MPMState(NamedTuple):
    x: jax.Array      # (N, 3) positions
    v: jax.Array      # (N, 3) velocities
    C: jax.Array      # (N, 3, 3) APIC affine matrix
    F: jax.Array      # (N, 3, 3) deformation gradient


@dataclass(init=False)
class MPMParams:
    """Derived runtime constants for the compiled MPM frame."""

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

    def __init__(self, sim):
        self.num_grids = int(sim.num_grids)
        self.dt = float(sim.dt)
        self.gravity = jnp.asarray(list(sim.gravity), dtype=jnp.float32)
        self.dx = 1.0 / self.num_grids
        self.inv_dx = float(self.num_grids)
        self.clip_bound = float(sim.clip_bound) * self.dx
        self.damping = float(sim.damping)
        self.vol = float(np.prod(list(sim.size))) / int(sim.n_particles)
        self.p_mass = float(sim.rho) * self.vol
        self.n_particles = int(sim.n_particles)
