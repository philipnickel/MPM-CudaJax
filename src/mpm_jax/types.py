from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class MPMState(NamedTuple):
    x: jax.Array      # (N, 3) positions
    v: jax.Array      # (N, 3) velocities
    C: jax.Array      # (N, 3, 3) APIC affine matrix
    F: jax.Array      # (N, 3, 3) deformation gradient


class StepIntermediates(NamedTuple):
    """Minimal state carried from the P2G stage into the G2P stage.

    Only the post-BC positions and the F that should feed G2P are kept.
    B-spline weight / dweight / dpos / index tensors are NOT cached here -
    they would be (N, 27, *) and at large N (a few million particles)
    materialising them across the JIT boundary blows the GPU memory
    budget. G2P recomputes them from x_post_bc - the math is cheap
    (~50 flops/particle, no SVD) and the savings are ~1100 bytes/particle.
    """
    x_post_bc: jax.Array     # (N, 3) positions after pre-particle BCs
    F_pre_plast: jax.Array   # (N, 3, 3) F that G2P should use as its F_p input

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
