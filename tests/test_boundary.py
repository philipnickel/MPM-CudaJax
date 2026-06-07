from types import SimpleNamespace

import jax.numpy as jnp

from mpm_jax.constitutive import stvk_elasticity_jacobi
from mpm_jax.grid import build_grid_x
from mpm_jax.p2g.backends import JaxBackend
from mpm_jax.solver import MPMSolver, RuntimeConfig


def _solver(num_grids):
    sim = SimpleNamespace(
        n_particles=8,
        num_grids=num_grids,
        dt=3e-4,
        steps_per_frame=1,
        clip_bound=0.5,
        damping=1.0,
        gravity=[0.0, 0.0, 0.0],
        rho=1000.0,
        size=[0.5, 0.5, 0.5],
        initial_velocity=[0.0, 0.0, 0.0],
        center=[0.5, 0.5, 0.5],
    )
    material = SimpleNamespace(elasticity=stvk_elasticity_jacobi())
    return MPMSolver(RuntimeConfig(material=material, sim=sim, backend=JaxBackend()))


def test_sticky_boundary_zeroes_velocity_below_surface():
    num_grids = 5
    solver = _solver(num_grids)
    grid_x = build_grid_x(num_grids)
    grid_mv = jnp.ones((num_grids**3, 3))
    result = solver._stages.grid_update(grid_mv, jnp.ones((num_grids**3,)))

    below = grid_x[:, 2] * solver.params.dx < 0.02
    assert jnp.allclose(result[below], 0.0)
    assert jnp.allclose(result[~below], 1.0)


def test_sticky_boundary_keeps_above_floor_normalized_velocity():
    num_grids = 5
    solver = _solver(num_grids)
    grid_x = build_grid_x(num_grids)
    grid_mv = jnp.ones((num_grids**3, 3)) * 4.0
    grid_m = jnp.ones((num_grids**3,)) * 2.0
    result = solver._stages.grid_update(grid_mv, grid_m)

    above = grid_x[:, 2] * solver.params.dx >= 0.02
    assert jnp.allclose(result[above], 2.0)
