"""hydra-zen experiment: build the MPMSolver from config via `instantiate`.

This is the declarative alternative to the imperative `registry.build_solver`:
the construction graph is expressed as a tree of `hydra_zen.builds(...)` configs
whose interpolations (`${sim.*}`, `${material.*}`, `${p2g.name}`) resolve
against the composed Hydra config. `instantiate(cfg.solver)` then builds it.

Two tiny adapters bridge the parts plain `instantiate` can't express:
- `_boundary` reads `dx`/`p_mass` off the built `params` object (instantiate
  can't reference a sibling object's *attributes* via interpolation) and builds
  `grid_x` internally.
- `_initial_state` does the jnp conversion + zeros/eye that the raw `MPMState`
  constructor doesn't.
- `_assemble` unpacks the (pre_fn, post_fn) boundary tuple into MPMSolver's two
  kwargs (instantiate can't index into a returned tuple).
"""

import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, instantiate
from omegaconf import OmegaConf

from mpm_jax.backends import build_backend
from mpm_jax.blocks.grid import build_grid_x
from mpm_jax.blocks.init import get_particles
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.constitutive import get_constitutive
from mpm_jax.solver import MPMSolver
from mpm_jax.types import MPMState, make_params


# --- adapters: the imperative bits instantiate can't express declaratively ---
def _initial_state(particles, initial_velocity):
    x = jnp.asarray(particles, dtype=jnp.float32)
    n = x.shape[0]
    return MPMState(
        x=x,
        v=jnp.broadcast_to(jnp.asarray(initial_velocity, dtype=jnp.float32), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )


def _boundary(boundary_conditions, params, init_pos):
    return build_boundary_fns(
        boundary_conditions, build_grid_x(params.num_grids), params.dx,
        init_pos=init_pos, dt=params.dt, p_mass=params.p_mass,
    )


def _assemble(params, elasticity_fn, plasticity_fn, boundary, backend, init_state,
              steps_per_frame):
    pre_fn, post_fn = boundary
    return MPMSolver(
        params, elasticity_fn=elasticity_fn, plasticity_fn=plasticity_fn,
        pre_fn=pre_fn, post_fn=post_fn, backend=backend,
        steps_per_frame=steps_per_frame, init_state=init_state,
    )


# --- the declarative construction graph (interpolations resolve against cfg) ---
_b = lambda *a, **k: builds(*a, hydra_convert="all", **k)  # noqa: E731

Particles = _b(get_particles, n_particles="${sim.n_particles}",
               center="${sim.center}", size="${sim.size}")
Params = _b(make_params, n_particles="${sim.n_particles}", num_grids="${sim.num_grids}",
            dt="${sim.dt}", gravity="${sim.gravity}", rho="${sim.rho}",
            clip_bound="${sim.clip_bound}", damping="${sim.damping}",
            center="${sim.center}", size="${sim.size}")
Backend = _b(build_backend, name="${p2g.name}", num_grids="${sim.num_grids}",
             autotune="${oc.select:p2g.autotune,true}")
# constitutive expects a DictConfig (uses cfg.name) -> keep OmegaConf (no convert)
Elasticity = builds(get_constitutive, cfg="${material.elasticity}")
Plasticity = builds(get_constitutive, cfg="${material.plasticity}")
Boundary = _b(_boundary, boundary_conditions="${sim.boundary_conditions}",
              params=Params, init_pos=Particles)
State = _b(_initial_state, particles=Particles, initial_velocity="${sim.initial_velocity}")
Solver = _b(_assemble, params=Params, elasticity_fn=Elasticity, plasticity_fn=Plasticity,
            boundary=Boundary, backend=Backend, init_state=State,
            steps_per_frame="${sim.steps_per_frame}")


# Register the Solver graph in Hydra's ConfigStore under group "solver", so a
# Hydra app whose defaults list includes `- solver: default` gets `cfg.solver`
# and can `instantiate(cfg.solver)` directly. (Import this module before
# @hydra.main composes — simulate.py / profile_nsight.py do.)
ConfigStore.instance().store(group="solver", name="default", node=Solver)


def build_solver(cfg):
    """Build the MPMSolver from a resolved Hydra config via `instantiate`.

    If the config already carries the `solver` node (the Hydra apps, via the
    `- solver: default` default), instantiate it directly; otherwise compose the
    Solver graph onto the config first (programmatic / test callers).
    """
    if "solver" in cfg:
        return instantiate(cfg.solver)
    root = OmegaConf.create({
        "sim": cfg.sim, "material": cfg.material, "p2g": cfg.p2g, "solver": Solver,
    })
    return instantiate(root.solver)
