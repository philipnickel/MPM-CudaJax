from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable

import jax.numpy as jnp

from mpm_jax.solver import MPMSolver
from mpm_jax.backends import (
    jax_v1_5_backend,
    cuda_v1_backend,
    cuda_v2_backend,
    cuda_v3_backend,
    cuda_v4_backend,
    warp_v3_supercell_backend,
)
from mpm_jax.types import MPMState, make_params
from mpm_jax.blocks.grid import build_grid_x
from mpm_jax.blocks.init import get_particles
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns


@dataclass(frozen=True)
class KernelSpec:
    solver_cls: type
    backend_factory: Callable
    defaults: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


KERNELS = {
    "jax_v1_5":               KernelSpec(MPMSolver, jax_v1_5_backend),
    "cuda_v1_inline":         KernelSpec(MPMSolver, cuda_v1_backend),
    "cuda_v2_inline":         KernelSpec(MPMSolver, cuda_v2_backend, MappingProxyType({"loop_kind": "fori"})),
    "cuda_v3_inline":         KernelSpec(MPMSolver, cuda_v3_backend, MappingProxyType({"loop_kind": "fori", "cuda_graph": False})),
    "cuda_v4_inline":         KernelSpec(MPMSolver, cuda_v4_backend),
    "warp_v3_supercell_tile": KernelSpec(MPMSolver, warp_v3_supercell_backend),
}

REMOVED_KERNELS = {
    "jax": "Use jax_v1_5 as the JAX baseline.",
    "cuda_v1": "Use cuda_v1_inline (scatter-only variant removed).",
    "cuda_v2": "Use cuda_v2_inline.",
    "cuda_v4": "Use cuda_v4_inline.",
    "cuda_fused": "Deprecated; use an inline kernel and profile=jax.",
    "cuda_v2_fori_inline": "Use kernel=cuda_v2_inline with loop_kind=fori.",
    "cuda_v3_fori_inline": "Use kernel=cuda_v3_inline with loop_kind=fori.",
    "cuda_v6_inline": "Use kernel=cuda_v3_inline with cuda_graph=true.",
    "warp_baseline_graph": "Pure-Warp solver removed; use warp_v3_supercell_tile for Warp-in-JAX comparisons.",
    "warp_bonus_graph": "Pure-Warp solver removed; use warp_v3_supercell_tile.",
    "warp_bonus_v2_graph": "Pure-Warp solver removed; use warp_v3_supercell_tile.",
}


def build_solver(cfg):
    """Construct the solver for cfg.kernel.name from the registry.

    Reads the resolved Hydra config: builds particles, params, grid, BCs, and
    initial state, then instantiates the registered solver class with the
    registered frame builder.
    """
    name = cfg.kernel.name
    if name in REMOVED_KERNELS:
        raise ValueError(f"kernel={name} removed. {REMOVED_KERNELS[name]}")
    spec = KERNELS[name]
    particles_np = get_particles(int(cfg.sim.n_particles),
                                 center=list(cfg.sim.center), size=list(cfg.sim.size))

    sim, mat = cfg.sim, cfg.material
    params = make_params(
        n_particles=int(sim.n_particles), num_grids=int(sim.num_grids), dt=float(sim.dt),
        gravity=list(sim.gravity), rho=float(sim.rho), clip_bound=float(sim.clip_bound),
        damping=float(sim.damping), center=list(sim.center), size=list(sim.size),
    )
    particles = jnp.array(particles_np, dtype=jnp.float32)
    grid_x = build_grid_x(params.num_grids)
    pre_fn, post_fn = build_boundary_fns(
        list(sim.boundary_conditions), grid_x, params.dx, particles, params.dt, params.p_mass)
    elasticity_fn = get_constitutive(mat.elasticity)
    plasticity_fn = get_constitutive(mat.plasticity)
    n = int(sim.n_particles)
    init = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity), dtype=jnp.float32), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )
    frame_opts = dict(spec.defaults)
    for k in ("loop_kind", "cuda_graph", "graph_mode"):
        if k in cfg.kernel:
            frame_opts[k] = cfg.kernel[k]
    backend = spec.backend_factory(
        num_grids=int(sim.num_grids),
        **frame_opts,
    )
    return spec.solver_cls(
        params, elasticity_fn=elasticity_fn, plasticity_fn=plasticity_fn,
        pre_fn=pre_fn, post_fn=post_fn, backend=backend,
        steps_per_frame=int(sim.steps_per_frame), init_state=init, **frame_opts,
    )
