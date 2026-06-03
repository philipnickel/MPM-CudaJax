from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable

import jax.numpy as jnp

from mpm_jax.solver import MPMSolver, WarpGraphSolver
from mpm_jax.stepping.jax_frames import build_jax_v1_5_frame
from mpm_jax.stepping.cuda_frames import (
    build_cuda_v1_frame, build_cuda_v2_frame, build_cuda_v3_frame, build_cuda_v4_frame,
)
from mpm_jax.stepping.warp_graph_frame import build_warp_graph
from mpm_jax.types import MPMState, make_params
from mpm_jax.blocks.grid import build_grid_x
from mpm_jax.blocks.init import get_particles
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns


@dataclass(frozen=True)
class KernelSpec:
    solver_cls: type
    build_frame: Callable
    defaults: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))


# NOTE: for the two WarpGraphSolver entries, `build_frame` is NOT a frame builder
# with the (params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame, **opts)
# signature — it is `build_warp_graph(cfg, *, particles, **opts) -> WarpGraphSolver`.
# build_solver() (added later) MUST special-case `spec.solver_cls is WarpGraphSolver`.
KERNELS = {
    "jax_v1_5":               KernelSpec(MPMSolver, build_jax_v1_5_frame),
    "cuda_v1_inline":         KernelSpec(MPMSolver, build_cuda_v1_frame),
    "cuda_v2_inline":         KernelSpec(MPMSolver, build_cuda_v2_frame, MappingProxyType({"loop_kind": "fori"})),
    "cuda_v3_inline":         KernelSpec(MPMSolver, build_cuda_v3_frame, MappingProxyType({"loop_kind": "fori", "cuda_graph": False})),
    "cuda_v4_inline":         KernelSpec(MPMSolver, build_cuda_v4_frame),
    "warp_baseline_graph":    KernelSpec(WarpGraphSolver, build_warp_graph, MappingProxyType({"baseline": True})),
    "warp_bonus_graph":       KernelSpec(WarpGraphSolver, build_warp_graph),
    "warp_bonus_v2_graph":    KernelSpec(WarpGraphSolver, build_warp_graph, MappingProxyType({"indexed_sort": True})),
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
}


def build_solver(cfg):
    """Construct the solver for cfg.kernel.name from the registry.

    Reads the resolved Hydra config: builds particles, params, grid, BCs, and
    initial state, then instantiates the registered solver class with the
    registered frame builder. WarpGraphSolver entries are special-cased because
    their builder takes (cfg, *, particles) and returns the solver directly.
    """
    name = cfg.kernel.name
    if name in REMOVED_KERNELS:
        raise ValueError(f"kernel={name} removed. {REMOVED_KERNELS[name]}")
    spec = KERNELS[name]
    particles_np = get_particles(int(cfg.sim.n_particles),
                                 center=list(cfg.sim.center), size=list(cfg.sim.size))

    if spec.solver_cls is WarpGraphSolver:
        return spec.build_frame(cfg, particles=particles_np, **dict(spec.defaults))

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
    for k in ("loop_kind", "cuda_graph"):
        if k in cfg.kernel:
            frame_opts[k] = cfg.kernel[k]
    return spec.solver_cls(
        params, elasticity_fn=elasticity_fn, plasticity_fn=plasticity_fn,
        pre_fn=pre_fn, post_fn=post_fn, build_frame=spec.build_frame,
        steps_per_frame=int(sim.steps_per_frame), init_state=init, **frame_opts,
    )
