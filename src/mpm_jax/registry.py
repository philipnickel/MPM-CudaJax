import jax.numpy as jnp

from mpm_jax.solver import MPMSolver
from mpm_jax.backends import build_backend, KERNEL_NAMES
from mpm_jax.types import MPMState, make_params
from mpm_jax.blocks.grid import build_grid_x
from mpm_jax.blocks.init import get_particles
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns

__all__ = ["build_solver", "build_backend", "KERNEL_NAMES"]


def build_solver(cfg):
    """Construct the solver for ``cfg.kernel.name``.

    Reads the resolved Hydra config: builds particles, params, grid, BCs, and
    initial state, builds + validates the backend for the selected P2G via
    ``build_backend`` (availability and super-cell divisibility are checked there,
    at backend init), then instantiates ``MPMSolver``. Only the P2G varies across
    kernels — the config selects it by name; G2P/grid/loop are fixed.
    """
    name = cfg.kernel.name
    if name not in KERNEL_NAMES:
        raise KeyError(f"Unknown P2G kernel {name!r}. Available: {', '.join(KERNEL_NAMES)}.")

    sim, mat = cfg.sim, cfg.material
    num_grids = int(sim.num_grids)
    particles_np = get_particles(int(sim.n_particles),
                                 center=list(sim.center), size=list(sim.size))
    params = make_params(
        n_particles=int(sim.n_particles), num_grids=num_grids, dt=float(sim.dt),
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

    backend = build_backend(
        name, num_grids, autotune=bool(cfg.kernel.get("autotune", True)))

    frame_opts = {}
    if "loop_kind" in cfg.kernel:
        frame_opts["loop_kind"] = cfg.kernel["loop_kind"]
    phase_barriers = bool(cfg.get("profile", {}).get("barriers", False))
    return MPMSolver(
        params, elasticity_fn=elasticity_fn, plasticity_fn=plasticity_fn,
        pre_fn=pre_fn, post_fn=post_fn, backend=backend,
        steps_per_frame=int(sim.steps_per_frame), init_state=init,
        **frame_opts, phase_barriers=phase_barriers,
    )
