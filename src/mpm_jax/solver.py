import copy

# Re-exported through mpm_jax.__init__._SOLVER_EXPORTS (the lazy loader pulls
# these off this module), so keep them importable here even though the solver
# body no longer references them directly.
from mpm_jax.types import MPMState, MPMParams, make_params  # noqa: F401
from mpm_jax.backends import build_backend_frame


class MPMSolver:
    """Stateful driver over the functional JAX core.

    A plain Python object: array state (`state`) is mutated in place by the
    driver API, while the backend object, constitutive/boundary closures, and
    the compiled `_frame` are fixed for the solver's lifetime. The solver itself
    is never a JAX argument — only `state` (an `MPMState` pytree) is traced; the
    backend/fns are baked into `_frame`'s closure at build time. So no pytree
    machinery is needed here.
    """

    @classmethod
    def from_cfg(cls, cfg):
        """Build a solver from a resolved Hydra config (the config-aware constructor).

        Reads each config section and constructs the pieces — params (with its
        derived dx/vol/p_mass), particles, the name-selected + validated backend,
        the boundary closures, and the initial state — then hands them to
        ``__init__``. The shared scalars (``n_particles``/``num_grids``) are read
        once and threaded as locals (``n``/``g``); the backend's divisibility check
        needs ``num_grids``, so it gets ``g`` even though it lives under ``sim``.
        """
        import jax.numpy as jnp  # noqa: E402  # pylint: disable=import-outside-toplevel
        from mpm_jax.backends import build_backend  # pylint: disable=import-outside-toplevel
        from mpm_jax.blocks.grid import build_grid_x  # pylint: disable=import-outside-toplevel
        from mpm_jax.blocks.init import get_particles  # pylint: disable=import-outside-toplevel
        from mpm_jax.boundary import build_boundary_fns  # pylint: disable=import-outside-toplevel
        from mpm_jax.constitutive import get_constitutive  # pylint: disable=import-outside-toplevel

        sim, mat = cfg.sim, cfg.material
        n, g = int(sim.n_particles), int(sim.num_grids)

        params = make_params(
            n_particles=n, num_grids=g, dt=float(sim.dt),
            gravity=list(sim.gravity), rho=float(sim.rho), clip_bound=float(sim.clip_bound),
            damping=float(sim.damping), center=list(sim.center), size=list(sim.size),
        )
        particles = jnp.asarray(
            get_particles(n, center=list(sim.center), size=list(sim.size)), dtype=jnp.float32)
        backend = build_backend(cfg.p2g.name, g, autotune=bool(cfg.p2g.get("autotune", True)))
        pre_fn, post_fn = build_boundary_fns(
            list(sim.boundary_conditions), build_grid_x(g),
            params.dx, particles, params.dt, params.p_mass)
        init_state = MPMState(
            x=particles,
            v=jnp.broadcast_to(
                jnp.asarray(list(sim.initial_velocity), dtype=jnp.float32), (n, 3)).copy(),
            C=jnp.zeros((n, 3, 3)),
            F=jnp.tile(jnp.eye(3), (n, 1, 1)),
        )
        return cls(
            params, elasticity_fn=get_constitutive(mat.elasticity),
            plasticity_fn=get_constitutive(mat.plasticity),
            pre_fn=pre_fn, post_fn=post_fn, backend=backend,
            steps_per_frame=int(sim.steps_per_frame), init_state=init_state,
        )

    def __init__(self, params, *, elasticity_fn, plasticity_fn,
                 pre_fn, post_fn, backend, steps_per_frame, init_state):
        self.params = params
        self.steps_per_frame = steps_per_frame
        self._init_state = init_state
        self.state = init_state
        self.elasticity_fn = elasticity_fn
        self.plasticity_fn = plasticity_fn
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.backend = backend
        self._frame = build_backend_frame(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
            backend, steps_per_frame,
        )

    def stepped(self):
        """Return a new solver with state advanced one frame (steps_per_frame)."""
        new_state = self._frame(self.state)
        new = copy.copy(self)          # shallow: shares _frame + backend + closures
        new.state = new_state
        return new

    def step(self):
        """Advance this solver in place and return the new state."""
        self.state = self.stepped().state
        return self.state

    def solve(self, num_frames, on_frame=None):
        for f in range(num_frames):
            self.step()
            if on_frame is not None:
                on_frame(f, self.state)
        return self.state

    def reset_to_initial(self):
        self.state = self._init_state
        return self.state
