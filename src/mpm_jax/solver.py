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

    def __init__(self, params, *, elasticity_fn, plasticity_fn,
                 pre_fn, post_fn, backend, steps_per_frame, init_state,
                 **frame_opts):
        self.params = params
        self.steps_per_frame = steps_per_frame
        self._init_state = init_state
        self.state = init_state
        self.elasticity_fn = elasticity_fn
        self.plasticity_fn = plasticity_fn
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.backend = backend
        self.frame_opts = dict(frame_opts)
        self._frame = build_backend_frame(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
            backend, steps_per_frame, **frame_opts,
        )

    def _frame_for_iterations(self, iterations):
        if iterations is None or int(iterations) == int(self.steps_per_frame):
            return self._frame
        return build_backend_frame(
            self.params, self.elasticity_fn, self.plasticity_fn,
            self.pre_fn, self.post_fn, self.backend, int(iterations), **self.frame_opts,
        )

    def stepped(self, iterations=None):
        """Return a new solver with state advanced by ``iterations`` substeps."""
        new_state = self._frame_for_iterations(iterations)(self.state)
        new = copy.copy(self)          # shallow: shares _frame + backend + closures
        new.state = new_state
        return new

    def step(self, iterations=None):
        """Advance this solver in place and return the new state."""
        self.state = self.stepped(iterations).state
        return self.state

    def solve(self, num_frames, on_frame=None):
        for f in range(num_frames):
            self.step()
            if on_frame is not None:
                on_frame(f, self.state)
        return self.state

    def reset(self, init_state):
        self.state = init_state
        return self.state

    def reset_to_initial(self):
        self.state = self._init_state
        return self.state
