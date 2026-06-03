import equinox as eqx

from mpm_jax.types import (
    MPMState,
    MPMParams, make_params,  # noqa: F401  (re-exported via __init__._SOLVER_EXPORTS)
)
from mpm_jax.backends import build_backend_frame


class MPMSolver(eqx.Module):
    """Equinox shell over the functional JAX core.

    Backend choices (backend object, constitutive functions, boundary closures)
    are static fields. Array state is a dynamic PyTree leaf. This keeps one
    solver object as the public API while preserving JAX's requirement that
    backend functions are fixed at trace time.
    """

    params: MPMParams
    state: MPMState
    _init_state: MPMState
    steps_per_frame: int
    elasticity_fn: object = eqx.field(static=True)
    plasticity_fn: object = eqx.field(static=True)
    pre_fn: object = eqx.field(static=True)
    post_fn: object = eqx.field(static=True)
    backend: object = eqx.field(static=True)
    frame_opts: object = eqx.field(static=True)
    _frame: object = eqx.field(static=True)

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
        frame = self._frame_for_iterations(iterations)
        new_state = frame(self.state)
        return eqx.tree_at(lambda solver: solver.state, self, new_state)

    def step(self, iterations=None):
        """Advance this solver in place and return the new state."""
        object.__setattr__(self, "state", self.stepped(iterations).state)
        return self.state

    def solve(self, num_frames, on_frame=None):
        for f in range(num_frames):
            self.step()
            if on_frame is not None:
                on_frame(f, self.state)
        return self.state

    def reset(self, init_state):
        object.__setattr__(self, "state", init_state)
        return self.state

    def reset_to_initial(self):
        object.__setattr__(self, "state", self._init_state)
        return self.state
