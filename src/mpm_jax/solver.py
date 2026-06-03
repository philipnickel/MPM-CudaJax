import jax
import equinox as eqx

from mpm_jax.types import (
    MPMState, StepIntermediates,
    MPMParams, make_params,  # noqa: F401  (re-exported via __init__._SOLVER_EXPORTS)
)
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.p2g import p2g_compute, p2g_scatter, p2g  # noqa: F401  (p2g_compute, p2g_scatter re-exported)
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update
from mpm_jax.backends import build_backend_frame


def step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=None):
    """One full pure-JAX P2G2P step.

    This is kept for tests and per-stage profiling helpers. The production
    solver path uses ``build_backend_frame`` with a backend object.
    """
    p2g_fn = p2g_fn or p2g

    with jax.named_scope("pre_particle"):
        x, v = pre_particle_fn(state.x, state.v, time)

    with jax.named_scope("weights"):
        weight, dweight, dpos, index = compute_weights_and_indices(
            x, params.inv_dx, params.dx, params.num_grids)

    with jax.named_scope("p2g"):
        grid_mv, grid_m = p2g_fn(
            v, state.C, stress, weight, dweight, dpos, index,
            params.dt, params.vol, params.p_mass, params.num_grids)

    with jax.named_scope("grid_update"):
        grid_mv = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)

    with jax.named_scope("post_grid"):
        grid_mv = post_grid_fn(grid_mv, grid_m, time)

    with jax.named_scope("g2p"):
        new_x, new_v, new_C, new_F = g2p(
            grid_mv, weight, dweight, dpos, index,
            state.F, x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)


def build_jit_step(params, elasticity_fn, plasticity_fn,
                   pre_particle_fn, post_grid_fn, p2g_fn=None):
    """Build a JIT-compiled single-step function.

    Captures all closures at trace time so the entire timestep compiles
    to one XLA program.
    """
    _p2g_fn = p2g_fn or p2g

    @jax.jit
    def jit_step(state):
        with jax.named_scope("elasticity"):
            stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_particle_fn, post_grid_fn,
                     0.0, p2g_fn=_p2g_fn)
        with jax.named_scope("plasticity"):
            return state._replace(F=plasticity_fn(state.F))

    return jit_step


def simulate_frame(params, state, elasticity_fn, plasticity_fn,
                   pre_particle_fn, post_grid_fn, steps_per_frame, time, p2g_fn=None):
    """Run multiple substeps for one frame (unjitted, for per-stage profiling)."""
    for _ in range(steps_per_frame):
        stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=p2g_fn)
        state = state._replace(F=plasticity_fn(state.F))
        time += params.dt
    return state, time


# ---------------------------------------------------------------------------
# Per-stage JIT path: one JIT per stage, host-side loop drives them.
# ---------------------------------------------------------------------------
#
# Trade-off vs the monolithic backend frame: extra Python
# dispatch + 2 sync points per substep, but enables per-stage timing and
# clean CUDA interop without baking the boundary into the traced graph.

def build_jit_stages(params, elasticity_fn, plasticity_fn,
                     pre_particle_fn, post_grid_fn, p2g_fn=None):
    """Build three individually-JIT'd stage functions.

    Returns:
        (jit_p2g_stage, jit_grid_stage, jit_g2p_stage)

    Signatures (time fixed at 0.0 internally to match the backend frame —
    boundary conditions don't see substep time in the JIT'd path):
        jit_p2g_stage(state)             -> (grid_mv, grid_m, intermediates)
        jit_grid_stage(grid_mv, grid_m)  -> grid_v
        jit_g2p_stage(state, grid_v, intermediates) -> MPMState
    """
    _p2g_fn = p2g_fn or p2g

    @jax.jit
    def jit_p2g_stage(state):
        x, v = pre_particle_fn(state.x, state.v, 0.0)
        stress = elasticity_fn(state.F)
        weight, dweight, dpos, index = compute_weights_and_indices(
            x, params.inv_dx, params.dx, params.num_grids)
        grid_mv, grid_m = _p2g_fn(
            v, state.C, stress, weight, dweight, dpos, index,
            params.dt, params.vol, params.p_mass, params.num_grids)
        inter = StepIntermediates(x_post_bc=x, F_pre_plast=state.F)
        return grid_mv, grid_m, inter

    @jax.jit
    def jit_grid_stage(grid_mv, grid_m):
        grid_mv_normalized = grid_update(
            grid_mv, grid_m, params.gravity, params.dt, params.damping)
        grid_v = post_grid_fn(grid_mv_normalized, grid_m, 0.0)
        return grid_v

    @jax.jit
    def jit_g2p_stage(state, grid_v, inter):
        # Recompute weights/indices instead of carrying them across the JIT
        # boundary - see StepIntermediates docstring.
        weight, dweight, dpos, index = compute_weights_and_indices(
            inter.x_post_bc, params.inv_dx, params.dx, params.num_grids)
        new_x, new_v, new_C, new_F = g2p(
            grid_v, weight, dweight, dpos, index,
            inter.F_pre_plast, inter.x_post_bc,
            params.dt, params.inv_dx, params.clip_bound)
        new_F = plasticity_fn(new_F)
        return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

    return jit_p2g_stage, jit_grid_stage, jit_g2p_stage


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
