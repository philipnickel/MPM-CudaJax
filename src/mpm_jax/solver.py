import jax
import jax.numpy as jnp

from mpm_jax.types import (
    MPMState, StepIntermediates, MPMParams, OFFSET_27, make_params,
)
from mpm_jax.blocks.weights import compute_weights_and_indices
from mpm_jax.blocks.p2g import p2g_compute, p2g_scatter, p2g
from mpm_jax.blocks.g2p import g2p
from mpm_jax.blocks.grid import grid_update
from mpm_jax.stepping.substep import step
from mpm_jax.stepping.jax_frames import build_jax_frame as build_jit_frame


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
# Trade-off vs build_jit_frame: extra Python dispatch + 2 sync points per
# substep, but enables per-stage timing and clean CUDA interop without
# baking the boundary into the traced graph.

def build_jit_stages(params, elasticity_fn, plasticity_fn,
                     pre_particle_fn, post_grid_fn, p2g_fn=None):
    """Build three individually-JIT'd stage functions.

    Returns:
        (jit_p2g_stage, jit_grid_stage, jit_g2p_stage)

    Signatures (time fixed at 0.0 internally to match build_jit_frame —
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


class MPMSolver:
    """Stateful shell over the functional JAX core.

    Builds one jit'd pure frame function at construction; step()/solve() only
    call it. `self` is never traced.
    """

    def __init__(self, params, *, elasticity_fn, plasticity_fn,
                 pre_fn, post_fn, build_frame, steps_per_frame, init_state,
                 **frame_opts):
        self.params = params
        self.steps_per_frame = steps_per_frame
        self._init_state = init_state
        self.state = init_state
        self._frame = build_frame(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn,
            steps_per_frame, **frame_opts,
        )

    def step(self):
        self.state = self._frame(self.state)
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
