import copy
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.grid import build_grid_x, grid_update
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.types import MPMParams, MPMState


def get_particles(n_particles, center, size):
    """Sample n_particles uniformly in a box."""
    start = np.array(center, dtype=np.float32) - np.array(size, dtype=np.float32) / 2
    end = np.array(center, dtype=np.float32) + np.array(size, dtype=np.float32) / 2
    rng = np.random.RandomState(42)
    return (start + rng.rand(n_particles, 3).astype(np.float32) * (end - start)).astype(
        np.float32, copy=False
    )


def build_backend_frame(
    params, elasticity_fn, pre_fn, post_fn, backend, steps_per_frame
):
    """Build one JIT-compiled frame from a backend object.

    The frame owns the common MPM control flow (boundary conditions, elasticity,
    grid update, the substep loop). The backend owns only the
    P2G — ``backend.step`` orders the particles and scatters — and ``g2p``.
    The ``steps_per_frame`` substeps run as a single ``lax.fori_loop``.
    """

    @jax.jit
    def jit_frame(state):
        def step_body(state):
            with jax.named_scope("pre_particle"):
                x, v = pre_fn(state.x, state.v, 0.0)
            state = state._replace(x=x, v=v)

            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope(f"{backend.name}_p2g"):
                prepared, grid_mv, grid_m = backend.step(params, state, stress)

            with jax.named_scope("grid_update"):
                grid_mv = grid_update(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping
                )
                grid_v = post_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope(f"{backend.name}_g2p"):
                new_x, new_v, new_C, new_F = backend.g2p(params, prepared, grid_v)

            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

        return jax.lax.fori_loop(0, steps_per_frame, lambda _, s: step_body(s), state)

    return jit_frame


@dataclass
class RuntimeConfig:
    """Hydra-instantiated runtime config for constructing an MPMSolver."""

    material: Any
    sim: Any
    backend: Any


class MPMSolver:
    """Stateful driver over the functional JAX core.

    A plain Python object: array state (`state`) is mutated in place by the
    driver API, while the backend object, constitutive/boundary closures, and
    the compiled `_frame` are fixed for the solver's lifetime. The solver itself
    is never a JAX argument — only `state` (an `MPMState` pytree) is traced; the
    backend/fns are baked into `_frame`'s closure at build time. So no pytree
    machinery is needed here.
    """

    def __init__(self, config):
        """Build a solver from an instantiated runtime config.

        Reads each config section and constructs the pieces — params (with its
        derived dx/vol/p_mass), particles, the target-instantiated backend,
        the boundary closures, and the initial state — then hands them to
        the compiled frame. The shared scalars (``n_particles``/``num_grids``)
        are read once and threaded as locals (``n``/``g``).
        """
        sim, mat = config.sim, config.material
        n, g = int(sim.n_particles), int(sim.num_grids)

        params = MPMParams(sim)
        backend = config.backend
        backend.validate_num_grids(params.num_grids)
        particles = jnp.asarray(
            get_particles(n, center=list(sim.center), size=list(sim.size)),
            dtype=jnp.float32,
        )
        pre_fn, post_fn = build_boundary_fns(
            list(sim.boundary_conditions),
            build_grid_x(g),
            params.dx,
        )
        init_state = MPMState(
            x=particles,
            v=jnp.broadcast_to(
                jnp.asarray(list(sim.initial_velocity), dtype=jnp.float32), (n, 3)
            ).copy(),
            C=jnp.zeros((n, 3, 3)),
            F=jnp.tile(jnp.eye(3), (n, 1, 1)),
        )
        self.params = params
        self.steps_per_frame = int(sim.steps_per_frame)
        self._init_state = init_state
        self.state = init_state
        self.elasticity_fn = mat.elasticity
        if not callable(self.elasticity_fn):
            raise TypeError(
                "material.elasticity must be a Hydra-instantiated callable. "
                "Use an `_target_` in conf/material/<name>.yaml."
            )
        self.pre_fn = pre_fn
        self.post_fn = post_fn
        self.backend = backend
        self._frame = build_backend_frame(
            params,
            self.elasticity_fn,
            pre_fn,
            post_fn,
            backend,
            self.steps_per_frame,
        )

    def stepped(self):
        """Return a new solver with state advanced one frame (steps_per_frame)."""
        new_state = self._frame(self.state)
        new = copy.copy(self)  # shallow: shares _frame + backend + closures
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
