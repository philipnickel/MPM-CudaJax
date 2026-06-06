import time
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from mpm_jax.p2g.backends.common import g2p_mls
from mpm_jax.grid import build_grid_x, grid_update
from mpm_jax.types import MPMParams, MPMState


def get_particles(
    n_particles, center, size
):  # says uniformly but uses random? also why is it not just done directly in jax arrays?
    """Sample n_particles uniformly in a box."""
    start = np.array(center, dtype=np.float32) - np.array(size, dtype=np.float32) / 2
    end = np.array(center, dtype=np.float32) + np.array(size, dtype=np.float32) / 2
    rng = np.random.RandomState(42)
    return (start + rng.rand(n_particles, 3).astype(np.float32) * (end - start)).astype(
        np.float32, copy=False
    )


def _instantiate_target(value):  # this seems redundant
    if isinstance(value, dict | DictConfig) and "_target_" in value:
        return instantiate(value)
    return value


def _target_name(value):  # this seems hacky and redundant
    if isinstance(value, dict | DictConfig) and "_target_" in value:
        return str(value["_target_"]).rsplit(".", maxsplit=1)[-1]
    return getattr(value, "__name__", type(value).__name__)


def _gpu_type():  # we can assume its looking for a cuda device....  and then just do inline instead of having a method for it
    for device in jax.devices():
        if device.platform in {"cuda", "gpu"}:
            return getattr(device, "device_kind", str(device))
    return getattr(jax.devices()[0], "device_kind", jax.default_backend())


def _sticky_floor_mask(grid_x, dx):  # can be done inline
    return grid_x[:, 2] * dx < 0.02


def _apply_sticky_floor(grid_v, sticky_floor):  # can be done inline
    return jnp.where(sticky_floor[:, None], 0.0, grid_v)


def _backend_substep(_, state, *, params, elasticity_fn, backend, sticky_floor):
    with jax.named_scope("elasticity"):
        stress = elasticity_fn(state.F)

    with jax.named_scope(f"{backend.name}_p2g"):
        prepared = backend.prepare(params, state, stress)
        grid_mv, grid_m = backend.scatter(params, prepared)

    with jax.named_scope("grid_update"):
        grid_mv = grid_update(
            grid_mv, grid_m, params.gravity, params.dt, params.damping
        )
        grid_v = _apply_sticky_floor(grid_mv, sticky_floor)

    with jax.named_scope(
        "g2p"
    ):  # do we even have multiple g2p backends?
        new_x, new_v, new_C, new_F = g2p_mls(params, prepared, grid_v)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)


def _run_backend_frame(
    state, *, params, elasticity_fn, backend, sticky_floor, steps_per_frame
):  # yet another wrapper? this is getting ridiculous. why not just have the fori_loop directly in the frame?
    substep = partial(
        _backend_substep,
        params=params,
        elasticity_fn=elasticity_fn,
        backend=backend,
        sticky_floor=sticky_floor,
    )
    return jax.lax.fori_loop(0, steps_per_frame, substep, state)


def build_backend_frame(
    params, elasticity_fn, backend, steps_per_frame
):  # this seems just like a wrapper?
    """Build one JIT-compiled frame from a backend object.

    The frame owns the common MPM control flow (elasticity, grid update, sticky
    floor, the substep loop). The backend owns only the
    P2G — ``backend.prepare`` orders particles and ``backend.scatter`` scatters.
    The ``steps_per_frame`` substeps run as a single ``lax.fori_loop``.
    """
    sticky_floor = _sticky_floor_mask(build_grid_x(params.num_grids), params.dx)
    return jax.jit(
        partial(
            _run_backend_frame,
            params=params,
            elasticity_fn=elasticity_fn,
            backend=backend,
            sticky_floor=sticky_floor,
            steps_per_frame=steps_per_frame,
        )
    )


@dataclass
class RuntimeConfig:
    """Hydra-instantiated runtime config for constructing an MPMSolver."""

    material: Any
    sim: Any
    backend: Any


class MPMSolver:
    """Stateful driver over the functional JAX core.

    A plain Python object: array state (`state`) is mutated in place by the
    driver API, while the backend object, constitutive closure, sticky-floor
    mask, and the compiled `_frame` are fixed for the solver's lifetime. The
    solver itself is never a JAX argument — only `state` (an `MPMState` pytree)
    is traced; the backend/fns are baked into `_frame`'s closure at build time.
    So no pytree machinery is needed here.
    """

    def __init__(self, config):
        """Build a solver from an instantiated runtime config.

        Reads each config section and constructs the pieces — params (with its
        derived dx/vol/p_mass), particles, the target-instantiated backend,
        and the initial state — then hands them to the compiled frame.
        """
        sim, mat = config.sim, config.material
        n = int(sim.n_particles)

        params = MPMParams(sim)
        backend = _instantiate_target(config.backend)
        backend.validate_num_grids(params.num_grids)
        particles = jnp.asarray(
            get_particles(n, center=list(sim.center), size=list(sim.size)),
            dtype=jnp.float32,
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
        self.num_frames = int(getattr(sim, "num_frames", 1))
        self.steps_per_frame = int(sim.steps_per_frame)
        self._init_state = init_state
        self.state = init_state
        self.material_elasticity = _target_name(mat.elasticity)
        self.gpu_type = str(_gpu_type())
        self.elasticity_fn = _instantiate_target(mat.elasticity)
        if not callable(self.elasticity_fn):
            raise TypeError(
                "material.elasticity must be a Hydra-instantiated callable. "
                "Use an `_target_` in conf/material/<name>.yaml."
            )
        self.backend = backend
        self._frame = build_backend_frame(
            params,
            self.elasticity_fn,
            backend,
            self.steps_per_frame,
        )

    def step(self):  # why have both step, _frame, build_backend_frame?
        """Advance this solver in place and return the new state."""
        self.state = self._frame(self.state)
        return self.state

    def run(self, *, capture_frames, progress=True):
        self.step()
        jax.block_until_ready(self.state.x)
        self.reset_to_initial()

        frames = []
        frame_iter = range(self.num_frames)
        if progress:  # we will always have progress...
            frame_iter = tqdm(frame_iter, desc="simulate")

        t0 = time.perf_counter()
        for _ in frame_iter:
            if capture_frames:
                frames.append(
                    np.array(self.state.x)
                )  # hmm shouldn't this be device arrays optimally?
            self.step()
            if capture_frames:
                jax.block_until_ready(self.state.x)
        if not capture_frames:
            jax.block_until_ready(self.state.x)

        return frames, time.perf_counter() - t0

    def metrics(
        self, elapsed_s
    ):  # this is ugly. metrics should have a dataclass container instead.
        elapsed_s = float(elapsed_s)
        total_steps = self.num_frames * self.steps_per_frame
        particle_steps = self.params.n_particles * total_steps
        return {
            "kernel": self.backend.name,
            "material_elasticity": self.material_elasticity,
            "n_particles": self.params.n_particles,
            "num_grids": self.params.num_grids,
            "num_frames": self.num_frames,
            "steps_per_frame": self.steps_per_frame,
            "total_steps": total_steps,
            "elapsed_s": elapsed_s,
            "ms_per_frame": elapsed_s / self.num_frames * 1000,
            "ms_per_step": elapsed_s / total_steps * 1000,
            "steps_per_sec": total_steps / elapsed_s,
            "particles_per_sec": particle_steps / elapsed_s,
            "gpu_type": self.gpu_type,
        }

    def reset_to_initial(self):  # this is unneccessary to have as a method
        self.state = self._init_state
        return self.state
