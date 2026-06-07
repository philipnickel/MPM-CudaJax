import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import nvtx
import numpy as np
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from mpm_jax.p2g.backends.common import g2p_mls
from mpm_jax.grid import build_grid_x, grid_update
from mpm_jax.profiling import NVTX_DOMAIN, block_until_ready
from mpm_jax.types import MPMParams, MPMState


def get_particles(n_particles, center, size):
    """Sample n_particles uniformly in a box."""
    start = np.array(center, dtype=np.float32) - np.array(size, dtype=np.float32) / 2
    end = np.array(center, dtype=np.float32) + np.array(size, dtype=np.float32) / 2
    rng = np.random.RandomState(42)
    return (start + rng.rand(n_particles, 3).astype(np.float32) * (end - start)).astype(
        np.float32, copy=False
    )


def _elasticity_stage(state, elasticity_fn):
    with jax.named_scope("elasticity"):
        return elasticity_fn(state.F)


def _prepare_stage(params, backend, state, stress):
    with jax.named_scope(f"{backend.name}_prepare"):
        return backend.prepare(params, state, stress)


def _scatter_stage(params, backend, prepared):
    with jax.named_scope(f"{backend.name}_scatter"):
        return backend.scatter(params, prepared)


def _grid_update_stage(params, sticky_floor, grid_mv, grid_m):
    with jax.named_scope("grid_update"):
        grid_v = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)
        return jnp.where(sticky_floor[:, None], 0.0, grid_v)


def _g2p_stage(params, prepared, grid_v):
    with jax.named_scope("g2p"):
        return MPMState(*g2p_mls(params, prepared, grid_v))


class StagedStepFns(NamedTuple):
    elasticity: Any
    prepare: Any
    scatter: Any
    grid_update: Any
    g2p: Any


def build_staged_step(params, elasticity_fn, backend, sticky_floor):
    """Build per-stage JIT callables for one host-driven MPM substep."""
    @jax.jit
    def elasticity(state):
        return _elasticity_stage(state, elasticity_fn)

    @jax.jit
    def prepare(state, stress):
        return _prepare_stage(params, backend, state, stress)

    @jax.jit
    def scatter(prepared):
        return _scatter_stage(params, backend, prepared)

    @jax.jit
    def update_grid(grid_mv, grid_m):
        return _grid_update_stage(params, sticky_floor, grid_mv, grid_m)

    @jax.jit
    def g2p(prepared, grid_v):
        return _g2p_stage(params, prepared, grid_v)

    return StagedStepFns(
        elasticity=elasticity,
        prepare=prepare,
        scatter=scatter,
        grid_update=update_grid,
        g2p=g2p,
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
        backend_cfg = config.backend
        backend = (
            hydra_instantiate(backend_cfg)
            if isinstance(backend_cfg, dict | DictConfig) and "_target_" in backend_cfg
            else backend_cfg
        )
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
        devices = jax.devices()
        device = next(
            (device for device in devices if device.platform in {"cuda", "gpu"}),
            devices[0],
        )
        self.gpu_type = str(getattr(device, "device_kind", device))
        elasticity_cfg = mat.elasticity
        self.elasticity_fn = (
            hydra_instantiate(elasticity_cfg)
            if isinstance(elasticity_cfg, dict | DictConfig)
            and "_target_" in elasticity_cfg
            else elasticity_cfg
        )
        target = (
            elasticity_cfg.get("_target_", None)
            if isinstance(elasticity_cfg, dict | DictConfig)
            else None
        )
        self.material_elasticity = (
            str(target).rsplit(".", maxsplit=1)[-1]
            if target
            else getattr(self.elasticity_fn, "__name__", type(self.elasticity_fn).__name__)
        )
        self.backend = backend
        self._sticky_floor = build_grid_x(params.num_grids)[:, 2] * params.dx < 0.02
        self._stages = build_staged_step(
            params, self.elasticity_fn, backend, self._sticky_floor
        )

        @jax.jit
        def frame(state):
            return jax.lax.fori_loop(
                0,
                self.steps_per_frame,
                lambda _, carry: self.step_state(carry),
                state,
            )

        self._frame = frame

    def step_state(self, state):
        """Pure one-substep transition used by the jitted frame path."""
        stress = _elasticity_stage(state, self.elasticity_fn)
        prepared = _prepare_stage(self.params, self.backend, state, stress)
        grid_mv, grid_m = _scatter_stage(self.params, self.backend, prepared)
        grid_v = _grid_update_stage(self.params, self._sticky_floor, grid_mv, grid_m)
        return _g2p_stage(self.params, prepared, grid_v)

    def step(self, *, profile=False):
        """Advance one host-driven substep using individually jitted stages."""
        stages = self._stages
        with (
            nvtx.annotate(f"{self.backend.name}_elasticity", domain=NVTX_DOMAIN)
            if profile
            else nullcontext()
        ):
            stress = stages.elasticity(self.state)
            if profile:
                stress = block_until_ready(stress)

        with (
            nvtx.annotate(f"{self.backend.name}_prepare", domain=NVTX_DOMAIN)
            if profile
            else nullcontext()
        ):
            prepared = stages.prepare(self.state, stress)
            if profile:
                prepared = block_until_ready(prepared)

        with (
            nvtx.annotate(f"{self.backend.name}_scatter", domain=NVTX_DOMAIN)
            if profile
            else nullcontext()
        ):
            grid_mv, grid_m = stages.scatter(prepared)
            if profile:
                grid_mv, grid_m = block_until_ready((grid_mv, grid_m))

        with (
            nvtx.annotate(f"{self.backend.name}_grid_update", domain=NVTX_DOMAIN)
            if profile
            else nullcontext()
        ):
            grid_v = stages.grid_update(grid_mv, grid_m)
            if profile:
                grid_v = block_until_ready(grid_v)

        with (
            nvtx.annotate(f"{self.backend.name}_g2p", domain=NVTX_DOMAIN)
            if profile
            else nullcontext()
        ):
            self.state = stages.g2p(prepared, grid_v)
            if profile:
                self.state = block_until_ready(self.state)

        return self.state

    def warmup(self, n=1, *, staged=False):
        """Run ``n`` throwaway frames to trigger JIT compilation, then reset.

        A standalone step the caller invokes before timing/tracing — keeping it
        out of :meth:`run` means a profiler trace wrapping ``run`` never
        captures one-time compilation.
        """
        for _ in range(max(1, int(n))):
            if staged:
                self.step()
            else:
                self.state = self._frame(self.state)
        jax.block_until_ready(self.state.x)
        self.state = self._init_state
        return self.state

    def run(self, *, capture_frames, progress=True, staged=False, profile_stages=False):
        """Drive the frame loop and return ``(frames, elapsed_s)``.

        Each frame is wrapped in a ``StepTraceAnnotation`` so an enclosing
        profiler trace shows per-frame steps; it is a no-op when no trace is
        active. Call :meth:`warmup` first for clean timing.
        """
        frames = []
        frame_iter = range(self.num_frames)
        if progress:
            frame_iter = tqdm(frame_iter, desc="simulate")

        t0 = time.perf_counter()
        for i in frame_iter:
            if capture_frames:
                frames.append(np.array(self.state.x))
            with jax.profiler.StepTraceAnnotation("frame", step_num=i):
                if staged:
                    for _ in range(self.steps_per_frame):
                        self.step(profile=profile_stages)
                else:
                    self.state = self._frame(self.state)
        jax.block_until_ready(self.state.x)

        return frames, time.perf_counter() - t0

    def metrics(self, elapsed_s):
        elapsed_s = float(elapsed_s)
        total_steps = self.num_frames * self.steps_per_frame
        particle_steps = self.params.n_particles * total_steps
        grid_cells = self.params.num_grids**3
        material_volume = self.params.vol * self.params.n_particles
        active_grid_cells = material_volume / (self.params.dx**3)
        return {
            "kernel": self.backend.name,
            "material_elasticity": self.material_elasticity,
            "n_particles": self.params.n_particles,
            "num_grids": self.params.num_grids,
            "dx": self.params.dx,
            "grid_cells": grid_cells,
            "material_volume": material_volume,
            "active_grid_cells": active_grid_cells,
            "particles_per_grid_cell": self.params.n_particles / grid_cells,
            "particles_per_active_cell": self.params.n_particles / active_grid_cells,
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
