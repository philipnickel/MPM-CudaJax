"""Focused P2G targets for Nsight Python and Nsight Compute."""

from collections.abc import Callable
from dataclasses import dataclass

import jax


NVTX_DOMAIN = "mpm_cudajax"
PROFILE_TARGETS = ("frame", "p2g", "prepare", "scatter")


def block_until_ready(value):
    """Synchronize all JAX arrays in a pytree and return the original value."""
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


@dataclass(frozen=True)
class P2GProfileTarget:
    """A warmed host callable that launches one profiling target."""

    name: str
    backend_name: str
    annotation: str
    run: Callable[[], object]


@dataclass(frozen=True)
class _PreparedTargets:
    backend_name: str
    state: object
    frame: Callable
    p2g: Callable
    prepare: Callable
    scatter: Callable


def _target_functions(solver):
    params = solver.params
    backend = solver.backend
    elasticity_fn = solver.elasticity_fn
    state = solver.state

    @jax.jit
    def prepare(state):
        stress = elasticity_fn(state.F)
        return backend.prepare(params, state, stress)

    @jax.jit
    def scatter(prepared):
        return backend.scatter(params, prepared)

    @jax.jit
    def p2g(state):
        stress = elasticity_fn(state.F)
        prepared = backend.prepare(params, state, stress)
        return backend.scatter(params, prepared)

    return _PreparedTargets(
        backend_name=backend.name,
        state=state,
        frame=solver._frame,
        p2g=p2g,
        prepare=prepare,
        scatter=scatter,
    )


def build_profile_target(solver, target_name):
    """Build one warmed profiling target by name."""
    if target_name not in PROFILE_TARGETS:
        names = ", ".join(PROFILE_TARGETS)
        raise ValueError(f"Unknown profile target {target_name!r}; choose one of {names}.")
    targets = _target_functions(solver)
    prepared = (
        block_until_ready(targets.prepare(targets.state))
        if target_name == "scatter"
        else None
    )
    calls = {
        "frame": lambda: targets.frame(targets.state),
        "p2g": lambda: targets.p2g(targets.state),
        "prepare": lambda: targets.prepare(targets.state),
        "scatter": lambda: targets.scatter(prepared),
    }
    call = calls[target_name]
    block_until_ready(call())
    return P2GProfileTarget(
        name=target_name,
        backend_name=targets.backend_name,
        annotation=f"{targets.backend_name}_{target_name}",
        run=call,
    )
