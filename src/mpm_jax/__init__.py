"""Public package exports, loaded lazily to avoid eager heavy imports."""

_SOLVER_EXPORTS = {
    "MPMState",
    "MPMParams",
    "make_params",
}
_CONSTITUTIVE_EXPORTS = {"get_constitutive", "ELASTICITY", "PLASTICITY"}
_BOUNDARY_EXPORTS = {"build_boundary_fns"}

__all__ = sorted(_SOLVER_EXPORTS | _CONSTITUTIVE_EXPORTS | _BOUNDARY_EXPORTS)


def __getattr__(name):
    if name in _SOLVER_EXPORTS:
        from . import solver

        value = getattr(solver, name)
    elif name in _CONSTITUTIVE_EXPORTS:
        from . import constitutive

        value = getattr(constitutive, name)
    elif name in _BOUNDARY_EXPORTS:
        from . import boundary

        value = getattr(boundary, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
