"""Post-processing / analysis tooling (not part of the solver runtime).

A top-level package, separate from the installed ``mpm_jax`` solver: it imports
only the plotting stack (matplotlib/pandas/seaborn), never the solver core.
Resolved by Hydra ``_target_`` strings (e.g. ``postprocessing.callbacks.
ScalingPlotCallback``) because the repo root is on ``sys.path`` when ``simulate.py``
is run from there.
"""
