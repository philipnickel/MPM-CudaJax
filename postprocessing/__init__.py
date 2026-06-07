"""Post-processing / analysis tooling (not part of the solver runtime).

Importing this package triggers the hydra-zen ``@store`` calls in
:mod:`postprocessing.scaling_plots` and commits the resulting configs to
Hydra's ConfigStore. ``simulate.py`` imports ``postprocessing`` once before
``@hydra.main`` runs so the ``plot`` config group is live at compose time.
"""

from hydra_zen import store

import postprocessing.scaling_plots  # noqa: F401 - triggers @store decorators

store.add_to_hydra_store(overwrite_ok=True)
