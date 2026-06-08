"""Register postprocessing Hydra config choices at import time."""

from hydra_zen import store

import postprocessing.scaling_plots  # noqa: F401 - registers plot configs

store.add_to_hydra_store(overwrite_ok=True)
