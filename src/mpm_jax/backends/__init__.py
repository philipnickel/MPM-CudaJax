"""Backend implementations registered as Hydra config-group choices."""

from hydra.core.config_store import ConfigStore
from hydra_zen import store
from omegaconf import OmegaConf

from mpm_jax.backends.common import BaseBackend, PreparedSubstep
from mpm_jax.backends.cuda import (
    CudaP2GBackend,
    CudaV1Backend,
    CudaV2Backend,
    CudaV3Backend,
    CudaV4Backend,
)
from mpm_jax.backends.cutile import CutileV1Backend, CutileV2Backend
from mpm_jax.backends.jax import JaxBackend

store.add_to_hydra_store(overwrite_ok=True)


def backend_choices() -> tuple[str, ...]:
    """Return registered Hydra backend config choices."""
    names = ConfigStore.instance().list("backend")
    return tuple(sorted(name.removesuffix(".yaml") for name in names))


def backend_config(choice: str):
    """Return a copy of a registered backend config by Hydra choice name."""
    try:
        node = ConfigStore.instance().load(f"backend/{choice}.yaml").node
    except (KeyError, FileNotFoundError) as exc:
        supported = ", ".join(backend_choices())
        raise RuntimeError(
            f"Unsupported backend choice {choice!r}. "
            f"Use Hydra backend choices: {supported}"
        ) from exc
    return OmegaConf.create(OmegaConf.to_container(node, resolve=False))


__all__ = [
    "BaseBackend",
    "CudaP2GBackend",
    "CudaV1Backend",
    "CudaV2Backend",
    "CudaV3Backend",
    "CudaV4Backend",
    "CutileV1Backend",
    "CutileV2Backend",
    "JaxBackend",
    "PreparedSubstep",
    "backend_choices",
    "backend_config",
]
