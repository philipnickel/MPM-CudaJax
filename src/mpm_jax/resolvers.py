"""OmegaConf resolvers shared between ``simulate.py`` and the sweep configs.

Importing this module registers the resolvers. Resolvers are global on the
OmegaConf level, so a single import anywhere is enough.

Currently registered:

``ppc_grid``  Maps an N (particle count) to G (grid resolution, rounded to a
              multiple of 4) such that the active-cell PPC stays near a target
              value. Used by ``conf/sweep/weak_scaling.yaml`` to derive G from
              the swept N without needing a per-point scale yaml.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def _ppc_grid(
    n_particles: int,
    target_ppc: float = 9.31,
    region_volume: float = 0.512,
) -> int:
    """Return G (multiple of 4) so N / (region_volume * G^3) ~= target_ppc.

    Defaults match the benchmark preset: region [0.1, 0.9]^3 has volume
    0.8^3 = 0.512, and the reference point N=10M / G=128 sits at PPC=9.31."""
    g = (n_particles / target_ppc / region_volume) ** (1 / 3)
    return max(4, 4 * round(g / 4))


OmegaConf.register_new_resolver("ppc_grid", _ppc_grid, replace=True)
