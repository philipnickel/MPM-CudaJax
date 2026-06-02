from mpm_jax.warp_graph import WarpBonusSimulator
from mpm_jax.solver import WarpGraphSolver


def build_warp_graph(cfg, *, particles, indexed_sort=False, **_ignored):
    """Construct a pure-Warp capture/replay solver.

    `cfg` is the resolved Hydra config; `particles` is the (N, 3) numpy init.
    """
    n = int(cfg.sim.n_particles)
    precompute_stress = not (indexed_sort and n >= 150_000_000)
    engine = WarpBonusSimulator(particles, cfg, indexed_sort=indexed_sort,
                                precompute_stress=precompute_stress)
    engine.warmup()
    return WarpGraphSolver(engine)
