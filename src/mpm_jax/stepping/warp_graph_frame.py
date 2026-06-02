from mpm_jax.warp_graph import WarpBonusSimulator
from mpm_jax.solver import WarpGraphSolver


def build_warp_graph(cfg, *, particles, indexed_sort=False, baseline=False, **_ignored):
    """Construct a pure-Warp capture/replay solver.

    `cfg` is the resolved Hydra config; `particles` is the (N, 3) numpy init.

    Calls `engine.warmup()`, which captures and launches the CUDA graph
    (a GPU side-effect) before returning.
    """
    kernel_name = cfg.get('kernel', {}).get('name', 'warp_bonus_graph')
    elasticity = cfg.get('material', {}).get('elasticity', {}).get('name', None)
    plasticity = cfg.get('material', {}).get('plasticity', {}).get('name', None)
    if elasticity != 'CorotatedElasticityJacobi' or plasticity != 'IdentityPlasticity':
        raise RuntimeError(
            f"kernel={kernel_name} currently supports material=jelly_jacobi "
            "only: CorotatedElasticityJacobi + IdentityPlasticity."
        )
    if (not baseline) and int(cfg.sim.num_grids) % 2 != 0:
        raise RuntimeError(f"kernel={kernel_name} requires sim.num_grids divisible by 2.")

    n = int(cfg.sim.n_particles)
    precompute_stress = not (indexed_sort and n >= 150_000_000)
    engine = WarpBonusSimulator(particles, cfg, indexed_sort=indexed_sort,
                                precompute_stress=precompute_stress, baseline=baseline)
    engine.warmup()
    return WarpGraphSolver(engine)
