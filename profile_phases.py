"""Per-phase MLS-MPM timing — JIT each classical phase separately and time it.

The fused frame (one ``jit`` over the whole timestep) is what you ship, but XLA
fuses it into ~2 kernels and drops most ``named_scope`` labels, so a profiler
can't attribute time to P2G / Grid / G2P. This harness instead compiles **each
phase as its own module** and times it with ``block_until_ready`` — a clean
per-phase breakdown with no dependence on label propagation.

Caveat: isolating the phases prevents cross-phase fusion and forces the grid /
particle intermediates through HBM, so the per-phase times sum to MORE than the
real fused frame. This is an *isolated-cost* breakdown (the classical MPM
phases), not the optimized frame's internal split.

Run (8M standard benchmark config):

    pixi run -e gpu python profile_phases.py -cn config sim=benchmark \
        backend=jax_baseline material=jelly
"""

import time

import hydra
import jax
from omegaconf import DictConfig

from mpm_jax.backends import PreparedSubstep
from mpm_jax.blocks.grid import grid_update
from mpm_jax.solver import MPMSolver


def _ms_per_call(fn, args, k=50):
    """Compile once, then time k back-to-back calls (one sync), ms/call."""
    jax.block_until_ready(fn(*args))  # warmup / compile
    t0 = time.perf_counter()
    for _ in range(k):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / k * 1000.0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    solver = MPMSolver(hydra.utils.instantiate(cfg.solver))
    p, be = solver.params, solver.backend
    pre_fn, post_fn = solver.pre_fn, solver.post_fn
    elasticity_fn, plasticity_fn = solver.elasticity_fn, solver.plasticity_fn

    # Advance to a representative steady state (the fused frame), then freeze it.
    for _ in range(2):
        solver.step()
    state = solver.state
    jax.block_until_ready(state.x)

    # --- the 3 classical phases, each its own jitted module --------------------
    @jax.jit
    def p2g_phase(state):  # pre-particle BC + stress (StVK) + 27-offset scatter
        x, v = pre_fn(state.x, state.v, 0.0)
        state = state._replace(x=x, v=v)
        stress = elasticity_fn(state.F)
        grid_mv, grid_m = be.p2g(
            p, PreparedSubstep(state.x, state.v, state.C, state.F, stress)
        )
        return grid_mv, grid_m, stress, state

    @jax.jit
    def grid_phase(grid_mv, grid_m):  # normalize + gravity/damping + BC
        grid_mv = grid_update(grid_mv, grid_m, p.gravity, p.dt, p.damping)
        return post_fn(grid_mv, grid_m, 0.0)

    @jax.jit
    def g2p_phase(
        state, stress, grid_v
    ):  # weights + gather + F-update + advect + return-map
        prepared = be.prepare(p, state, stress)
        nx, nv, nC, nF = be.g2p(p, prepared, grid_v)
        return nx, nv, nC, nF, plasticity_fn(nF)

    # bonus: isolate the two expensive "physics" kernels inside the phases
    elast = jax.jit(elasticity_fn)
    plast = jax.jit(plasticity_fn)

    # produce each phase's representative input once
    grid_mv, grid_m, stress, state2 = p2g_phase(state)
    jax.block_until_ready((grid_mv, grid_m, stress))
    grid_v = grid_phase(grid_mv, grid_m)
    jax.block_until_ready(grid_v)

    K = 50
    t_p2g = _ms_per_call(p2g_phase, (state,), K)
    t_grid = _ms_per_call(grid_phase, (grid_mv, grid_m), K)
    t_g2p = _ms_per_call(g2p_phase, (state2, stress, grid_v), K)
    t_elast = _ms_per_call(elast, (state.F,), K)
    t_plast = _ms_per_call(plast, (state2.F,), K)
    total = t_p2g + t_grid + t_g2p

    print(
        f"\nPer-phase timing — {solver.backend.name}, {int(cfg.sim.n_particles):,} particles "
        f"(jit each phase separately, {K} iters, ms/substep):\n"
    )
    rows = [
        ("P2G  (BC + stress (StVK) + scatter)", t_p2g),
        ("Grid (normalize + gravity + BC)", t_grid),
        ("G2P  (weights + gather + F + advect + plast)", t_g2p),
    ]
    for name, t in rows:
        print(f"  {name:46} {t:8.3f} ms   {t / total * 100:5.1f}%")
    print(f"  {'-' * 46} {'-' * 8}")
    print(f"  {'sum (unfused phases)':46} {total:8.3f} ms")
    print("\n  of which (isolated):")
    print(f"    {'elasticity / stress (StVK)':44} {t_elast:8.3f} ms")
    print(f"    {'plasticity / return-mapping':44} {t_plast:8.3f} ms")


if __name__ == "__main__":
    main()
